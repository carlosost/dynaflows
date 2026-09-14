"""The FTS5 index. ADR-009, layers 1 and 2.

SQLite ships FTS5 with bm25() (verified by `dynaflows doctor` on every machine
that runs it), so the retrieval layer needs no dependency the checkpointer did
not already require. That is the entire argument against a vector store here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from dynaflows.contracts.playbook import Chunk, DriftReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    path        TEXT PRIMARY KEY,
    file_sha256 TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    source_path  TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    anchors      TEXT NOT NULL,
    tokens       INTEGER NOT NULL,
    body         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_path, ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading_path, anchors, body,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_SCHEMA)
    return connection


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_files(root: Path) -> list[Path]:
    """Every .md under root, in a stable order.

    Sorted because the index is a derived artifact and a derived artifact that
    changes when nothing changed cannot support a drift check.
    """
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        source_path=row["source_path"],
        heading_path=row["heading_path"],
        anchors=tuple(json.loads(row["anchors"])),
        body=row["body"],
        tokens=row["tokens"],
    )


def _fts_row(chunk_row: sqlite3.Row) -> tuple[str, str, str]:
    """The three column values exactly as they were indexed.

    `anchors` is stored as JSON in `chunks` and as a space-joined string in
    `chunks_fts`. Handing FTS5 the JSON on delete would subtract a different
    set of tokens than were added -- which fails the same way as the empty
    strings did, only harder to see. One function so the write and the unwrite
    cannot drift.
    """
    return (
        chunk_row["heading_path"],
        " ".join(json.loads(chunk_row["anchors"])),
        chunk_row["body"],
    )


def _unindex_source(connection: sqlite3.Connection, relative: str) -> None:
    """Remove one source's postings from the external-content FTS5 index.

    FTS5 `'delete'` SUBTRACTS the column values you pass; it does not look
    them up. This passed `('', '', '')`, which subtracts nothing, so every
    posting outlived its chunk. Nothing raised: the index simply grew, and
    `INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')` passes on
    the result -- verified.

    Today the surviving postings mostly point at rowids no chunk owns, so
    `search`'s JOIN drops them and the visible damage is a query that returns
    fewer rows than it should, plus a document-frequency count that makes
    bm25's IDF wrong for every affected term. The moment a rowid is REUSED --
    which is what happens as soon as one file is reindexed on its own -- the
    posting points at whatever now occupies it and the search returns the
    wrong document for a term the corpus no longer contains.

    Called before the rows are deleted from `chunks`, because the values it
    has to subtract live there.
    """
    for row in connection.execute(
        "SELECT rowid, heading_path, anchors, body FROM chunks WHERE source_path = ?",
        (relative,),
    ).fetchall():
        heading_path, anchors, body = _fts_row(row)
        connection.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, anchors, body) "
            "VALUES ('delete', ?, ?, ?, ?)",
            (row["rowid"], heading_path, anchors, body),
        )


def replace_source(
    connection: sqlite3.Connection, path: Path, relative: str, chunks: Sequence[Chunk]
) -> None:
    """Reindex one file atomically. External-content FTS5 is kept in step by
    hand rather than by triggers -- one place to read, no hidden writes."""
    with connection:
        _unindex_source(connection, relative)
        connection.execute("DELETE FROM chunks WHERE source_path = ?", (relative,))
        for ordinal, chunk in enumerate(chunks):
            cursor = connection.execute(
                "INSERT INTO chunks(id, source_path, ordinal, heading_path, anchors, tokens, body)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    relative,
                    ordinal,
                    chunk.heading_path,
                    # ensure_ascii=False, and it is load-bearing. The default
                    # escapes "§" to "\\u00a7", so `chunks.anchors` held
                    # "\\u00a73.3" while the FTS index was fed "§3.3" -- the two
                    # tokenize differently, and external-content FTS5 is
                    # DEFINED by the index matching the content table. Any
                    # FTS5 operation that re-derives from content (`rebuild`)
                    # therefore built a different index than incremental
                    # indexing did, and `integrity-check, 1` could never pass,
                    # which is why nothing could detect the delete bug.
                    json.dumps(list(chunk.anchors), ensure_ascii=False),
                    chunk.tokens,
                    chunk.body,
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts(rowid, heading_path, anchors, body) VALUES (?, ?, ?, ?)",
                (
                    cursor.lastrowid,
                    chunk.heading_path,
                    " ".join(chunk.anchors),
                    chunk.body,
                ),
            )
        connection.execute(
            "INSERT INTO sources(path, file_sha256, indexed_at) VALUES (?, ?, ?)"
            " ON CONFLICT(path) DO UPDATE SET file_sha256=excluded.file_sha256,"
            " indexed_at=excluded.indexed_at",
            (relative, file_sha256(path), datetime.now(UTC).isoformat()),
        )


def forget_sources(connection: sqlite3.Connection, relatives: Iterable[str]) -> None:
    with connection:
        for relative in relatives:
            _unindex_source(connection, relative)
            connection.execute("DELETE FROM chunks WHERE source_path = ?", (relative,))
            connection.execute("DELETE FROM sources WHERE path = ?", (relative,))


def fts_is_consistent(connection: sqlite3.Connection) -> bool:
    """Does the search index still agree with the chunks it claims to index?

    The drift check answers "is the index stale?". This answers "is the index
    a lie?" -- a different fault with a different remedy, so it gets its own
    check rather than being folded into drift (AP-20).

    The obvious implementation is wrong and was written first: an anti-join
    against `chunks_fts` finds nothing, because querying an external-content
    FTS5 table reads its rowids from the CONTENT table. It reports what the
    chunks say, not what the index holds, so a corrupt index looks perfect.

    `integrity-check` takes an argument, and the default is not the one we
    need: rank 0 checks the index against itself and passes over this fault,
    which is why the bug survived so long. Rank 1 cross-checks the index
    against the content table and raises `DatabaseError` when they disagree.
    """
    try:
        connection.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)")
    except sqlite3.DatabaseError:
        return False
    return True


def rebuild_fts(connection: sqlite3.Connection) -> None:
    """Re-derive the whole FTS index from the content table.

    The repair for a database indexed before the delete fix. Those postings
    have no chunk row left, so nothing can subtract them one at a time --
    reconstruction is the only remedy, and FTS5 supports it directly.
    """
    with connection:
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


def drift(connection: sqlite3.Connection, root: Path) -> DriftReport:
    """Compare recorded hashes against the corpus on disk.

    AP-19 habit 2: a generated artifact plus a drift check makes staleness
    impossible rather than unlikely. Three outcomes, kept apart (AP-20).
    """
    on_disk = {str(p.relative_to(root)): file_sha256(p) for p in markdown_files(root)}
    indexed = {
        row["path"]: row["file_sha256"]
        for row in connection.execute("SELECT path, file_sha256 FROM sources")
    }
    return DriftReport(
        changed=tuple(sorted(p for p, h in on_disk.items() if p in indexed and indexed[p] != h)),
        added=tuple(sorted(p for p in on_disk if p not in indexed)),
        removed=tuple(sorted(p for p in indexed if p not in on_disk)),
    )
