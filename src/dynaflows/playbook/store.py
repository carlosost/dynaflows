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


def replace_source(
    connection: sqlite3.Connection, path: Path, relative: str, chunks: Sequence[Chunk]
) -> None:
    """Reindex one file atomically. External-content FTS5 is kept in step by
    hand rather than by triggers -- one place to read, no hidden writes."""
    with connection:
        old = connection.execute(
            "SELECT rowid FROM chunks WHERE source_path = ?", (relative,)
        ).fetchall()
        for row in old:
            connection.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, anchors, body) "
                "VALUES ('delete', ?, '', '', '')",
                (row["rowid"],),
            )
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
                    json.dumps(list(chunk.anchors)),
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
            for row in connection.execute(
                "SELECT rowid FROM chunks WHERE source_path = ?", (relative,)
            ).fetchall():
                connection.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, anchors, body) "
                    "VALUES ('delete', ?, '', '', '')",
                    (row["rowid"],),
                )
            connection.execute("DELETE FROM chunks WHERE source_path = ?", (relative,))
            connection.execute("DELETE FROM sources WHERE path = ?", (relative,))


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
