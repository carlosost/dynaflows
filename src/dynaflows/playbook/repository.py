"""PlaybookRepository -- the retrieval interface. ADR-009.

Three layers, in priority order:
  1. by_anchor  -- exact, deterministic, zero LLM. The PRIMARY path.
  2. search     -- BM25 over FTS5. The fallback, and the way to find mentions.
  3. catalog    -- what the planner reads in order to name anchors at all.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.playbook import Chunk, DriftReport
from dynaflows.playbook import store
from dynaflows.playbook.chunker import chunk_markdown
from dynaflows.playbook.tokens import estimate_tokens

# FTS5 has its own query syntax: bare `-`, `"` or `*` in a planner's phrase is
# a syntax error, not a search. Everything from outside is reduced to quoted
# terms -- the same rule that Rich taught us, in a different parser.
_TERM_RE = re.compile(r"[A-Za-z0-9§._-]+")
_SUMMARY_RE = re.compile(r"\s+")


class PlaybookRepository(Protocol):
    def by_anchor(self, anchors: list[str]) -> list[Chunk]: ...
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...
    def section_map(self, budget_tokens: int) -> "SectionMap": ...
    def drift(self) -> DriftReport: ...
    def count(self) -> int: ...


def _fts_query(text: str) -> str:
    terms = [t for t in _TERM_RE.findall(text) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms)


_FORMAL_RE = re.compile(r"^(AP-|ADR-|§)")


def _summary(chunk: Chunk, limit: int = 90) -> str:
    first = _SUMMARY_RE.sub(" ", chunk.body.split("\n\n", 1)[0]).strip()
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def section_line(chunk: Chunk) -> str:
    """One catalogue row: how to name this section, where it sits, what it says.

    Every byte here is paid for on EVERY planner call, so two economies are
    deliberate. Only formal anchors are listed -- a slug restates the heading
    that is printed two columns over, so listing both bills twice for one fact.
    And the heading path is trimmed to its last two levels: the document title
    and top-level part are constant across most rows and carry no signal.
    """
    formal = [a for a in chunk.anchors if _FORMAL_RE.match(a)]
    names = " ".join(formal) if formal else chunk.anchors[0] if chunk.anchors else "-"
    path = " > ".join(chunk.heading_path.split(" > ")[-2:])
    return f"{names} | {path} | {_summary(chunk)}"


def _has_formal_anchor(chunk: Chunk) -> bool:
    return any(_FORMAL_RE.match(a) for a in chunk.anchors)


@dataclass(frozen=True, slots=True)
class SectionMap:
    """The playbook catalogue actually sent, and how much of it that is.

    Mirrors `store.source_map.SourceMap` on purpose: same shape, same reason.
    A planner shown a partial catalogue that looks complete will cite an
    anchor it cannot see and route work against a section that was dropped.
    """

    text: str
    listed: int
    total: int

    @property
    def truncated(self) -> bool:
        return self.listed < self.total

    def render(self) -> str:
        """What goes in the prompt. Truncation is stated, not implied."""
        if not self.total:
            return "(no playbook sections indexed)"
        if not self.truncated:
            return self.text
        return (
            f"{self.text}\n\n"
            f"... TRUNCATED: {self.listed} of {self.total} sections listed. The rest "
            f"were omitted for space. Do NOT assume a section is absent because it is "
            f"missing from this list; cite only anchors you can see here."
        )


def _pack_section_map(chunks: Sequence[Chunk], budget_tokens: int) -> SectionMap:
    """Fill `budget_tokens` with catalogue rows.

    Filtering to formal anchors alone only removes ~14% of rows (90 of 105 on
    this corpus) -- not enough to be the lever that gets a catalogue under a
    cap. It is used here only to decide WHICH rows survive a cut: ADR-009's
    planner can only cite an anchor this catalogue still prints, so a row with
    one is worth more than a row without when something has to go. The actual
    size is set by row COUNT and row LENGTH (`_summary`'s 90-character cap),
    and both are paid on every planner call regardless of which rows survive.

    Deterministic, like `store.source_map.build_source_map`: once a row does
    not fit, nothing after it is considered, even a smaller one further down.
    Survivors are then rendered back in their original document order.
    """
    total = len(chunks)
    order = sorted(range(total), key=lambda i: 0 if _has_formal_anchor(chunks[i]) else 1)
    survive: list[int] = []
    used = 0
    exhausted = False
    for i in order:
        if exhausted:
            continue
        line = section_line(chunks[i])
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens:
            exhausted = True
            continue
        used += cost
        survive.append(i)
    kept = sorted(survive)
    return SectionMap(
        text="\n".join(section_line(chunks[i]) for i in kept),
        listed=len(kept),
        total=total,
    )


class SqlitePlaybookRepository:
    """Reads the index. Never builds it -- `index_corpus` does that."""

    def __init__(self, connection: sqlite3.Connection, root: Path) -> None:
        self._connection = connection
        self._root = root

    def by_anchor(self, anchors: list[str]) -> list[Chunk]:
        """Exact lookup, returned in the order requested.

        Order matters: pack() fills the budget in the order it is handed, so
        the caller's priority has to survive the round trip. A slug shared by
        two documents returns both -- an ambiguous request gets an honest
        ambiguous answer rather than an arbitrary pick.
        """
        out: list[Chunk] = []
        seen: set[str] = set()
        for anchor in anchors:
            rows = self._connection.execute(
                "SELECT c.* FROM chunks c JOIN json_each(c.anchors) a ON a.value = ?"
                " ORDER BY c.source_path, c.ordinal",
                (anchor,),
            ).fetchall()
            for row in rows:
                chunk = store.row_to_chunk(row)
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    out.append(chunk)
        return out

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        match = _fts_query(query)
        if not match:
            return []
        rows = self._connection.execute(
            "SELECT c.* FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid"
            " WHERE chunks_fts MATCH ?"
            # anchors weigh most, then the heading, then the body: a section
            # ABOUT a term beats one that merely mentions it.
            " ORDER BY bm25(chunks_fts, 2.0, 3.0, 1.0), c.source_path, c.ordinal"
            " LIMIT ?",
            (match, k),
        ).fetchall()
        return [store.row_to_chunk(row) for row in rows]

    def section_map(self, budget_tokens: int) -> SectionMap:
        """The compact map the planner reads instead of the corpus.

        MEASURED 2026-09-13: **48.9 tokens per section**, 4,840 for 99 chunks
        on this repository. This said "roughly 25" from the day it was written
        and nothing ever checked it -- the figure was a guess wearing the
        grammar of a measurement, and ADR-009's argument that the catalogue is
        cheap enough to send on every planner call rests on it.

        **This used to be unbudgeted.** `catalog()` rendered every chunk in
        the corpus into the planner's system prompt on every run, with no
        truncation, no cap, no error -- exactly what §7 predicted: it grew
        with the PMA (append-only by design) until a provider rejected the
        request outright. On this repository the catalogue reached 5,191
        tokens, 54% of the planner's system prompt, before that happened on
        2026-09-16 at 9,342 total against a 3,340-token ceiling.

        `budget_tokens` is now required, with no module-level default here:
        which number is safe depends on which model answered that request and
        how the rest of the prompt is composed, neither of which this file
        can see. See `graph.budgets.SECTION_MAP_BUDGET_TOKENS` for the current
        call-site value and the trade-off it is standing in for -- more rows
        make `playbook_anchors` better informed, fewer rows make the prompt
        safer against rejection -- which is deliberately not settled here.

        Still a full scan: `SELECT *` materialises every chunk's body to build
        a line that uses four columns.
        """
        rows = self._connection.execute(
            "SELECT * FROM chunks ORDER BY source_path, ordinal"
        ).fetchall()
        chunks = [store.row_to_chunk(row) for row in rows]
        return _pack_section_map(chunks, budget_tokens)

    def drift(self) -> DriftReport:
        return store.drift(self._connection, self._root)

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])


class InMemoryPlaybookRepository:
    """The deterministic tier's implementation. Same interface, no file."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = list(chunks or [])

    def by_anchor(self, anchors: list[str]) -> list[Chunk]:
        out: list[Chunk] = []
        seen: set[str] = set()
        for anchor in anchors:
            for chunk in self._chunks:
                if anchor in chunk.anchors and chunk.id not in seen:
                    seen.add(chunk.id)
                    out.append(chunk)
        return out

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        terms = [t.lower() for t in _TERM_RE.findall(query) if len(t) > 1]
        scored = [
            (
                sum(c.body.lower().count(t) + 3 * c.heading_path.lower().count(t) for t in terms),
                i,
                c,
            )
            for i, c in enumerate(self._chunks)
        ]
        return [c for score, _, c in sorted(scored, key=lambda s: (-s[0], s[1])) if score][:k]

    def section_map(self, budget_tokens: int) -> SectionMap:
        return _pack_section_map(self._chunks, budget_tokens)

    def drift(self) -> DriftReport:
        return DriftReport()

    def count(self) -> int:
        return len(self._chunks)


def index_corpus(connection: sqlite3.Connection, root: Path) -> int:
    """Rebuild the index from every markdown file under root. Returns chunks."""
    if not root.is_dir():
        raise DynaflowsError.of(ErrorCode.CONFIG_INVALID, f"corpus root not found: {root}")
    report = store.drift(connection, root)
    store.forget_sources(connection, report.removed)
    total = 0
    for path in store.markdown_files(root):
        relative = str(path.relative_to(root))
        chunks = chunk_markdown(path.read_text(encoding="utf-8"), relative)
        store.replace_source(connection, path, relative, chunks)
        total += len(chunks)
    return total


def get_playbook_repository(
    db_path: Path | None = None, root: Path | None = None
) -> PlaybookRepository:
    """Sole construction path (playbook §3.1). Tests patch THIS, not sqlite3."""
    from dynaflows.settings import get_settings

    settings = get_settings()
    return SqlitePlaybookRepository(
        store.connect(db_path or settings.playbook_db), root or settings.playbook_root
    )
