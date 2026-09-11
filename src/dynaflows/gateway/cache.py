"""Content-addressed response cache. ADR-013.

Its justification on the record is NOT crash-safety -- a crashed 16-worker run
wastes about half a cent, and half a cent does not buy a subsystem. It is
iteration speed: tuning a synthesizer prompt without re-paying and re-waiting
for sixteen worker calls. Crash-safety is a side effect.

Separate database from the checkpointer (ADR-013.1): clearing the cache to
force fresh calls must never destroy a resumable run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from dynaflows.contracts.calls import RawResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key           TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    served_by     TEXT,
    variant       INTEGER NOT NULL DEFAULT 0,
    raw_response  TEXT NOT NULL,
    tokens_in     INTEGER NOT NULL,
    tokens_out    INTEGER NOT NULL,
    cost_usd      REAL,
    created_at    TEXT NOT NULL
);
"""


class CacheBackend(Protocol):
    """The real signature, not `**kwargs: Any`.

    A protocol that accepts anything documents nothing and lets an
    incompatible implementation typecheck -- which is the opposite of why the
    factory boundary exists (playbook §3.1).
    """

    def get(self, key: str) -> CachedCall | None: ...
    def put(
        self,
        key: str,
        *,
        model_id: str,
        variant: int,
        response: RawResponse,
        cost_usd: float | None,
    ) -> None: ...
    def clear(self) -> int: ...
    def count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CachedCall:
    response: RawResponse
    # None where the original call could not be priced. SQLite stores it as
    # NULL and hands it back as None, so the unknown survives a cache round
    # trip instead of becoming a confident zero on the next run.
    cost_usd: float | None


class ResponseCache:
    """Reads and writes `calls.db`. Never raises on a miss."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, key: str) -> CachedCall | None:
        row = self._connection.execute(
            "SELECT raw_response, tokens_in, tokens_out, cost_usd, served_by, model_id"
            " FROM calls WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return CachedCall(
            response=RawResponse(
                text=row["raw_response"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                served_by=row["served_by"],
                cost_usd=row["cost_usd"],
            ),
            cost_usd=row["cost_usd"],
        )

    def put(
        self,
        key: str,
        *,
        model_id: str,
        variant: int,
        response: RawResponse,
        cost_usd: float | None,
    ) -> None:
        """Called ONLY for a validated success (ADR-013.5).

        Caching a 429 or a schema failure would make a transient error
        permanent, which is the worst outcome available here and the easiest
        mistake to make. The caller enforces it; this docstring is the reminder
        for whoever adds the second call site.
        """
        with self._connection:
            self._connection.execute(
                "INSERT INTO calls(key, model_id, served_by, variant, raw_response,"
                " tokens_in, tokens_out, cost_usd, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET raw_response=excluded.raw_response,"
                " tokens_in=excluded.tokens_in, tokens_out=excluded.tokens_out,"
                " cost_usd=excluded.cost_usd, served_by=excluded.served_by,"
                " created_at=excluded.created_at",
                (
                    key,
                    model_id,
                    response.served_by,
                    variant,
                    response.text,
                    response.tokens_in,
                    response.tokens_out,
                    cost_usd,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def clear(self) -> int:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM calls")
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) AS n FROM calls").fetchone()["n"])


class NullCache:
    """`--no-cache`. Same interface, remembers nothing."""

    def get(self, key: str) -> CachedCall | None:
        return None

    def put(
        self,
        key: str,
        *,
        model_id: str,
        variant: int,
        response: RawResponse,
        cost_usd: float | None,
    ) -> None:
        return None

    def clear(self) -> int:
        return 0

    def count(self) -> int:
        return 0


def connect_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    # N workers write cache rows at fan-in, through SQLite's single writer --
    # the same pressure the checkpointer sees (ADR-013 consequences).
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_SCHEMA)
    _migrate_nullable_cost(connection)
    return connection


def _migrate_nullable_cost(connection: sqlite3.Connection) -> None:
    """`cost_usd` was NOT NULL until an unpriced call had to be stored.

    `CREATE TABLE IF NOT EXISTS` is silent about a table that already exists
    with the old constraint, so an existing calls.db would keep rejecting the
    write with an IntegrityError at fan-in -- inside a worker, where ADR-010
    says nothing may raise. SQLite cannot drop a NOT NULL in place, so the
    table is rebuilt. Rows are copied, not discarded: this is a cache, but
    silently emptying one is how a "why is everything suddenly slow and
    expensive" afternoon starts.
    """
    columns = connection.execute("PRAGMA table_info(calls)").fetchall()
    if not any(c["name"] == "cost_usd" and c["notnull"] for c in columns):
        return
    with connection:
        connection.executescript(
            "ALTER TABLE calls RENAME TO calls_old;\n"
            + _SCHEMA
            + "INSERT INTO calls SELECT key, model_id, served_by, variant, raw_response,"
            " tokens_in, tokens_out, cost_usd, created_at FROM calls_old;\n"
            "DROP TABLE calls_old;"
        )


def get_response_cache(db_path: Path | None = None, *, enabled: bool = True) -> CacheBackend:
    """Sole construction path (playbook §3.1). Tests patch THIS, not sqlite3."""
    if not enabled:
        return NullCache()
    from dynaflows.settings import get_settings

    return ResponseCache(connect_cache(db_path or get_settings().calls_db))
