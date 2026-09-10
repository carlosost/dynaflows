"""The checkpointer. ADR-008.

Constructed from our own aiosqlite connection rather than
`AsyncSqliteSaver.from_conn_string`, for one reason: WAL. The fan-in writes N
branch results through SQLite's single writer, and the journal mode has to be
set on the connection before the saver uses it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dynaflows.contracts import calls, errors, state, tiers

# Our own types appear in checkpointed state. LangGraph currently deserialises
# unregistered types with a warning and says it "will be blocked in a future
# version" -- AP-05's exact shape: works now, silently stops working on an
# upgrade, and here it would take resume with it. Every checkpoint ever written
# would become unreadable.
#
# Registered as CLASS REFERENCES, not module/name strings. A string list can
# name a class that no longer exists and nothing notices; an import fails on
# the spot. tests/architecture/ then asserts this list covers every state type,
# so adding a type without registering it fails the suite rather than a future
# upgrade.
ALLOWED_STATE_TYPES = (
    state.ArtifactRef,
    state.EvaluationReport,
    state.GateDecision,
    state.GateOutcome,
    state.Plan,
    state.PlanTask,
    state.WorkerResult,
    calls.CostLedger,
    errors.ErrorCode,
    errors.ErrorEnvelope,
    tiers.Tier,
)


def state_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)


@asynccontextmanager
async def open_checkpointer(db_path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Sole construction path (playbook §3.1). Tests patch THIS."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    try:
        await connection.execute("PRAGMA journal_mode=WAL")
        saver = AsyncSqliteSaver(connection, serde=state_serializer())
        await saver.setup()
        yield saver
    finally:
        await connection.close()
