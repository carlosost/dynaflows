"""The graph topology, its reducers, and resume. Step 1.3.

No LLM calls anywhere: every node is a pass-through. That is the point --
proving a run survives being killed is far easier while nothing in it is
expensive or non-deterministic, and the proof stays valid once they are.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.contracts.calls import CostLedger
from dynaflows.contracts.state import (
    MAX_FANOUT,
    Plan,
    PlanTask,
    WorkerResult,
    cost_delta,
    initial_state,
    merge_cost,
)
from dynaflows.graph import build_graph, open_checkpointer
from dynaflows.graph.builder import dispatch_workers

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def a_plan(n: int = 3) -> Plan:
    return Plan(
        tasks=[PlanTask(task_id=f"t{i}", capability="analyse", objective="x") for i in range(n)]
    )


def config(thread: str = "th-1") -> dict:
    return {"configurable": {"thread_id": thread}}


# --- topology ------------------------------------------------------------


async def test_the_skeleton_runs_end_to_end(tmp_path: Path) -> None:
    async with open_checkpointer(tmp_path / "state.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(initial_state("r1", "th-1", "audit auth"), config())
    assert final["enhanced_prompt"] == "audit auth"
    assert final["evaluation"] is not None
    assert final["prompt_gate"].proceeds


async def test_fan_out_width_comes_from_the_plan(tmp_path: Path) -> None:
    """ADR-001's one genuinely runtime-variable dimension."""
    state = initial_state("r1", "th-1", "x")
    state["plan"] = a_plan(5)
    sends = dispatch_workers(state)
    assert isinstance(sends, list)
    assert len(sends) == 5
    assert {s.node for s in sends} == {"worker"}


async def test_an_empty_plan_routes_to_evaluate_instead_of_dispatching_nothing() -> None:
    """A fan-out of zero leaves the barrier waiting on a superstep that never
    runs."""
    state = initial_state("r1", "th-1", "x")
    assert dispatch_workers(state) == "evaluate"


# --- reducers ------------------------------------------------------------


async def test_concurrent_branches_accumulate_instead_of_overwriting(tmp_path: Path) -> None:
    """The real proof that FAN_OUT_KEYS is right: run an actual fan-out. If a
    reducer were missing, LangGraph raises InvalidUpdateError here."""
    from dynaflows.graph import nodes

    async def counting_worker(state, config=None):  # noqa: ANN001, ANN202
        task = state["task"]
        return {
            "results": [WorkerResult(task_id=task.task_id, status="ok", summary="done")],
            "cost": cost_delta(tokens_in=10, tokens_out=5, calls_made=1, usd_spent=0.5),
        }

    original = nodes.worker
    nodes.worker = counting_worker  # type: ignore[assignment]
    try:
        async with open_checkpointer(tmp_path / "state.db") as saver:
            graph = build_graph(saver)
            state = initial_state("r1", "th-fan", "x")
            state["plan"] = a_plan(6)
            final = await graph.ainvoke(state, config("th-fan"))
    finally:
        nodes.worker = original  # type: ignore[assignment]

    assert len(final["results"]) == 6
    assert {r.task_id for r in final["results"]} == {f"t{i}" for i in range(6)}
    assert final["cost"].calls_made == 6
    assert final["cost"].tokens_in == 60
    assert final["cost"].usd_spent == pytest.approx(3.0)


def test_merge_cost_sums_every_field() -> None:
    a = CostLedger(usd_spent=1.0, tokens_in=5, calls_made=1, schema_failures=1)
    b = CostLedger(usd_avoided=2.0, tokens_out=7, calls_cached=1, fallbacks=2)
    merged = merge_cost(a, b)
    assert (merged.usd_spent, merged.usd_avoided) == (1.0, 2.0)
    assert (merged.tokens_in, merged.tokens_out) == (5, 7)
    assert (merged.calls_made, merged.calls_cached) == (1, 1)
    assert (merged.schema_failures, merged.fallbacks) == (1, 2)


def test_a_cost_delta_is_a_single_node_contribution_not_a_total() -> None:
    """Every branch starts from the same input state, so a branch returning a
    running total would overwrite its siblings rather than combine."""
    assert cost_delta(usd_spent=0.25).calls_made == 0


# --- checkpointing and resume -------------------------------------------


async def test_a_run_stops_at_a_breakpoint_and_keeps_its_state(tmp_path: Path) -> None:
    async with open_checkpointer(tmp_path / "state.db") as saver:
        graph = build_graph(saver, interrupt_before=("plan",))
        await graph.ainvoke(initial_state("r1", "th-2", "audit auth"), config("th-2"))
        snapshot = await graph.aget_state(config("th-2"))
    assert snapshot.next == ("plan",)
    assert snapshot.values["enhanced_prompt"] == "audit auth"


async def test_resuming_by_thread_id_continues_rather_than_restarting(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver, interrupt_before=("plan",))
        await graph.ainvoke(initial_state("r1", "th-3", "audit auth"), config("th-3"))

    # A fresh process: new connection, new saver, new compiled graph.
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(None, config("th-3"))
    assert final["enhanced_prompt"] == "audit auth"
    assert final["evaluation"] is not None


async def test_state_survives_a_crash_mid_graph(tmp_path: Path) -> None:
    """Not a breakpoint -- an exception, which is what a real kill looks like.
    Work completed before the failing superstep must still be there."""
    from dynaflows.graph import nodes

    async def exploding_plan(state, config=None):  # noqa: ANN001, ANN202
        raise RuntimeError("process died")

    db = tmp_path / "state.db"
    original = nodes.plan
    nodes.plan = exploding_plan  # type: ignore[assignment]
    try:
        async with open_checkpointer(db) as saver:
            graph = build_graph(saver)
            with pytest.raises(RuntimeError):
                await graph.ainvoke(initial_state("r1", "th-4", "audit auth"), config("th-4"))
    finally:
        nodes.plan = original  # type: ignore[assignment]

    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        snapshot = await graph.aget_state(config("th-4"))
        assert snapshot.values["enhanced_prompt"] == "audit auth"
        final = await graph.ainvoke(None, config("th-4"))
    assert final["evaluation"] is not None


async def test_two_threads_do_not_share_state(tmp_path: Path) -> None:
    async with open_checkpointer(tmp_path / "state.db") as saver:
        graph = build_graph(saver)
        a = await graph.ainvoke(initial_state("r1", "A", "first"), config("A"))
        b = await graph.ainvoke(initial_state("r2", "B", "second"), config("B"))
    assert a["enhanced_prompt"] == "first"
    assert b["enhanced_prompt"] == "second"


async def test_the_checkpoint_database_uses_wal(tmp_path: Path) -> None:
    """N branches write through SQLite's single writer at fan-in (ADR-008)."""
    import sqlite3

    db = tmp_path / "state.db"
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r1", "th-5", "x"), config("th-5"))
    with sqlite3.connect(db) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# --- plan contract -------------------------------------------------------


def test_a_plan_may_not_exceed_max_fanout() -> None:
    """ADR-014. The bound is enforced by the schema, not by a comment."""
    with pytest.raises(ValueError, match="at most"):
        Plan(
            tasks=[
                PlanTask(task_id=f"t{i}", capability="c", objective="o")
                for i in range(MAX_FANOUT + 1)
            ]
        )


def test_a_plan_needs_at_least_one_task() -> None:
    with pytest.raises(ValueError):
        Plan(tasks=[])


def test_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Plan(tasks=[PlanTask(task_id="t", capability="c", objective="o") for _ in range(2)])


def test_intra_plan_dependencies_are_rejected_while_oq_02_is_open() -> None:
    """A dependency that is silently ignored produces a plan that runs in the
    wrong order with no error anywhere."""
    with pytest.raises(ValueError, match="OQ-02"):
        PlanTask(task_id="t", capability="c", objective="o", depends_on=["other"])


async def test_a_stub_node_does_not_erase_state_an_earlier_step_set(tmp_path: Path) -> None:
    """The regression that motivated the rule in nodes.py's docstring.

    `plan` used to return {"plan": None}, which is a destructive write dressed
    as a pass-through: it erased an injected plan, the conditional edge saw no
    tasks, and the fan-out silently never dispatched. "Nothing to contribute"
    and "the value is None" are different statements.
    """
    async with open_checkpointer(tmp_path / "state.db") as saver:
        graph = build_graph(saver)
        state = initial_state("r1", "th-stub", "x")
        state["plan"] = a_plan(2)
        final = await graph.ainvoke(state, config("th-stub"))
    assert final["plan"] is not None
    assert len(final["plan"].tasks) == 2
    assert final["evaluation"].task_count == 2


async def test_checkpointed_state_round_trips_under_strict_msgpack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward compatibility for resume, proven rather than assumed.

    LangGraph deserialises unregistered types with a warning today and says it
    will BLOCK them in a future version -- AP-05's shape, and here it would
    take resume with it: every checkpoint ever written becomes unreadable.
    Strict mode makes that future the present, so this test fails now if a new
    state type is added without registering its module.
    """
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    db = tmp_path / "state.db"
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver, interrupt_before=("plan",))
        state = initial_state("r1", "th-strict", "audit auth")
        state["plan"] = a_plan(2)
        await graph.ainvoke(state, config("th-strict"))

    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        snapshot = await graph.aget_state(config("th-strict"))
        # Every one of our types has to survive the round trip, not just str.
        assert snapshot.values["prompt_gate"].proceeds is True
        assert snapshot.values["plan"].tasks[0].task_id == "t0"
        assert snapshot.values["cost"].calls_made == 0
        final = await graph.ainvoke(None, config("th-strict"))
    assert final["evaluation"].task_count == 2
