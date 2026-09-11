"""The planner and gate G2. Step 1.5, ADR-005 / ADR-006 / ADR-014.

G2 is the load-bearing gate. Approving a well-worded brief tells you nothing
about the twelve workers about to run on the wrong twelve files, so this gate
shows the task list itself and stops before the fan-out spends anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from dynaflows.contracts.state import MAX_FANOUT, GateDecision, Plan, PlanTask, initial_state
from dynaflows.contracts.tiers import Tier
from dynaflows.graph import build_graph, open_checkpointer
from dynaflows.graph.planner import (
    draft_to_plan,
    estimate_tokens,
    normalise_task_id,
    plan_hash,
    violation_of,
)
from dynaflows.graph.prompts import PlanDraft, PlannedTask
from tests.conftest import FakeGateway, draft_with, make_playbook

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def cfg(thread: str, gateway: FakeGateway, **extra: object) -> dict:
    return {
        "configurable": {
            "thread_id": thread,
            "gateway": gateway,
            "playbook": make_playbook(),
            "auto_approve": ["prompt", *(extra.pop("auto_approve", []) or [])],  # type: ignore[misc]
            **extra,
        }
    }


def a_task(task_id: str = "t", **kwargs: object) -> PlannedTask:
    return PlannedTask(
        task_id=task_id,
        capability=kwargs.pop("capability", "analyse"),  # type: ignore[arg-type]
        objective="do the thing",
        **kwargs,  # type: ignore[arg-type]
    )


# --- pure planning rules -------------------------------------------------


def test_a_messy_task_id_is_normalised_rather_than_rejected() -> None:
    """A model returning "Task 1: Review auth" made a formatting error, not a
    planning one. Re-planning for that spends a frontier call on a hyphen."""
    assert normalise_task_id("Task 1: Review auth", 0) == "task-1-review-auth"
    assert normalise_task_id("   ", 2) == "task-3"


def test_an_oversized_plan_names_the_bound_and_what_to_do() -> None:
    """The message goes back to the model verbatim. "ValidationError" teaches
    it nothing; naming the limit gives it something to change."""
    draft = PlanDraft(rationale="r", tasks=[a_task(f"t{i}") for i in range(MAX_FANOUT + 3)])
    message = violation_of(draft)
    assert message is not None
    assert str(MAX_FANOUT) in message and "Merge" in message


def test_an_empty_plan_is_a_violation() -> None:
    assert violation_of(PlanDraft(rationale="r", tasks=[])) is not None


def test_an_unregistered_capability_is_a_violation_not_a_crash() -> None:
    draft = PlanDraft(rationale="r", tasks=[a_task("t", capability="invent")])
    assert "unknown capability" in (violation_of(draft) or "")


def test_a_valid_draft_has_no_violation() -> None:
    assert violation_of(draft_with(3)) is None


def test_the_plan_hash_covers_what_the_plan_was_planned_against() -> None:
    """A plan means something different under a different catalogue or model
    set; comparing across either would be comparing nothing (ADR-001/006)."""
    plan = draft_to_plan(draft_with(2))
    assert plan_hash(plan, "v1") == plan_hash(plan, "v1")
    assert plan_hash(plan, "v1") != plan_hash(plan, "v2")


def test_the_plan_hash_changes_when_a_task_changes() -> None:
    base = draft_to_plan(draft_with(2))
    other = Plan(
        rationale=base.rationale,
        tasks=[
            base.tasks[0],
            PlanTask(task_id="different", capability="analyse", objective="other"),
        ],
    )
    assert plan_hash(base, "v1") != plan_hash(other, "v1")


def test_the_token_estimate_scales_with_the_fan_out() -> None:
    assert estimate_tokens(draft_to_plan(draft_with(4))) == 4 * estimate_tokens(
        draft_to_plan(draft_with(1))
    )


# --- the planner node ----------------------------------------------------


async def test_the_planner_runs_at_frontier_tier(tmp_path: Path) -> None:
    """ADR-006's cost asymmetry: one call that decides the cost of N."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p1", "audit"), cfg("p1", gateway))
    planner_request = next(r for r in gateway.requests if r.schema is PlanDraft)
    assert planner_request.tier is Tier.FRONTIER


async def test_the_planner_is_shown_the_playbook_catalogue(tmp_path: Path) -> None:
    """ADR-009: one frontier call routes context for every worker, so it has
    to see what there is to route."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p2", "audit"), cfg("p2", gateway))
    system = next(r for r in gateway.requests if r.schema is PlanDraft).system or ""
    assert "AP-11" in system, "the anchor catalogue is missing from the planner prompt"
    assert "analyse" in system, "the capability catalogue is missing"


async def test_the_planner_is_shown_the_brief_the_human_approved(tmp_path: Path) -> None:
    gateway = FakeGateway("APPROVED BRIEF")
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p3", "raw"), cfg("p3", gateway))
    assert next(r for r in gateway.requests if r.schema is PlanDraft).prompt == "APPROVED BRIEF"


async def test_an_oversized_plan_is_re_planned_not_trimmed(tmp_path: Path) -> None:
    """ADR-014. Silently dropping tasks produces a synthesis that is
    incomplete without saying so."""
    gateway = FakeGateway()
    attempts = iter([draft_with(MAX_FANOUT + 5), draft_with(4)])
    gateway.plan_draft = lambda: next(attempts)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            initial_state("r", "p4", "audit"), cfg("p4", gateway, auto_approve=["plan"])
        )
    assert gateway.planner_calls == 2
    assert len(final["plan"].tasks) == 4


async def test_the_re_plan_tells_the_model_what_was_wrong(tmp_path: Path) -> None:
    gateway = FakeGateway()
    attempts = iter([draft_with(MAX_FANOUT + 1), draft_with(2)])
    gateway.plan_draft = lambda: next(attempts)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(
            initial_state("r", "p5", "audit"), cfg("p5", gateway, auto_approve=["plan"])
        )
    second = [r for r in gateway.requests if r.schema is PlanDraft][1]
    assert "PREVIOUS ATTEMPT FAILED" in second.prompt
    assert str(MAX_FANOUT) in second.prompt


async def test_the_re_plan_is_not_served_from_cache(tmp_path: Path) -> None:
    """A re-plan must be a fresh call, not a cache hit on the answer that was
    just rejected -- hence a different variant (ADR-013)."""
    gateway = FakeGateway()
    attempts = iter([draft_with(MAX_FANOUT + 1), draft_with(2)])
    gateway.plan_draft = lambda: next(attempts)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(
            initial_state("r", "p6", "audit"), cfg("p6", gateway, auto_approve=["plan"])
        )
    planner_requests = [r for r in gateway.requests if r.schema is PlanDraft]
    assert planner_requests[0].variant != planner_requests[1].variant


async def test_two_failed_attempts_stop_the_run_and_say_why(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(MAX_FANOUT + 2)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(initial_state("r", "p7", "audit"), cfg("p7", gateway))
    assert gateway.planner_calls == 2
    assert final["plan"] is None
    assert final["halted"] == "no usable plan after two attempts"
    assert str(MAX_FANOUT) in (final["plan_gate"].note or "")
    assert final.get("results") == [], "the fan-out ran without a plan"


# --- the gate ------------------------------------------------------------


async def test_the_gate_shows_every_task_not_a_count(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(5)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        out = await graph.ainvoke(initial_state("r", "p8", "audit"), cfg("p8", gateway))
    payload = out["__interrupt__"][0].value
    assert payload["gate"] == "plan"
    assert len(payload["tasks"]) == 5
    assert payload["tasks"][0]["anchors"] == ["AP-11"]
    assert payload["plan_hash"]
    assert payload["estimated_tokens"] > 0


async def test_rejecting_at_g2_stops_before_any_worker_runs(tmp_path: Path) -> None:
    """The whole point of the gate: the fan-out is the irreversible spend."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p9", "audit"), cfg("p9", gateway))
        final = await graph.ainvoke(Command(resume={"decision": "reject"}), cfg("p9", gateway))
    assert final["plan_gate"].decision is GateDecision.REJECT
    assert final["halted"] == "rejected by human at gate G2"
    assert final.get("evaluation") is None
    assert final.get("results") == []


async def test_approving_at_g2_dispatches_one_branch_per_task(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(4)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p10", "audit"), cfg("p10", gateway))
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("p10", gateway))
    assert final["evaluation"].task_count == 4


async def test_resuming_does_not_re_run_the_planner(tmp_path: Path) -> None:
    """ADR-007 again, at the expensive gate. A re-planned brief on resume
    would cost a frontier call AND could differ from what was approved."""
    gateway = FakeGateway()
    db = tmp_path / "s.db"
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        out = await graph.ainvoke(initial_state("r", "p11", "audit"), cfg("p11", gateway))
        shown = out["__interrupt__"][0].value["plan_hash"]
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("p11", gateway))
    assert gateway.planner_calls == 1, "the planner was paid for twice"
    assert final["plan_hash"] == shown, "the approved plan is not the plan that ran"


async def test_yes_plan_skips_the_gate(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            initial_state("r", "p12", "audit"), cfg("p12", gateway, auto_approve=["plan"])
        )
    assert "__interrupt__" not in final
    assert final["plan_gate"].note == "--yes-plan"
