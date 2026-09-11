"""The planner and gate G2. Step 1.5, ADR-005 / ADR-006 / ADR-014.

G2 is the load-bearing gate. Approving a well-worded brief tells you nothing
about the twelve workers about to run on the wrong twelve files, so this gate
shows the task list itself and stops before the fan-out spends anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from dynaflows.contracts.state import (
    MAX_FANOUT,
    GateDecision,
    Plan,
    PlanTask,
    WorkerResult,
    initial_state,
)
from dynaflows.contracts.tiers import Tier
from dynaflows.graph import build_graph, nodes, open_checkpointer
from dynaflows.graph.planner import (
    draft_to_plan,
    estimate_tokens,
    normalise_task_id,
    plan_hash,
    violation_of,
)
from dynaflows.graph.prompts import PlanDraft, PlannedTask
from dynaflows.store.catalogue import build_catalogue
from tests.conftest import FakeGateway, cfg_factory, draft_with

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cfg(workspace: Path) -> Any:
    """G2 is the subject here; G1 is auto-approved so it never appears."""
    return cfg_factory(workspace, ["prompt"])


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


async def test_the_planner_runs_at_frontier_tier(tmp_path: Path, cfg: Any) -> None:
    """ADR-006's cost asymmetry: one call that decides the cost of N."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p1", "audit"), cfg("p1", gateway))
    planner_request = next(r for r in gateway.requests if r.schema is PlanDraft)
    assert planner_request.tier is Tier.FRONTIER


async def test_the_planner_is_shown_the_playbook_catalogue(tmp_path: Path, cfg: Any) -> None:
    """ADR-009: one frontier call routes context for every worker, so it has
    to see what there is to route."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p2", "audit"), cfg("p2", gateway))
    system = next(r for r in gateway.requests if r.schema is PlanDraft).system or ""
    assert "AP-11" in system, "the anchor catalogue is missing from the planner prompt"
    assert "analyse" in system, "the capability catalogue is missing"


async def test_the_planner_is_shown_the_brief_the_human_approved(tmp_path: Path, cfg: Any) -> None:
    gateway = FakeGateway("APPROVED BRIEF")
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p3", "raw"), cfg("p3", gateway))
    assert next(r for r in gateway.requests if r.schema is PlanDraft).prompt == "APPROVED BRIEF"


async def test_an_oversized_plan_is_re_planned_not_trimmed(tmp_path: Path, cfg: Any) -> None:
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


async def test_the_re_plan_tells_the_model_what_was_wrong(tmp_path: Path, cfg: Any) -> None:
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


async def test_the_re_plan_is_not_served_from_cache(tmp_path: Path, cfg: Any) -> None:
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


async def test_two_failed_attempts_stop_the_run_and_say_why(tmp_path: Path, cfg: Any) -> None:
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


async def test_the_gate_shows_every_task_not_a_count(tmp_path: Path, cfg: Any) -> None:
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


async def test_rejecting_at_g2_stops_before_any_worker_runs(tmp_path: Path, cfg: Any) -> None:
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


async def test_approving_at_g2_dispatches_one_branch_per_task(tmp_path: Path, cfg: Any) -> None:
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(4)
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state("r", "p10", "audit"), cfg("p10", gateway))
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("p10", gateway))
    assert final["evaluation"].task_count == 4


async def test_resuming_does_not_re_run_the_planner(tmp_path: Path, cfg: Any) -> None:
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


async def test_yes_plan_skips_the_gate(tmp_path: Path, cfg: Any) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            initial_state("r", "p12", "audit"), cfg("p12", gateway, auto_approve=["plan"])
        )
    assert "__interrupt__" not in final
    assert final["plan_gate"].note == "--yes-plan"


# --- ADR-004 rule 1: no vacuous pass -------------------------------------


async def test_a_planned_task_with_no_result_at_all_does_not_pass() -> None:
    """ADR-004 rule 1, tested on the evaluator rather than through the graph.

    It used to be provoked by running a graph whose workers were stubs. Step
    1.6 gave those workers bodies, so the graph can no longer produce this
    state on demand -- and a test that can only be set up by a bug that has
    since been fixed is a test that quietly stops testing anything.

    The rule still matters and is still reachable: a `Send` branch that returns
    {} contributes no result, and a missing result is strictly worse than a
    failed one, because nothing says it happened.
    """
    plan = Plan(
        rationale="r",
        tasks=[PlanTask(task_id=f"t{i}", capability="analyse", objective="o") for i in range(5)],
    )
    state = {
        "plan": plan,
        # Two of five came back. The other three left no trace at all.
        "results": [
            WorkerResult(task_id="t0", status="ok", summary="found things"),
            WorkerResult(task_id="t1", status="ok", summary="found things"),
        ],
    }

    out = await nodes.evaluate(state)  # type: ignore[arg-type]
    report = out["evaluation"]

    assert report.task_count == 5
    assert report.passed is False
    assert "produced no result at all" in " ".join(report.reasons)
    # AP-20: a task that produced nothing is not a task that failed. Counting
    # it as failed would have made the vacuous pass merely a wrong number.
    assert report.failed_count == 0


# --- ADR-004 rule 2: a run where nothing succeeded did not succeed --------


async def test_a_run_where_every_task_degraded_does_not_pass() -> None:
    """Live run w1. Five degraded results counted as zero ok and zero failed,
    `reasons` came out empty, and the run reported passed=True -- while one of
    the five had invented four source files and a HIGH severity finding.

    Third time a fact without a counter became silence, after the zero-token
    ledger and the zero-cost ledger. The counter is the fix; this is the test
    that says the counter has to affect the verdict too.
    """
    plan = Plan(
        rationale="r",
        tasks=[PlanTask(task_id=f"t{i}", capability="analyse", objective="o") for i in range(5)],
    )
    state = {
        "plan": plan,
        "results": [
            WorkerResult(task_id=f"t{i}", status="degraded", summary="hedged") for i in range(5)
        ],
    }

    out = await nodes.evaluate(state)  # type: ignore[arg-type]
    report = out["evaluation"]

    assert report.passed is False
    assert report.degraded_count == 5
    assert "no task succeeded cleanly" in " ".join(report.reasons)
    # The counters must account for every task, or a status vanishes again.
    assert report.unaccounted == 0
    assert "5 degraded" in report.render()


async def test_a_partly_degraded_run_passes_but_never_silently() -> None:
    """Not a failure -- some work succeeded. But the degraded ones are exactly
    the results a reader must not trust equally, so they are never unsaid."""
    plan = Plan(
        rationale="r",
        tasks=[PlanTask(task_id=f"t{i}", capability="analyse", objective="o") for i in range(3)],
    )
    state = {
        "plan": plan,
        "results": [
            WorkerResult(task_id="t0", status="ok", summary="grounded"),
            WorkerResult(task_id="t1", status="ok", summary="grounded"),
            WorkerResult(task_id="t2", status="degraded", summary="hedged"),
        ],
    }

    out = await nodes.evaluate(state)  # type: ignore[arg-type]
    report = out["evaluation"]

    assert report.passed is False
    assert report.ok_count == 2
    assert "1 of 3 task(s) degraded" in " ".join(report.reasons)


async def test_every_status_has_a_counter() -> None:
    """The structural guard. A new WorkerResult status with no counter here
    would repeat the bug exactly, so the sum is asserted rather than the
    individual numbers."""
    plan = Plan(
        rationale="r",
        tasks=[PlanTask(task_id=f"t{i}", capability="analyse", objective="o") for i in range(4)],
    )
    state = {
        "plan": plan,
        "results": [
            WorkerResult(task_id="t0", status="ok", summary="s"),
            WorkerResult(task_id="t1", status="degraded", summary="s"),
            WorkerResult(task_id="t2", status="failed", summary=""),
            WorkerResult(task_id="t3", status="degraded", summary=""),
        ],
    }

    out = await nodes.evaluate(state)  # type: ignore[arg-type]
    report = out["evaluation"]

    assert report.unaccounted == 0, report.render()


# --- ADR-018: a plan that names what does not exist ----------------------


def _catalogue_of(tmp_path: Path) -> Any:
    from tests.conftest import make_workspace

    return build_catalogue(make_workspace(tmp_path))


def test_a_task_naming_no_inputs_is_a_violation(tmp_path: Path) -> None:
    """Run `w1`: five tasks, no inputs, and a worker that invented an audit to
    fill the vacuum. `analyse` means "read the named inputs"; with none named
    there is nothing to read, so the plan is sent back rather than run."""
    draft = PlanDraft(rationale="r", tasks=[a_task("t1")])

    reason = violation_of(draft, _catalogue_of(tmp_path))

    assert reason is not None
    assert "named no inputs" in reason
    # Phrased for the model to act on: naming the fix, not the error class.
    assert "source catalogue" in reason


def test_a_task_naming_a_path_that_does_not_exist_is_a_violation(tmp_path: Path) -> None:
    draft = PlanDraft(rationale="r", tasks=[a_task("t1", inputs=["src/logger.py"])])

    reason = violation_of(draft, _catalogue_of(tmp_path))

    assert reason is not None
    assert "src/logger.py" in reason
    assert "not in the source catalogue" in reason


def test_a_task_naming_a_real_path_passes(tmp_path: Path) -> None:
    draft = PlanDraft(rationale="r", tasks=[a_task("t1", inputs=["src/auth.py"])])

    assert violation_of(draft, _catalogue_of(tmp_path)) is None


def test_a_directory_input_passes(tmp_path: Path) -> None:
    draft = PlanDraft(rationale="r", tasks=[a_task("t1", inputs=["src"])])

    assert violation_of(draft, _catalogue_of(tmp_path)) is None


def test_without_a_catalogue_the_path_rules_are_off_rather_than_vacuously_green() -> None:
    """A check that silently passes when its input is missing is worse than one
    that is not there: it reads as coverage. The other rules still apply."""
    empty_inputs = PlanDraft(rationale="r", tasks=[a_task("t1")])

    assert violation_of(empty_inputs, None) is None
    assert violation_of(PlanDraft(rationale="r", tasks=[]), None) is not None


async def test_the_planner_is_shown_the_source_catalogue(tmp_path: Path, cfg: Any) -> None:
    """The prompt is the deliverable here. Without it the planner guesses."""
    gateway = FakeGateway()

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(
            initial_state("r", "cat1", "audit"), cfg("cat1", gateway, auto_approve=["plan"])
        )

    system = next(r.system for r in gateway.requests if r.schema is PlanDraft)
    assert "Source catalogue" in system
    assert "src/auth.py" in system
    assert "Login and session handling." in system


async def test_an_invented_path_costs_exactly_one_re_plan_then_halts(
    tmp_path: Path, cfg: Any
) -> None:
    """ADR-014's existing path, reached by ADR-018's rule. Never a silent trim
    and never an unbounded retry loop."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: PlanDraft(
        rationale="r", tasks=[a_task("t1", inputs=["src/invented.py"])]
    )

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            initial_state("r", "cat2", "audit"), cfg("cat2", gateway, auto_approve=["plan"])
        )

    assert gateway.planner_calls == 2
    assert final["plan"] is None
    assert "src/invented.py" in final["plan_rejected_reason"]


async def test_the_correction_reaches_the_second_attempt(tmp_path: Path, cfg: Any) -> None:
    """A re-plan that does not tell the model what was wrong is a re-roll."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: PlanDraft(
        rationale="r", tasks=[a_task("t1", inputs=["src/invented.py"])]
    )

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        await graph.ainvoke(
            initial_state("r", "cat3", "audit"), cfg("cat3", gateway, auto_approve=["plan"])
        )

    second = [r for r in gateway.requests if r.schema is PlanDraft][1]
    assert "PREVIOUS ATTEMPT FAILED" in second.prompt
    assert "src/invented.py" in second.prompt


# --- ADR-021: the gate must authorise against a number that means something --


async def test_the_gate_shows_measured_context_not_an_estimate(tmp_path: Path, cfg: Any) -> None:
    """Run `s1` displayed "~10,500 tokens estimated" for three tasks whose
    inputs came to 40,000 -- the task count times a constant I invented. The
    gate that authorises the entire fan-out spend was showing a number with no
    relationship to what the workers would receive."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: PlanDraft(
        rationale="r", tasks=[a_task("t1", inputs=["src/auth.py"])]
    )

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        out = await graph.ainvoke(initial_state("r", "cb1", "audit"), cfg("cb1", gateway))

    payload = out["__interrupt__"][0].value
    task = payload["tasks"][0]
    assert task["context_tokens"] > 0
    assert payload["context_budget"] > 0
    # The measurement is of the real file, through the real resolver.
    assert task["context_tokens"] < payload["context_budget"]
    assert payload["over_budget"] == []


async def test_a_task_that_will_not_fit_is_named_at_the_gate(
    tmp_path: Path, cfg: Any, workspace: Path
) -> None:
    """Not an error -- the worker is shown what fits and told what it did not
    get. It IS something the person approving the spend has to see, because
    the remedy is theirs."""
    # Large, but under MAX_FILE_BYTES -- a file over that cap never enters the
    # catalogue at all, so the plan naming it is rejected before the gate. That
    # is correct and it is a different test.
    big = workspace / "src" / "huge.py"
    big.write_text('"""Big."""\n' + "x = 1\n" * 25_000, encoding="utf-8")

    gateway = FakeGateway()
    gateway.plan_draft = lambda: PlanDraft(
        rationale="r", tasks=[a_task("t1", inputs=["src/huge.py"])]
    )

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        out = await graph.ainvoke(initial_state("r", "cb2", "audit"), cfg("cb2", gateway))

    payload = out["__interrupt__"][0].value
    assert payload["over_budget"] == ["t1"]
    assert payload["tasks"][0]["context_tokens"] > payload["context_budget"]
