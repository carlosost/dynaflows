"""Step 1.6: the worker node and the fan-out it runs in.

The rule that shapes this file: a worker must never raise. An exception inside
a Send branch aborts the whole superstep, so "one of twelve failed" becomes
"the run is gone" -- the single worst outcome available in this design, and the
easiest one to cause by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.state import PlanTask, initial_state
from dynaflows.graph import build_graph, nodes, open_checkpointer
from dynaflows.graph.prompts import WorkerReport
from tests.conftest import FakeGateway, cfg_factory, draft_with, graph_config

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cfg(workspace: Path) -> Any:
    """Both gates skipped: the fan-out is the subject here."""
    return cfg_factory(workspace, ["prompt", "plan"])


def a_task(**kwargs: Any) -> PlanTask:
    return PlanTask(
        task_id=kwargs.pop("task_id", "t1"),
        capability="analyse",
        objective=kwargs.pop("objective", "review the login path"),
        **kwargs,
    )


async def run_worker(task: PlanTask, gateway: FakeGateway, root: Path, **extra: Any) -> dict:
    config = graph_config("w", gateway, root, **extra)
    state = {**initial_state("r1", "w", "audit"), "task": task}
    return await nodes.worker(state, config)  # type: ignore[arg-type]


# --- the context a worker is given (ADR-017) -----------------------------


async def test_the_named_input_reaches_the_prompt(workspace: Path) -> None:
    gateway = FakeGateway()

    await run_worker(a_task(inputs=["src/auth.py"]), gateway, workspace)

    prompt = gateway.requests[-1].prompt
    assert "def login" in prompt
    assert "src/auth.py" in prompt


async def test_playbook_anchors_are_packed_before_source(workspace: Path) -> None:
    """A tight budget must drop the code before it drops the rules the code is
    being judged against. Order in the pack is what decides that."""
    gateway = FakeGateway()

    await run_worker(a_task(inputs=["src/auth.py"], playbook_anchors=["AP-11"]), gateway, workspace)

    prompt = gateway.requests[-1].prompt
    assert prompt.index("AP-11") < prompt.index("def login")


async def test_a_refused_input_is_named_in_the_prompt_not_just_the_trace(
    workspace: Path,
) -> None:
    """A model told 'this was withheld' reports a gap. A model handed a
    silently shorter context reports confidently on a subset."""
    gateway = FakeGateway()

    await run_worker(a_task(inputs=["src/nope.py"]), gateway, workspace)

    prompt = gateway.requests[-1].prompt
    assert "NOT AVAILABLE" in prompt
    assert "src/nope.py" in prompt


async def test_the_worker_never_receives_a_file_tool(workspace: Path) -> None:
    """ADR-016's grant, checked from the call site: a worker gets text."""
    gateway = FakeGateway()

    await run_worker(a_task(inputs=["src/auth.py"]), gateway, workspace)

    request = gateway.requests[-1]
    assert getattr(request, "tools", None) in (None, (), [])
    assert request.schema is WorkerReport


# --- what it returns ------------------------------------------------------


async def test_a_clean_run_writes_an_artifact_and_keeps_state_small(workspace: Path) -> None:
    """ADR-008: the payload goes to disk, the reference goes into state."""
    gateway = FakeGateway()

    out = await run_worker(
        a_task(inputs=["src/auth.py"], playbook_anchors=["AP-11"]), gateway, workspace
    )

    result = out["results"][0]
    assert result.status == "ok"
    assert result.artifact is not None
    assert Path(result.artifact.path).read_text(encoding="utf-8").startswith("# t1")
    # The full report is NOT in state; only a preview of it.
    assert len(result.summary) < 200


async def test_a_model_that_says_it_lacked_context_is_degraded_not_ok(workspace: Path) -> None:
    """Hiding that as `ok` is how a run passes while telling the user nothing."""
    gateway = FakeGateway()
    gateway.worker_report = lambda request: WorkerReport(
        summary="partial",
        findings="# partial",
        context_was_sufficient=False,
        missing=["the session store"],
    )

    out = await run_worker(a_task(), gateway, workspace)

    assert out["results"][0].status == "degraded"


async def test_a_refused_input_degrades_the_result_even_if_the_model_is_happy(
    workspace: Path,
) -> None:
    """The model cannot know what it was not shown. The node can."""
    gateway = FakeGateway()

    out = await run_worker(a_task(inputs=["src/nope.py"]), gateway, workspace)

    assert out["results"][0].status == "degraded"


async def test_the_worker_records_its_own_cost(workspace: Path) -> None:
    gateway = FakeGateway()

    out = await run_worker(a_task(), gateway, workspace)

    assert out["cost"].calls_made == 1
    assert out["cost"].usd_spent == pytest.approx(0.002)


# --- a worker never raises ------------------------------------------------


async def test_a_gateway_failure_becomes_a_failed_result(workspace: Path) -> None:
    gateway = FakeGateway()

    async def boom(request: Any, **_: object) -> Any:
        raise DynaflowsError.of(ErrorCode.RATE_LIMIT, "429 everywhere")

    gateway.call = boom  # type: ignore[method-assign]

    out = await run_worker(a_task(), gateway, workspace)

    result = out["results"][0]
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code is ErrorCode.RATE_LIMIT


async def test_an_unexpected_exception_becomes_a_failed_result_too(workspace: Path) -> None:
    """The 'NoneType is not iterable' class of failure. An untyped exception
    from below must not take eleven siblings with it."""
    gateway = FakeGateway()

    async def boom(request: Any, **_: object) -> Any:
        raise TypeError("'NoneType' object is not iterable")

    gateway.call = boom  # type: ignore[method-assign]

    out = await run_worker(a_task(), gateway, workspace)

    assert out["results"][0].status == "failed"


async def test_a_missing_dependency_fails_the_task_not_the_run(workspace: Path) -> None:
    gateway = FakeGateway()
    state = {**initial_state("r1", "w", "audit"), "task": a_task()}

    out = await nodes.worker(state, {"configurable": {"gateway": gateway}})  # type: ignore[arg-type]

    assert out["results"][0].status == "failed"
    assert out["results"][0].error.code is ErrorCode.CONFIG_INVALID


async def test_an_unwritable_store_degrades_rather_than_discarding_the_work(
    workspace: Path,
) -> None:
    """The analysis succeeded and the disk did not. Calling that `failed`
    throws away work already paid for."""
    gateway = FakeGateway()

    class BrokenStore:
        def write(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError("read-only file system")

    out = await run_worker(a_task(), gateway, workspace, store=BrokenStore())

    result = out["results"][0]
    assert result.status == "degraded"
    assert "not saved" in result.summary
    assert result.summary.startswith("findings for")


# --- the fan-out ----------------------------------------------------------


async def test_n_tasks_produce_n_distinct_results_and_n_distinct_artifacts(
    tmp_path: Path, cfg: Any
) -> None:
    """The reducer merges concurrent branches; this proves they were different
    branches and not one answer written five times."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(5)

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(initial_state("r2", "fan", "audit"), cfg("fan", gateway))

    results = final["results"]
    assert len(results) == 5
    assert gateway.worker_calls == 5
    assert len({r.task_id for r in results}) == 5
    assert len({r.artifact.sha for r in results}) == 5
    assert final["evaluation"].task_count == 5
    assert final["evaluation"].ok_count == 5


async def test_one_failing_worker_does_not_take_the_others_with_it(
    tmp_path: Path, cfg: Any
) -> None:
    """The whole reason a worker never raises."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(4)
    original = gateway.call

    async def sometimes(request: Any, **kw: object) -> Any:
        if dict(request.metadata).get("task_id") == "task-2":
            raise DynaflowsError.of(ErrorCode.MODEL_UNAVAILABLE, "down")
        return await original(request, **kw)

    gateway.call = sometimes  # type: ignore[method-assign]

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(initial_state("r3", "part", "audit"), cfg("part", gateway))

    statuses = {r.task_id: r.status for r in final["results"]}
    assert statuses["task-2"] == "failed"
    assert sum(1 for s in statuses.values() if s == "ok") == 3
    assert final["evaluation"].failed_count == 1


# --- the run that passed while a worker invented an audit -----------------


async def test_a_worker_with_no_source_is_told_not_to_invent_one(workspace: Path) -> None:
    """Live run w1: five workers got no source, four said so, and the fifth
    reported six findings against four files that do not exist -- with line
    numbers, one of them HIGH severity. "Audit the gateway package" reads like
    permission to describe what such a package usually contains, so the
    instruction not to has to be explicit."""
    gateway = FakeGateway()

    await run_worker(a_task(inputs=[]), gateway, workspace)

    prompt = gateway.requests[-1].prompt
    assert "NO SOURCE FILES WERE PROVIDED" in prompt
    assert "line numbers you were not shown" in prompt


async def test_a_worker_that_saw_source_is_not_told_that(workspace: Path) -> None:
    """The warning must not fire on the ordinary path, or it becomes noise the
    model learns to skip (playbook 5.2, Pattern 5, applied to a prompt)."""
    gateway = FakeGateway()

    await run_worker(a_task(inputs=["src/auth.py"]), gateway, workspace)

    assert "NO SOURCE FILES WERE PROVIDED" not in gateway.requests[-1].prompt
