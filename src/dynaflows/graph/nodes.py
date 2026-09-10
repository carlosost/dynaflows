"""Graph nodes. Step 1.3: every one is a typed pass-through.

Deliberately empty. Building the topology, the reducers and resume BEFORE the
nodes do anything is what makes the hard part testable: proving a run survives
being killed is far easier when nothing in it is expensive or
non-deterministic. Steps 1.4-1.7 fill these in one at a time.

Two rules, and the first cost a real bug in this very file:

  * A stub returns `{}`. A stub that returns `{"key": None}` is NOT a
    pass-through -- it is a destructive write, and it silently erased a plan
    that an earlier step had set. "Nothing to contribute" and "the value is
    None" are different statements and LangGraph cannot tell them apart.

  * A node returns ONLY the keys it owns. Returning whole state from a
    concurrent branch overwrites its siblings even where a reducer exists.
"""

# NOT `from __future__ import annotations`. LangGraph inspects node signatures
# at add_node() time to decide how to call them, and postponed annotations
# leave it comparing the STRING "RunnableConfig | None" against a type -- which
# it cannot resolve, so it warns on every node. A warning that fires on
# ordinary work is a warning people learn to ignore (playbook §5.1), so the
# annotations here stay real objects.
from typing import Any

from langchain_core.runnables import RunnableConfig

from dynaflows.contracts.state import (
    EvaluationReport,
    GateDecision,
    GateOutcome,
    WorkflowState,
)


async def enhance_prompt(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Step 1.4 replaces the body with a real call.

    The LLM call belongs HERE, never in the gate that follows: ADR-007, a
    resumed graph re-runs the interrupted node from its first line.
    """
    return {"enhanced_prompt": state.get("raw_prompt", "")}


async def approve_prompt(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Gate G1. Step 1.4 adds `interrupt()` and nothing else ever may -- this
    node stays pure so re-running it on resume costs nothing."""
    return {"prompt_gate": GateOutcome(decision=GateDecision.APPROVE)}


async def plan(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Step 1.5. Returns {} -- it has nothing to say yet, which is not the
    same as saying None."""
    return {}


async def approve_plan(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Gate G2, the load-bearing one (ADR-005). Step 1.5 adds `interrupt()`."""
    return {"plan_gate": GateOutcome(decision=GateDecision.APPROVE)}


async def worker(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Step 1.6, dispatched by Send. Returns a DELTA on every reduced key.

    A worker must never raise: an exception inside a Send branch aborts the
    whole superstep, turning 'one of twelve failed' into 'the run is gone'
    (ADR-010).
    """
    return {}


async def evaluate(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """ADR-004: deterministic, no LLM. Step 1.7 fills in the thresholds."""
    results = state.get("results") or []
    plan_obj = state.get("plan")
    task_count = len(plan_obj.tasks) if plan_obj else len(results)
    failed = sum(1 for r in results if r.status == "failed")
    empty = sum(1 for r in results if r.status != "failed" and not r.produced_something)
    report = EvaluationReport(
        task_count=task_count,
        ok_count=sum(1 for r in results if r.status == "ok"),
        failed_count=failed,
        empty_count=empty,
        passed=not (failed or empty),
    )
    return {"evaluation": report, "degraded": not report.passed}


async def synthesize(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Step 1.7. A degraded synthesis must SAY it is degraded rather than
    quietly returning a shorter answer."""
    return {}
