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
from langgraph.types import interrupt

from dynaflows.contracts.calls import CallRequest
from dynaflows.contracts.state import (
    EvaluationReport,
    GateDecision,
    GateOutcome,
    WorkflowState,
    cost_delta,
)
from dynaflows.contracts.tiers import Tier
from dynaflows.graph.deps import auto_approved, gateway_from
from dynaflows.graph.prompts import ENHANCER_SYSTEM, EnhancedPrompt


async def enhance_prompt(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Rewrite the raw request into a precise brief. Small tier (ADR-006).

    The LLM call belongs HERE and never in the gate that follows. ADR-007: a
    resumed graph re-runs the interrupted node from its first line, so an
    enhancer sharing a node with its gate would be paid for again on every
    resume -- and could show the human different text than the text they were
    approving.
    """
    gateway = gateway_from(config)
    result = await gateway.call(
        CallRequest(
            tier=Tier.SMALL,
            system=ENHANCER_SYSTEM,
            prompt=state["raw_prompt"],
            schema=EnhancedPrompt,
            label="enhance_prompt",
            metadata=(("run_id", state.get("run_id", "")), ("node", "enhance_prompt")),
        )
    )
    payload: EnhancedPrompt = result.payload
    return {
        "enhanced_prompt": payload.enhanced,
        "enhancer_assumptions": list(payload.assumptions),
        "cost": cost_delta(
            usd_spent=0.0 if result.cache_hit else result.cost_usd,
            usd_avoided=result.cost_usd if result.cache_hit else 0.0,
            tokens_in=0 if result.cache_hit else result.tokens_in,
            tokens_out=0 if result.cache_hit else result.tokens_out,
            calls_made=0 if result.cache_hit else 1,
            calls_cached=1 if result.cache_hit else 0,
        ),
    }


async def approve_prompt(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Gate G1 (ADR-005). PURE: it reads state, asks, and writes the answer.

    Nothing else may ever live in this node. A resumed graph re-runs it from
    line one, so any I/O here would be repeated on every resume -- and
    `scripts/lint_architecture.py` now rejects an `await` in any function that
    calls `interrupt()`, so the rule is enforced rather than remembered.
    """
    if auto_approved(config, "prompt"):
        return {"prompt_gate": GateOutcome(decision=GateDecision.APPROVE, note="--yes-prompt")}

    answer = interrupt(
        {
            "gate": "prompt",
            "original": state.get("raw_prompt", ""),
            "enhanced": state.get("enhanced_prompt", ""),
            "assumptions": list(state.get("enhancer_assumptions") or []),
        }
    )
    outcome = _gate_outcome(answer)
    update: dict[str, Any] = {"prompt_gate": outcome}
    if outcome.decision is GateDecision.EDIT and outcome.replacement:
        # The human's text beats the model's. Replacing it here rather than
        # re-asking the model is the entire point of offering "edit".
        update["enhanced_prompt"] = outcome.replacement
    if outcome.decision is GateDecision.REJECT:
        update["halted"] = "rejected by human at gate G1"
    return update


def _gate_outcome(answer: Any) -> GateOutcome:
    """Normalise whatever `Command(resume=...)` carried.

    A bare string is accepted so a human answering "approve" at a terminal is
    not a crash, but anything unrecognised is a REJECT: defaulting an
    unparseable answer to approval would let a typo authorise a fan-out.
    """
    if isinstance(answer, GateOutcome):
        return answer
    if isinstance(answer, dict):
        return GateOutcome.model_validate(answer)
    if isinstance(answer, str) and answer.strip().lower() in set(GateDecision):
        return GateOutcome(decision=GateDecision(answer.strip().lower()))
    return GateOutcome(decision=GateDecision.REJECT, note=f"unparseable gate answer: {answer!r}")


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
