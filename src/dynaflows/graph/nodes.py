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
    MAX_FANOUT,
    EvaluationReport,
    GateDecision,
    GateOutcome,
    WorkflowState,
    cost_delta,
)
from dynaflows.contracts.tiers import Tier
from dynaflows.graph import planner as planning
from dynaflows.graph.capabilities import render_catalogue
from dynaflows.graph.deps import auto_approved, gateway_from, playbook_from
from dynaflows.graph.prompts import (
    ENHANCER_SYSTEM,
    PLANNER_SYSTEM,
    EnhancedPrompt,
    PlanDraft,
)


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
    """Decompose the approved brief into independent tasks. Frontier tier.

    ADR-006's cost-asymmetry rule lands here: this call happens ONCE and
    decides the cost and correctness of N worker calls, so it gets the
    expensive model while the workers get the cheap one.

    ADR-009: the planner is also the context router. It picks
    `playbook_anchors` per task from the catalogue, so one frontier call
    decides what every worker will be shown.
    """
    gateway = gateway_from(config)
    repository = playbook_from(config)
    system = PLANNER_SYSTEM.format(
        max_fanout=MAX_FANOUT,
        capabilities=render_catalogue(),
        catalogue=repository.catalog(),
    )

    brief = state.get("enhanced_prompt") or state.get("raw_prompt", "")
    ledger = cost_delta()
    correction = ""

    # ADR-014: one re-plan, then stop. Never a silent trim -- dropping tasks
    # for capacity produces a synthesis that is incomplete without saying so.
    for attempt in (1, 2):
        prompt = brief if not correction else f"{brief}\n\nPREVIOUS ATTEMPT FAILED: {correction}"
        result = await gateway.call(
            CallRequest(
                tier=Tier.FRONTIER,
                system=system,
                prompt=prompt,
                schema=PlanDraft,
                # A different variant, so the re-plan is a fresh call and not
                # a cache hit on the answer that was just rejected (ADR-013).
                variant=attempt - 1,
                label="plan",
                metadata=(("run_id", state.get("run_id", "")), ("node", "plan")),
            )
        )
        ledger = _merge_delta(ledger, result)
        draft: PlanDraft = result.payload
        correction = planning.violation_of(draft) or ""
        if not correction:
            plan_obj = planning.draft_to_plan(draft)
            plan_obj.estimated_tokens = planning.estimate_tokens(plan_obj)
            return {
                "plan": plan_obj,
                "plan_hash": planning.plan_hash(plan_obj, _models_version(gateway)),
                "cost": ledger,
            }

    # Both attempts failed. The human sees the reason at G2 rather than a
    # trimmed plan that looks fine.
    return {"plan": None, "plan_rejected_reason": correction, "cost": ledger}


def _models_version(gateway: Any) -> str:
    registry = getattr(gateway, "registry", None)
    return str(getattr(registry, "version", "unknown"))


def _merge_delta(ledger: Any, result: Any) -> Any:
    from dynaflows.contracts.state import merge_cost

    return merge_cost(
        ledger,
        cost_delta(
            usd_spent=0.0 if result.cache_hit else result.cost_usd,
            usd_avoided=result.cost_usd if result.cache_hit else 0.0,
            tokens_in=0 if result.cache_hit else result.tokens_in,
            tokens_out=0 if result.cache_hit else result.tokens_out,
            calls_made=0 if result.cache_hit else 1,
            calls_cached=1 if result.cache_hit else 0,
        ),
    )


async def approve_plan(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Gate G2 -- the load-bearing gate (ADR-005). PURE, like G1.

    The fan-out is the irreversible spend. Approving a well-worded brief tells
    you nothing about the twelve workers about to run on the wrong twelve
    files, which is why this gate shows the task list rather than a summary.
    """
    plan_obj = state.get("plan")
    if plan_obj is None:
        # Two failed planning attempts. Stop and say why (ADR-014).
        return {
            "plan_gate": GateOutcome(
                decision=GateDecision.REJECT,
                note=state.get("plan_rejected_reason") or "the planner produced no usable plan",
            ),
            "halted": "no usable plan after two attempts",
        }

    if auto_approved(config, "plan"):
        return {"plan_gate": GateOutcome(decision=GateDecision.APPROVE, note="--yes-plan")}

    answer = interrupt(
        {
            "gate": "plan",
            "rationale": plan_obj.rationale,
            "plan_hash": state.get("plan_hash"),
            "estimated_tokens": plan_obj.estimated_tokens,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "capability": t.capability,
                    "objective": t.objective,
                    "inputs": list(t.inputs),
                    "anchors": list(t.playbook_anchors),
                }
                for t in plan_obj.tasks
            ],
        }
    )
    outcome = _gate_outcome(answer)
    update: dict[str, Any] = {"plan_gate": outcome}
    if outcome.decision is GateDecision.REJECT:
        update["halted"] = "rejected by human at gate G2"
    return update


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

    reasons: list[str] = []

    # ADR-004 rule 1: every plan task has exactly one result.
    #
    # Checked first and separately, because its absence is the only way to
    # report a VACUOUS pass -- five planned tasks, zero results, and
    # `passed=True` because nothing came back to fail. A missing result is not
    # a silent success; it means a branch never ran or never returned, which
    # is strictly worse than one that failed and said so.
    missing = task_count - len(results)
    if missing > 0:
        reasons.append(f"{missing} of {task_count} task(s) produced no result at all")
    elif missing < 0:
        reasons.append(f"{-missing} more result(s) than planned tasks")

    if failed:
        reasons.append(f"{failed} task(s) failed")
    if empty:
        reasons.append(f"{empty} task(s) returned nothing usable")

    report = EvaluationReport(
        task_count=task_count,
        ok_count=sum(1 for r in results if r.status == "ok"),
        failed_count=failed,
        empty_count=empty,
        passed=not reasons,
        reasons=reasons,
    )
    return {"evaluation": report, "degraded": not report.passed}


async def synthesize(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Step 1.7. A degraded synthesis must SAY it is degraded rather than
    quietly returning a shorter answer."""
    return {}
