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
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from dynaflows.contracts.calls import CallRequest
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.playbook import ContextPack
from dynaflows.contracts.state import (
    MAX_FANOUT,
    EvaluationReport,
    GateDecision,
    GateOutcome,
    PlanTask,
    WorkerResult,
    WorkflowState,
    cost_delta,
)
from dynaflows.contracts.tiers import Tier
from dynaflows.graph import planner as planning
from dynaflows.graph.capabilities import render_catalogue
from dynaflows.graph.deps import (
    auto_approved,
    gateway_from,
    playbook_from,
    source_root_from,
    store_from,
)
from dynaflows.graph.prompts import (
    ENHANCER_SYSTEM,
    PLANNER_SYSTEM,
    WORKER_SYSTEM,
    EnhancedPrompt,
    PlanDraft,
    WorkerReport,
)
from dynaflows.playbook.pack import pack
from dynaflows.store.catalogue import build_catalogue
from dynaflows.store.sources import SourceRefusal, read_sources


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
            # A rewritten brief plus a few assumptions. Reserving more is not
            # free: providers price a request as prompt + the FULL allowance.
            max_tokens=1024,
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
    # ADR-018: the planner has to see the code it is planning against. Before
    # this it saw the playbook and nothing else, so `inputs` was guesswork --
    # right by luck on one run, abandoned entirely on the next.
    catalogue = build_catalogue(Path(source_root_from(config)))
    system = PLANNER_SYSTEM.format(
        max_fanout=MAX_FANOUT,
        capabilities=render_catalogue(),
        catalogue=repository.catalog(),
        sources=catalogue.render(),
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
                # 12 tasks x objective + anchors, plus a rationale.
                max_tokens=4096,
                # A different variant, so the re-plan is a fresh call and not
                # a cache hit on the answer that was just rejected (ADR-013).
                variant=attempt - 1,
                label="plan",
                metadata=(("run_id", state.get("run_id", "")), ("node", "plan")),
            )
        )
        ledger = _merge_delta(ledger, result)
        draft: PlanDraft = result.payload
        correction = planning.violation_of(draft, catalogue) or ""
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


# PLACEHOLDER (playbook 4.5). Sized so twelve workers at MAX_FANOUT stay well
# inside the 32,000-token floor `doctor` enforces, with room for the objective,
# the system prompt and the answer. Step 1.8 replaces it with a measurement.
WORKER_CONTEXT_BUDGET = 6_000


async def worker(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """One analysis task. Dispatched by Send; returns a DELTA on reduced keys.

    ADR-017: the context is assembled HERE from things already resolved, not
    fetched by the model. Playbook anchors go into the pack before source
    files, so a tight budget drops the code before it drops the rules the code
    is being judged against -- and `dropped_ids` says which, in the trace.

    A worker must never raise. An exception inside a Send branch aborts the
    whole superstep, turning "one of twelve failed" into "the run is gone", so
    every failure below becomes a `failed` WorkerResult that the evaluator can
    count (ADR-004) instead of an exception nobody catches.
    """
    task = state.get("task")
    if not isinstance(task, PlanTask):
        # Not reachable through dispatch_workers, and that is exactly why it
        # is handled: an unreachable branch that returns {} would erase nothing
        # and report nothing, which is the quietest possible bug.
        return {}

    ledger = cost_delta()
    try:
        gateway = gateway_from(config)
        repository = playbook_from(config)
        store = store_from(config)
        root = Path(source_root_from(config))
    except DynaflowsError as exc:
        return {"results": [_failed(task, exc)], "cost": ledger}

    playbook_chunks = repository.by_anchor(list(task.playbook_anchors))
    source_chunks, refusals = read_sources(list(task.inputs), root)
    context = pack([*playbook_chunks, *source_chunks], WORKER_CONTEXT_BUDGET)

    prompt = _worker_prompt(task, context, refusals, sources=len(source_chunks))
    try:
        result = await gateway.call(
            CallRequest(
                tier=task.tier_override or Tier.MID,
                system=WORKER_SYSTEM,
                prompt=prompt,
                schema=WorkerReport,
                max_tokens=2048,
                label=f"worker.{task.task_id}",
                metadata=(
                    ("run_id", state.get("run_id", "")),
                    ("node", "worker"),
                    ("task_id", task.task_id),
                    ("capability", task.capability),
                    # ADR-011: what this worker saw and what it did not, in the
                    # trace, so "why did it miss that" is answerable without
                    # re-running anything.
                    ("context_pack_tokens", str(context.tokens)),
                    ("context_dropped_ids", ",".join(context.dropped_ids)),
                    ("source_refusals", ",".join(r.render() for r in refusals)),
                ),
            )
        )
    except DynaflowsError as exc:
        return {"results": [_failed(task, exc)], "cost": ledger}
    except Exception as exc:  # noqa: BLE001 -- see the docstring: never raise
        return {
            "results": [_failed(task, DynaflowsError.of(ErrorCode.UNKNOWN, str(exc)))],
            "cost": ledger,
        }

    ledger = _merge_delta(ledger, result)
    report: WorkerReport = result.payload

    try:
        artifact = store.write(state.get("run_id", "unknown"), task.task_id, report.findings)
    except OSError as exc:
        # The analysis succeeded and the disk did not. Degraded, not failed:
        # the summary is still in state and still worth synthesising, and
        # saying "failed" here would discard work already paid for.
        return {
            "results": [
                WorkerResult(
                    task_id=task.task_id,
                    status="degraded",
                    summary=f"{report.summary}\n\n(report not saved: {exc})",
                    model_id=result.model_id,
                    tier=result.tier,
                    fallback_depth=result.fallback_depth,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost_usd=result.cost_usd,
                )
            ],
            "cost": ledger,
        }

    # A model that says it lacked context is reporting a real limitation, and
    # ADR-004 needs that separable from a clean success. `degraded` is what the
    # evaluator counts; hiding it as `ok` is how a run passes while telling the
    # user nothing.
    # A worker that wrote nothing did not succeed, whatever it says about its
    # context. Run `w2`: one worker produced a zero-byte findings file and
    # three produced one sentence each, and three of the four reported `ok`.
    wrote_nothing = not report.findings.strip()
    incomplete = (
        not report.context_was_sufficient
        or bool(context.dropped_ids)
        or bool(refusals)
        or wrote_nothing
    )
    return {
        "results": [
            WorkerResult(
                task_id=task.task_id,
                status="degraded" if incomplete else "ok",
                summary=report.summary,
                artifact=artifact,
                model_id=result.model_id,
                tier=result.tier,
                fallback_depth=result.fallback_depth,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cost_usd=result.cost_usd,
            )
        ],
        "cost": ledger,
    }


def _failed(task: PlanTask, exc: DynaflowsError) -> WorkerResult:
    """One failure shape, so eleven siblings keep going and the twelfth is
    visible rather than merely absent."""
    return WorkerResult(
        task_id=task.task_id,
        status="failed",
        summary="",
        error=exc.envelope,
    )


def _worker_prompt(
    task: PlanTask,
    context: ContextPack,
    refusals: list[SourceRefusal],
    *,
    sources: int,
) -> str:
    """What the worker is shown, including what it is NOT shown.

    The refusals and drops are in the prompt, not only in the trace. A model
    told "these files were withheld" reports a gap; a model shown a silently
    shorter context reports confidently on a subset (AP-19, applied to a
    prompt).
    """
    parts = [f"OBJECTIVE\n{task.objective}"]
    if task.inputs:
        parts.append("INPUTS REQUESTED\n" + "\n".join(f"- {i}" for i in task.inputs))
    if refusals:
        parts.append(
            "NOT AVAILABLE (do not speculate about these)\n"
            + "\n".join(f"- {r.render()}" for r in refusals)
        )
    if context.dropped_ids:
        parts.append(
            f"CONTEXT TRUNCATED: {len(context.dropped_ids)} section(s) did not fit the budget."
        )
    if not sources:
        # The run that motivated this: five workers were given no source at
        # all, four said so, and the fifth invented four filenames, six
        # findings and a set of line numbers, one of them HIGH severity. The
        # honest instruction has to be explicit, because "audit the gateway
        # package" reads like permission to describe what such a package
        # usually contains.
        parts.append(
            "NO SOURCE FILES WERE PROVIDED.\n"
            "You have not seen any code. Do not name files, symbols or line "
            "numbers you were not shown, and do not describe what such code "
            "usually looks like. Report that you could not examine anything, "
            "set context_was_sufficient to false, and say what you would need."
        )
    parts.append("CONTEXT\n" + (context.text or "(nothing was retrievable)"))
    return "\n\n".join(parts)


async def evaluate(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """ADR-004: deterministic, no LLM. Step 1.7 fills in the thresholds.

    The rules below each exist because a run passed when it should not have.
    They are checked separately and reported separately (AP-20): "nothing came
    back", "everything came back broken" and "everything came back hedged" are
    three different failures needing three different answers, and one `passed`
    flag with one reason string answers none of them.
    """
    results = state.get("results") or []
    plan_obj = state.get("plan")
    task_count = len(plan_obj.tasks) if plan_obj else len(results)
    failed = sum(1 for r in results if r.status == "failed")
    degraded = sum(1 for r in results if r.status == "degraded")
    empty = sum(1 for r in results if r.status != "failed" and not r.produced_something)
    ok = sum(1 for r in results if r.status == "ok")

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

    # ADR-004 rule 2: a run where nothing succeeded did not succeed.
    #
    # The second vacuous pass, and it shipped in the commit that INTRODUCED
    # `degraded`. Five degraded results counted as zero ok and zero failed, so
    # `reasons` was empty and the run passed -- while one of those five workers
    # had invented four source files and reported a HIGH severity finding
    # against them. "Degraded" has to cost something or it is a synonym for
    # "fine".
    if task_count and not ok:
        reasons.append(
            f"no task succeeded cleanly ({degraded} degraded, {failed} failed of {task_count})"
        )
    elif degraded:
        # Some succeeded, some did not. Not a failure, but never silent: the
        # degraded ones are exactly the results a reader must not trust
        # equally, and the synthesizer says so too (step 1.7).
        reasons.append(f"{degraded} of {task_count} task(s) degraded")

    report = EvaluationReport(
        task_count=task_count,
        ok_count=ok,
        failed_count=failed,
        empty_count=empty,
        degraded_count=degraded,
        passed=not reasons,
        reasons=reasons,
    )
    return {"evaluation": report, "degraded": not report.passed}


async def synthesize(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Step 1.7. A degraded synthesis must SAY it is degraded rather than
    quietly returning a shorter answer."""
    return {}
