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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from dynaflows.contracts.calls import CallRequest
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.playbook import Chunk, ContextPack
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
from dynaflows.graph import grounding as grounding_check
from dynaflows.graph import planner as planning
from dynaflows.graph import synthesis as synth
from dynaflows.graph.budgets import worker_context_budget
from dynaflows.graph.capabilities import render_capabilities
from dynaflows.graph.deps import (
    auto_approved,
    gateway_from,
    playbook_from,
    source_root_from,
    store_from,
)
from dynaflows.graph.grounding import Grounding
from dynaflows.graph.prompts import (
    ANSWER_SYSTEM,
    ENHANCER_SYSTEM,
    PLANNER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    WORKER_SYSTEM,
    AnswerReport,
    EnhancedPrompt,
    Finding,
    Observation,
    PlanDraft,
    SynthesisDraft,
    WorkerReport,
    asks_a_question,
)
from dynaflows.playbook.pack import pack_sections
from dynaflows.store.source_map import build_source_map, is_test_path, unknown_paths
from dynaflows.store.sources import SourceRefusal, read_sources

_MAX_RELEVANT_PATHS = 6


def _settled_intent(raw_prompt: str, claimed: str) -> str:
    """The model's reading of the request, corrected by its grammar.

    A 2.6B model labelled "how does the playbook index avoid returning stale
    results after a file changes" as `work` on the first live run. Approving
    that would have told the planner to audit the retrieval code for defects
    rather than explain it, at frontier-tier prices for an answer to a
    question nobody asked.

    `intent` is a field a model fills, which makes it a request. Whether a
    sentence opens with "how does" is a fact about the sentence, so it is
    checked. One way only -- see `asks_a_question`.
    """
    if claimed != "question" and asks_a_question(raw_prompt):
        return "question"
    return claimed


def _useful_paths(paths: list[str]) -> list[str]:
    """Trim a grounded path list to the ones worth showing a human.

    The prompt asks for two to six and no test modules. A prompt is a request,
    not a guarantee -- this project has the scar tissue -- so the rule is
    enforced where it can be, and a run that named eleven paths including
    `PROJECT_MEMORY.md` and three test modules is what prompted it.

    Test modules go first rather than last: a test module tells you where the
    thing under test is ASSERTED, never where it lives, so it is the least
    useful entry in a list whose whole job is to say where to look. Order is
    otherwise preserved -- the model put its best guess first, and re-ranking
    it here would be this function inventing an opinion it does not have.
    """
    kept = [p for p in paths if not is_test_path(p)]
    # Unless they were all tests, in which case the model was probably right
    # about the subject and dropping everything would say less than saying so.
    return (kept or paths)[:_MAX_RELEVANT_PATHS]


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
    # ADR-023 step A: the enhancer was blind to the codebase. The PLANNER got
    # the source catalogue in ADR-018 and the enhancer never did, so "fix the
    # login bug" was sharpened into generic precision instead of naming where
    # login lives in this repository -- which is the one thing a rewrite can
    # add that the developer could not have typed faster themselves.
    catalogue = build_source_map(Path(source_root_from(config)))
    result = await gateway.call(
        CallRequest(
            tier=Tier.SMALL,
            system=ENHANCER_SYSTEM.format(sources=catalogue.render()),
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
    # Checked against the catalogue, exactly as a plan's inputs are (ADR-018),
    # but NOT fatal: a wrong path in a brief is a bad suggestion, not a plan
    # that cannot run. The unknown ones are dropped and named at G1, so the
    # human sees what it invented rather than approving it silently.
    invented = unknown_paths(list(payload.relevant_paths), catalogue)
    grounded = _useful_paths([p for p in payload.relevant_paths if p not in invented])
    return {
        # The SHARPENED REQUEST, and nothing else. `compose_brief` appends the
        # standing requirements at the boundary where the brief is handed to
        # something that can act on them -- see `cli.brief`.
        #
        # It used to be applied here, and the requirements travelled into
        # state, which meant into the PLANNER, which meant into every task
        # objective. Live run q3 produced three tasks each told to "run
        # focused checks in an external scratch directory" and one built
        # entirely around indexing a file, changing it and re-querying --
        # handed to a worker that is a single LLM call with no tools
        # (`test_the_worker_never_receives_a_file_tool`). A worker told to
        # report observed outputs it has no way to observe invents them, and
        # the citation check cannot catch that: it verifies `quoted_lines`
        # against the packed source and never sees the prose.
        #
        # The brief has two readers. One can run commands; the other cannot.
        # Executor policy belongs only to the first.
        "enhanced_prompt": payload.enhanced,
        "enhancer_model": result.model_id,
        "enhancer_intent": _settled_intent(state.get("raw_prompt", ""), payload.intent),
        "enhancer_intent_claimed": payload.intent,
        "relevant_paths": grounded,
        "invented_paths": invented,
        "enhancer_assumptions": list(payload.assumptions),
        "cost": delta_for(result),
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
            "relevant_paths": list(state.get("relevant_paths") or []),
            "invented_paths": list(state.get("invented_paths") or []),
            "model": state.get("enhancer_model", ""),
            "intent": state.get("enhancer_intent", ""),
            "intent_claimed": state.get("enhancer_intent_claimed", ""),
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
    catalogue = build_source_map(Path(source_root_from(config)))
    system = PLANNER_SYSTEM.format(
        max_fanout=MAX_FANOUT,
        capabilities=render_capabilities(),
        section_map=repository.section_map(),
        sources=catalogue.render(),
    )

    # ADR-024 divided the WORKERS by whether a claim reports a defect or
    # describes the code. The planner picks that capability, and until now it
    # inferred the user's intent from the brief's wording. Stating it is
    # cheaper and far more reliable than hoping the wording carries it.
    intent = state.get("enhancer_intent") or "question"
    brief = state.get("enhanced_prompt") or state.get("raw_prompt", "")
    brief = f"THE USER ASKED A {intent.upper()}.\n\n{brief}"
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
            plan_obj = planning.measure_context(
                plan_obj,
                repository,
                Path(source_root_from(config)),
                worker_context_budget(getattr(gateway, "registry", None)),
            )
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


def delta_for(result: Any) -> Any:
    """One call's contribution to the ledger. The ONLY place this is written.

    It used to be a hand-listed set of six fields here and another in the
    enhancer, and both omitted `fallbacks`. Live run `q4` ran both workers on
    `openai/gpt-oss-20b` -- mid[3], three models past `llama-4-scout`, the
    only model in that chain anyone has calibrated -- and reported
    `fallbacks=0`. The run silently used an unmeasured model and the figure
    that would have said so read zero.

    `merge_cost` had exactly this fault and was fixed by enumerating
    `dataclasses.fields()` instead of a list. The list it replaced had a
    sibling one file away, and nobody looked. **A rule that fixes one hand-
    written list should send you looking for the others.**
    """
    from dynaflows.contracts.state import cost_delta as _delta

    if result.cache_hit:
        return _delta(
            usd_avoided=result.cost_usd or 0.0,
            calls_cached=1,
            calls_unpriced=1 if result.cost_usd is None else 0,
        )
    return _delta(
        usd_spent=result.cost_usd or 0.0,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        calls_made=1,
        calls_unpriced=1 if result.cost_usd is None else 0,
        # The two the hand-written version dropped. `fallback_depth` is how
        # many models in the chain were tried and did not answer; a depth of
        # three means the head and two others were skipped or failed, which
        # decides whether a result can be attributed to the model you think
        # you are measuring.
        fallbacks=1 if result.fallback_depth else 0,
        calls_attempted=result.fallback_depth,
    )


def _merge_delta(ledger: Any, result: Any) -> Any:
    from dynaflows.contracts.state import merge_cost

    return merge_cost(ledger, delta_for(result))


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
            "context_budget": plan_obj.context_budget,
            "over_budget": planning.over_budget(plan_obj),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "capability": t.capability,
                    "objective": t.objective,
                    "inputs": list(t.inputs),
                    "anchors": list(t.playbook_anchors),
                    "context_tokens": t.context_tokens,
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


# What `WorkerResult.summary` accepts. Read from the model rather than
# repeated, because the last time this number lived in two places they
# disagreed and a worker took the run down with it.
_SUMMARY_LIMIT = WorkerResult.model_fields["summary"].metadata[0].max_length


def _fit(summary: str) -> str:
    """Trim a summary to what state will accept, saying so when it trims.

    Belt to the schema's braces. `AnswerReport.answer` is capped at the same
    number now, so this should never fire -- but the failure it guards against
    was an exception inside a Send branch, which does not fail one task, it
    ends the run. A silent truncation is a much smaller harm than that, and
    the marker keeps it from being silent.
    """
    if len(summary) <= _SUMMARY_LIMIT:
        return summary
    marker = " […trimmed; full text in the artifact]"
    return summary[: _SUMMARY_LIMIT - len(marker)] + marker


@dataclass(frozen=True, slots=True)
class _Analysed:
    """One worker's output, with the capability-specific shape already gone.

    The worker node does exactly one thing differently per capability: it asks
    a different model a different question and checks the reply by a different
    substance rule. Everything after that -- degraded-vs-ok, the artifact, the
    counters, the cost -- is identical, and duplicating it per capability is
    how the two drift until one of them silently stops counting something.
    """

    summary: str
    examined: list[str]
    context_was_sufficient: bool
    reported: int
    grounded: int
    all_ungrounded: bool
    document: str
    data: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CapabilityHandler:
    system: str
    schema: type[BaseModel]
    normalise: Callable[[Any, list[Chunk]], _Analysed]


def _analysed_findings(report: WorkerReport, seen: list[Chunk]) -> _Analysed:
    result = grounding_check.verify(report.findings, seen)
    return _Analysed(
        summary=_fit(report.summary),
        examined=list(report.examined),
        context_was_sufficient=report.context_was_sufficient,
        reported=result.reported,
        grounded=len(result.kept),
        all_ungrounded=result.all_ungrounded,
        document=_render_findings(report, result),
        data=[f.model_dump() for f in result.kept],
    )


def _analysed_answer(report: AnswerReport, seen: list[Chunk]) -> _Analysed:
    result = grounding_check.verify_observations(report.observations, seen)
    return _Analysed(
        summary=_fit(report.answer),
        examined=list(report.examined),
        context_was_sufficient=report.context_was_sufficient,
        reported=result.reported,
        grounded=len(result.kept),
        all_ungrounded=result.all_ungrounded,
        document=_render_answer(report, result),
        data=[o.model_dump() for o in result.kept],
    )


CAPABILITY_HANDLERS: dict[str, CapabilityHandler] = {
    "analyse": CapabilityHandler(WORKER_SYSTEM, WorkerReport, _analysed_findings),
    "answer": CapabilityHandler(ANSWER_SYSTEM, AnswerReport, _analysed_answer),
}


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
    # Two shares of one budget, not one ordered list. Passing the playbook
    # first so a tight budget drops code before rules sounded careful and
    # starved run `s1`'s workers of six of their seven files (ADR-021).
    context = pack_sections(
        playbook_chunks, source_chunks, worker_context_budget(getattr(gateway, "registry", None))
    )

    # ADR-001: the capability is validated plan data, so this is a lookup and
    # never a branch on free text. A task whose capability has no handler
    # cannot exist -- PlanTask rejects it -- and the test that asserts the
    # catalogue equals the handler table is what keeps that true.
    handler = CAPABILITY_HANDLERS[task.capability]
    prompt = _worker_prompt(task, context, refusals, sources=len(source_chunks))
    try:
        result = await gateway.call(
            CallRequest(
                tier=task.tier_override or Tier.MID,
                system=handler.system,
                prompt=prompt,
                schema=handler.schema,
                # 4,096, not 2,048. A model in the small chain spent 2,297
                # tokens reasoning against a 2,048 cap and returned nothing
                # parseable at all -- on a reasoning model the thinking counts
                # against this number and is invisible until it is gone. The
                # capable worker measured in ADR-006's amendment claimed four
                # findings inside this allowance.
                max_tokens=4096,
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

    # ADR-019. Checked against what this worker ACTUALLY saw -- the packed
    # chunks, after truncation -- not what it was meant to see. A claim about a
    # section dropped for budget is ungrounded from this worker's point of
    # view, which is the honest reading: it could not have read what it quotes.
    seen = [c for c in (*playbook_chunks, *source_chunks) if c.id in context.included_ids]
    analysed = handler.normalise(result.payload, seen)

    run_id = state.get("run_id", "unknown")
    try:
        artifact = store.write(run_id, task.task_id, analysed.document)
        # ADR-020: the synthesizer needs claims, not prose it has to parse back
        # out of markdown. Written here, beside the human report, because state
        # carries references and not payloads (ADR-008).
        findings_ref = store.write_data(run_id, task.task_id, analysed.data)
    except OSError as exc:
        # The analysis succeeded and the disk did not. Degraded, not failed:
        # the summary is still in state and still worth synthesising, and
        # saying "failed" here would discard work already paid for.
        return {
            "results": [
                WorkerResult(
                    task_id=task.task_id,
                    capability=task.capability,
                    status="degraded",
                    summary=f"{analysed.summary}\n\n(report not saved: {exc})",
                    model_id=result.model_id,
                    tier=result.tier,
                    fallback_depth=result.fallback_depth,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost_usd=result.cost_usd,
                    findings_reported=analysed.reported,
                    findings_grounded=analysed.grounded,
                )
            ],
            "cost": ledger,
        }

    # Four separate reasons to distrust this result, kept apart rather than
    # collapsed into one flag (AP-20). `degraded` is what ADR-004 counts, and
    # hiding any of these as `ok` is how a run passes while telling the user
    # nothing -- which is exactly what runs w1 and w2 did.
    looked_at_nothing = not analysed.examined
    everything_invented = analysed.all_ungrounded
    # An ANSWER with no surviving citation is the failure this capability
    # exists to prevent, and it was passing as `ok`.
    #
    # Live run q4: the worker that wrote unsupported prose scored `ok`, and
    # the worker that produced one verified citation scored `degraded`. The
    # artifact for the first one says "None verified. The answer above is
    # unsupported." while its status said the run was fine.
    #
    # `all_ungrounded` is false when a worker claimed nothing, which is
    # correct for `analyse` -- "I read these and found no defects" is a
    # legitimate result. It is not a legitimate ANSWER. A question was asked
    # and prose came back with nothing behind it, which is exactly the
    # `answered_from_priors` shape ADR-022's fixture measures.
    #
    # Same family as this project's first bug: `passed=True` while five
    # workers fabricated, because the fact had no counter.
    unsupported_answer = task.capability == "answer" and not analysed.grounded
    incomplete = (
        not analysed.context_was_sufficient
        or bool(context.dropped_ids)
        or bool(refusals)
        or looked_at_nothing
        or everything_invented
        or unsupported_answer
    )
    try:
        return {
            "results": [
                _ok_result(task, analysed, result, artifact, findings_ref, incomplete=incomplete)
            ],
            "cost": ledger,
        }
    except ValidationError as exc:
        # The docstring at the top promises this node never raises, and until
        # a live run proved otherwise that promise was enforced only around
        # the gateway call -- every `except` above is upstream of here. A
        # capped field disagreeing with its source got past all of them and
        # killed the superstep from the last statement in the function.
        #
        # A contract violation in OUR OWN assembly is a bug in dynaflows, not
        # a failed analysis, so it is reported as one rather than dressed up
        # as a worker result.
        return {
            "results": [
                _failed(task, DynaflowsError.of(ErrorCode.UNKNOWN, f"result rejected: {exc}"))
            ],
            "cost": ledger,
        }


def _ok_result(
    task: PlanTask,
    analysed: _Analysed,
    result: Any,
    artifact: Any,
    findings_ref: Any,
    *,
    incomplete: bool,
) -> WorkerResult:
    return WorkerResult(
        task_id=task.task_id,
        capability=task.capability,
        status="degraded" if incomplete else "ok",
        summary=analysed.summary,
        artifact=artifact,
        model_id=result.model_id,
        tier=result.tier,
        fallback_depth=result.fallback_depth,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        findings_reported=analysed.reported,
        findings_grounded=analysed.grounded,
        findings_ref=findings_ref,
    )


def _render_findings(report: WorkerReport, grounding: Grounding[Finding]) -> str:
    """The artifact. Verified findings, and what was discarded.

    The discarded ones are written down rather than dropped silently. A worker
    that made six claims and had all six thrown out is a very different thing
    from one that made none, and a reader who cannot see the difference will
    read the second as diligence.
    """
    lines = [f"# {report.summary}", ""]
    lines.append("## Examined")
    lines.extend(f"- {item}" for item in report.examined or ["(nothing recorded)"])
    lines.append("")
    lines.append(f"## Findings ({len(grounding.kept)} verified)")
    if not grounding.kept:
        lines.append("_None._")
    for finding in grounding.kept:
        lines.append(f"### [{finding.severity}] {finding.claim}")
        lines.append(f"`{finding.file}:{finding.lines}`")
        lines.append("")
        lines.append("```")
        lines.append(finding.quoted_lines)
        lines.append("```")
        lines.append(f"**Remediation.** {finding.remediation}")
        lines.append("")
    if grounding.dropped:
        lines.append(f"## Discarded as ungrounded ({len(grounding.dropped)})")
        for dropped in grounding.dropped:
            lines.append(f"- {dropped.render()}: {dropped.finding.claim}")
            # The QUOTE, not just the verdict. The first version of this said
            # "quoted evidence does not appear" and did not say what was
            # quoted, so a run where all seven findings were discarded could
            # not be diagnosed at all -- the same defect as an error message
            # that names no cause.
            lines.append("")
            lines.append("  ```")
            lines.extend(f"  {line}" for line in dropped.finding.quoted_lines.splitlines()[:12])
            lines.append("  ```")
        lines.append("")
    if report.missing:
        lines.append("## Not available to this worker")
        lines.extend(f"- {item}" for item in report.missing)
    return "\n".join(lines).rstrip() + "\n"


def _render_answer(report: AnswerReport, grounding: Grounding[Observation]) -> str:
    """The artifact for an `answer` task.

    The answer comes FIRST and the citations follow it. A findings report is a
    list and reads as one; an answer is prose, and burying it under a table of
    evidence makes a reader reconstruct it. The discarded citations are still
    written down, for the same reason as in `_render_findings`: a worker whose
    every citation failed is not a worker that cited nothing, and a reader who
    cannot see the difference reads the second as diligence.
    """
    lines = [f"# {report.answer}", ""]
    lines.append("## Examined")
    lines.extend(f"- {item}" for item in report.examined or ["(nothing recorded)"])
    lines.append("")
    lines.append(f"## Evidence ({len(grounding.kept)} verified)")
    if not grounding.kept:
        # Prose with no surviving citation is the shape a fabricated answer
        # takes, so it is labelled rather than left to look like brevity.
        lines.append("_None verified. The answer above is unsupported._")
    for observation in grounding.kept:
        lines.append(f"### {observation.claim}")
        lines.append(f"`{observation.file}:{observation.lines}`")
        lines.append("")
        lines.append("```")
        lines.append(observation.quoted_lines)
        lines.append("```")
        lines.append("")
    if grounding.dropped:
        lines.append(f"## Discarded as ungrounded ({len(grounding.dropped)})")
        for dropped in grounding.dropped:
            lines.append(f"- {dropped.render()}: {dropped.finding.claim}")
            lines.append("")
            lines.append("  ```")
            lines.extend(f"  {line}" for line in dropped.finding.quoted_lines.splitlines()[:12])
            lines.append("  ```")
        lines.append("")
    if report.missing:
        lines.append("## Not available to this worker")
        lines.extend(f"- {item}" for item in report.missing)
    return "\n".join(lines).rstrip() + "\n"


def _failed(task: PlanTask, exc: DynaflowsError) -> WorkerResult:
    """One failure shape, so eleven siblings keep going and the twelfth is
    visible rather than merely absent."""
    return WorkerResult(
        task_id=task.task_id,
        capability=task.capability,
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
    """The final report. ADR-020: two documents, one computed and one written.

    The computed half is written first and comes entirely from state, so a
    reader always learns what the run actually did -- even when the model call
    below fails, returns nothing, or is never made. Asking a model to describe
    the limits of its own output produces an understatement that reads exactly
    like an accurate summary, which is why that half is not its to write.
    """
    ledger = cost_delta()
    results = state.get("results") or []
    evaluation = state.get("evaluation")

    try:
        store = store_from(config)
    except DynaflowsError as exc:
        return {"halted": f"cannot write the synthesis: {exc}", "cost": ledger}

    findings, accounting = synth.collect(results, store.read_data)
    computed = synth.render_accounting(accounting, evaluation, has_summary=bool(findings))

    # ADR-020: nothing verified means nothing to synthesise. Paying a frontier
    # model to write prose about an empty list produces confident prose about
    # nothing, which is run w1's failure relocated to the last step.
    if not findings:
        artifact = store.write(
            state.get("run_id", "unknown"),
            "synthesis",
            f"# No verified findings\n\n{computed}\n",
        )
        return {"synthesis": artifact, "cost": ledger}

    try:
        gateway = gateway_from(config)
        result = await gateway.call(
            CallRequest(
                tier=Tier.FRONTIER,
                system=SYNTHESIZER_SYSTEM,
                prompt=(
                    f"OBJECTIVE\n{state.get('enhanced_prompt') or state.get('raw_prompt', '')}"
                    f"\n\nVERIFIED FINDINGS\n{synth.render_findings(findings)}"
                ),
                schema=SynthesisDraft,
                max_tokens=3072,
                label="synthesize",
                metadata=(
                    ("run_id", state.get("run_id", "")),
                    ("node", "synthesize"),
                    ("findings", str(len(findings))),
                ),
            )
        )
    except DynaflowsError as exc:
        # The computed half still ships. The findings are all in the run store
        # and the reader can see every one of them; losing the summary is a
        # smaller loss than losing the report.
        artifact = store.write(
            state.get("run_id", "unknown"),
            "synthesis",
            f"# Synthesis unavailable\n\n{computed}\n\n"
            f"The summary could not be written: {exc}\n\n"
            f"{_render_raw_findings(findings)}\n",
        )
        return {"synthesis": artifact, "degraded": True, "cost": ledger}

    ledger = _merge_delta(ledger, result)
    draft: SynthesisDraft = result.payload
    known = {item.id for item in findings}
    kept = [s for s in draft.sections if s.finding_ids and set(s.finding_ids) <= known]
    invented = len(draft.sections) - len(kept)

    artifact = store.write(
        state.get("run_id", "unknown"),
        "synthesis",
        _render_synthesis(draft.headline, kept, invented, findings, computed),
    )
    return {
        "synthesis": artifact,
        "degraded": bool(invented) or (evaluation is not None and not evaluation.passed),
        "cost": ledger,
    }


def _render_raw_findings(findings: list[synth.Corroborated]) -> str:
    return "## Verified findings\n\n" + synth.render_findings(findings)


def _render_synthesis(
    headline: str,
    sections: list[Any],
    invented: int,
    findings: list[synth.Corroborated],
    computed: str,
) -> str:
    """Computed first, written second. The order is the decision.

    A reader who stops after the first screen has seen what the run did rather
    than a confident summary of part of it.
    """
    lines = [f"# {headline}", "", computed, ""]
    if invented:
        lines.append(
            f"> {invented} section(s) were discarded for citing findings that do not exist. "
            "The synthesizer is a summariser and may not introduce claims (ADR-020)."
        )
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    for section in sections:
        lines.append(f"### {section.heading}")
        lines.append(section.body)
        lines.append(f"_Findings: {', '.join(section.finding_ids)}_")
        lines.append("")
    lines.append(_render_raw_findings(findings))
    return "\n".join(lines).rstrip() + "\n"
