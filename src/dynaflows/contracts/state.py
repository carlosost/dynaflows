"""The graph state and the plan contracts. PMA §2.1-2.3, ADR-008, ADR-014.

Written before the nodes that fill them (playbook §1.4). Two rules here are
load-bearing and easy to lose:

  * Any key written by more than one concurrent branch MUST carry a reducer.
    Without one LangGraph raises InvalidUpdateError -- but only at runtime,
    only under fan-out, and therefore only in the situation that is hardest to
    reproduce. `FAN_OUT_KEYS` plus an architecture test pins it down instead.

  * A node that fans out returns a DELTA, never a total. Every branch starts
    from the same input state, so a branch returning a running total would
    overwrite its siblings rather than combine with them.
"""

from __future__ import annotations

import operator
from dataclasses import fields, replace
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from dynaflows.contracts.calls import CostLedger
from dynaflows.contracts.errors import ErrorEnvelope
from dynaflows.contracts.tiers import Tier

# ADR-014. A PLACEHOLDER, not a measurement (playbook §4.5): wide enough to
# exercise partial failure, real 429s and checkpoint pressure; narrow enough
# that a debug cycle finishes while you are still watching it. Step 1.8
# replaces it with numbers.
MAX_FANOUT = 12

# Keys that concurrent Send branches write. Every one must be Annotated with a
# reducer; tests/architecture/test_state_reducers.py binds this list to the
# annotations so the two cannot drift apart.
FAN_OUT_KEYS = frozenset({"results", "cost"})


class GateDecision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class GateOutcome(BaseModel):
    decision: GateDecision
    # What the human replaced the model's output with, when they edited it.
    # Kept verbatim: the whole point of the gate is that a human's text beats
    # a model's, so paraphrasing it here would defeat the feature.
    replacement: str | None = None
    note: str | None = None

    @property
    def proceeds(self) -> bool:
        return self.decision is not GateDecision.REJECT


class ArtifactRef(BaseModel):
    """ADR-008: state carries references, payloads live on disk.

    LangGraph checkpoints the whole state at every superstep. Sixteen workers
    putting raw model output in state re-serialises all of it, repeatedly,
    through SQLite's single writer.
    """

    sha: str
    path: str
    kind: str = "markdown"
    tokens: int = 0
    preview: str = Field(default="", max_length=280)


class PlanTask(BaseModel):
    task_id: str
    capability: str
    objective: str
    inputs: list[str] = Field(default_factory=list)
    # ADR-009: the planner routes retrieval for every worker in one call.
    playbook_anchors: list[str] = Field(default_factory=list)
    tier_override: Tier | None = None
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("capability")
    @classmethod
    def _capability_must_be_registered(cls, value: str) -> str:
        """ADR-001: the planner selects from a catalogue, it does not invent.

        Validated here rather than trusted, because a task naming a capability
        no worker implements would dispatch a branch that can only fail -- and
        it would fail N supersteps later, far from the plan that caused it.
        """
        from dynaflows.graph.capabilities import CAPABILITY_IDS

        if value not in CAPABILITY_IDS:
            raise ValueError(f"unknown capability {value!r}; registered: {sorted(CAPABILITY_IDS)}")
        return value

    @field_validator("depends_on")
    @classmethod
    def _phase_1_is_a_flat_map(cls, value: list[str]) -> list[str]:
        """OQ-02 is open and Phase 1 is a pure map.

        Enforced rather than documented: a planner that quietly emits a
        dependency would otherwise have it silently ignored, producing a plan
        that runs in the wrong order with no error anywhere.
        """
        if value:
            raise ValueError(
                "intra-plan dependencies are not supported in Phase 1 (OQ-02); "
                "express the task as independent work or split the run"
            )
        return value


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(min_length=1, max_length=MAX_FANOUT)
    rationale: str = ""
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @field_validator("tasks")
    @classmethod
    def _task_ids_are_unique(cls, value: list[PlanTask]) -> list[PlanTask]:
        ids = [t.task_id for t in value]
        if len(set(ids)) != len(ids):
            raise ValueError("task_id must be unique within a plan")
        return value


class WorkerResult(BaseModel):
    task_id: str
    status: Literal["ok", "degraded", "failed"]
    summary: str = Field(default="", max_length=1200)
    artifact: ArtifactRef | None = None
    model_id: str = ""
    tier: Tier | None = None
    fallback_depth: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    # Optional for the same reason as CallResult.cost_usd: a call nobody
    # could price and a call that cost nothing are different facts.
    cost_usd: float | None = None
    # ADR-019, and AP-20 again: claims made and claims that survived checking
    # are two facts. Equal numbers mean a careful worker; a large gap means one
    # that is inventing, and one number cannot say which.
    findings_reported: int = 0
    findings_grounded: int = 0
    error: ErrorEnvelope | None = None

    @property
    def produced_something(self) -> bool:
        """Whether this worker actually delivered anything.

        An ArtifactRef is not evidence of output: run `w2` wrote a ZERO-BYTE
        findings file, got a perfectly valid reference to it, and counted as
        having produced something -- so `empty_count` stayed 0 and the task
        reported `ok`. A reference to nothing is nothing.
        """
        if self.status == "failed":
            return False
        has_artifact = self.artifact is not None and self.artifact.tokens > 0
        return bool(self.summary.strip()) or has_artifact

    @property
    def findings_dropped(self) -> int:
        return self.findings_reported - self.findings_grounded


class EvaluationReport(BaseModel):
    """ADR-004: deterministic, no LLM. The judge arrives in Phase 2.

    Every status a worker can return has a counter here, and the counters must
    account for every task. That is not tidiness. `degraded` existed for a
    whole commit with no counter, so a run of five degraded results reported
    "0 ok, 0 failed" and `passed=True` -- and one of those five had fabricated
    an audit of four files that do not exist. A fact with no counter becomes
    silence, and silence reads as good news (AP-20, third occurrence).
    """

    task_count: int
    ok_count: int
    failed_count: int
    empty_count: int
    degraded_count: int = 0
    passed: bool
    reasons: list[str] = Field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_count / self.task_count if self.task_count else 0.0

    @property
    def accounted_for(self) -> int:
        """Results this report can explain.

        `empty_count` is deliberately NOT in this sum. ok/degraded/failed are
        statuses and they partition the results; "empty" is an orthogonal
        observation -- a degraded result that produced nothing is both, and
        adding all four double-counts it. The first version of this property
        did add all four, and a test written at the same time caught it, which
        is the only reason it is not a fourth wrong number.
        """
        return self.ok_count + self.degraded_count + self.failed_count

    @property
    def unaccounted(self) -> int:
        return self.task_count - self.accounted_for

    def render(self) -> str:
        """One line, naming every non-zero fact. A status line that can only
        say 'ok' and 'failed' cannot describe a degraded run at all."""
        parts = [f"{self.task_count} task(s)", f"{self.ok_count} ok"]
        if self.degraded_count:
            parts.append(f"{self.degraded_count} degraded")
        if self.failed_count:
            parts.append(f"{self.failed_count} failed")
        if self.empty_count:
            parts.append(f"{self.empty_count} empty")
        if self.unaccounted:
            parts.append(f"{self.unaccounted} unaccounted for")
        return ", ".join(parts) + f", passed={self.passed}"


def merge_cost(current: CostLedger, update: CostLedger) -> CostLedger:
    """Sum two ledgers field by field.

    Concurrent branches each contribute their own spend. Anything other than a
    sum here loses money that was actually spent, which would make the budget
    ceiling and the G2 estimate both wrong.

    Written as a loop over the dataclass fields rather than ten hand-written
    additions. The hand-written version omitted `calls_unpriced` for two
    commits -- added to the ledger, never added here -- so every unpriced call
    was discarded at the first merge and the "LOWER BOUND" warning built to
    surface them could not fire in a real run. Enumerating the fields by hand
    is a list that must be updated in two places, and it was not.
    """
    return CostLedger(
        **{
            field.name: getattr(current, field.name) + getattr(update, field.name)
            for field in fields(CostLedger)
        }
    )


class WorkflowState(TypedDict, total=False):
    """PMA §2.1. `total=False` because the graph fills it in stages."""

    run_id: str
    thread_id: str

    raw_prompt: str
    enhanced_prompt: str | None
    # Surfaced at gate G1: a human approving a rewrite needs to see what was
    # assumed on their behalf, not just the result.
    enhancer_assumptions: list[str]
    prompt_gate: GateOutcome | None

    plan: Plan | None
    plan_hash: str | None
    # Why planning failed, when it did. Surfaced at G2 instead of a trimmed
    # plan that looks fine (ADR-014).
    plan_rejected_reason: str | None
    plan_gate: GateOutcome | None

    # --- concurrent keys: reducers are mandatory (see FAN_OUT_KEYS) --------
    results: Annotated[list[WorkerResult], operator.add]
    cost: Annotated[CostLedger, merge_cost]

    # The tree this run was planned against and whose files its workers read
    # (ADR-017, ADR-018). Checkpointed because `resume` builds its config
    # BEFORE it can read state, and a resume that guesses a different root
    # analyses different files without saying so -- a wrong answer, not an
    # inconvenience.
    source_root: str

    # Present ONLY inside a Send branch: dispatch_workers puts one task on
    # each copy of the state. Declared here so the worker node reads a typed
    # field rather than an untyped bag, and absent everywhere else on purpose.
    task: PlanTask | None

    evaluation: EvaluationReport | None
    degraded: bool
    synthesis: ArtifactRef | None
    halted: str | None


def initial_state(
    run_id: str, thread_id: str, raw_prompt: str, source_root: str = ""
) -> WorkflowState:
    """The only place an initial state is built.

    Reduced keys are seeded explicitly: a reducer is called with whatever is
    already there, and 'already there' must not be None.
    """
    return WorkflowState(
        run_id=run_id,
        thread_id=thread_id,
        raw_prompt=raw_prompt,
        source_root=source_root,
        enhanced_prompt=None,
        enhancer_assumptions=[],
        prompt_gate=None,
        plan=None,
        plan_hash=None,
        plan_rejected_reason=None,
        plan_gate=None,
        results=[],
        cost=CostLedger(),
        evaluation=None,
        degraded=False,
        synthesis=None,
        halted=None,
    )


def cost_delta(**kwargs: float | int) -> CostLedger:
    """A single node's own spend, never a running total (see module docstring)."""
    return replace(CostLedger(), **kwargs)  # type: ignore[arg-type]
