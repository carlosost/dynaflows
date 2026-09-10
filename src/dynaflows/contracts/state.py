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
from dataclasses import replace
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
    cost_usd: float = 0.0
    error: ErrorEnvelope | None = None

    @property
    def produced_something(self) -> bool:
        return self.status != "failed" and bool(self.summary or self.artifact)


class EvaluationReport(BaseModel):
    """ADR-004: deterministic, no LLM. The judge arrives in Phase 2."""

    task_count: int
    ok_count: int
    failed_count: int
    empty_count: int
    passed: bool
    reasons: list[str] = Field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_count / self.task_count if self.task_count else 0.0


def merge_cost(current: CostLedger, update: CostLedger) -> CostLedger:
    """Sum two ledgers field by field.

    Concurrent branches each contribute their own spend. Anything other than a
    sum here loses money that was actually spent, which would make the budget
    ceiling and the G2 estimate both wrong.
    """
    return CostLedger(
        usd_spent=current.usd_spent + update.usd_spent,
        usd_avoided=current.usd_avoided + update.usd_avoided,
        tokens_in=current.tokens_in + update.tokens_in,
        tokens_out=current.tokens_out + update.tokens_out,
        calls_made=current.calls_made + update.calls_made,
        calls_cached=current.calls_cached + update.calls_cached,
        schema_failures=current.schema_failures + update.schema_failures,
        calls_skipped=current.calls_skipped + update.calls_skipped,
        fallbacks=current.fallbacks + update.fallbacks,
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
    plan_gate: GateOutcome | None

    # --- concurrent keys: reducers are mandatory (see FAN_OUT_KEYS) --------
    results: Annotated[list[WorkerResult], operator.add]
    cost: Annotated[CostLedger, merge_cost]

    evaluation: EvaluationReport | None
    degraded: bool
    synthesis: ArtifactRef | None
    halted: str | None


def initial_state(run_id: str, thread_id: str, raw_prompt: str) -> WorkflowState:
    """The only place an initial state is built.

    Reduced keys are seeded explicitly: a reducer is called with whatever is
    already there, and 'already there' must not be None.
    """
    return WorkflowState(
        run_id=run_id,
        thread_id=thread_id,
        raw_prompt=raw_prompt,
        enhanced_prompt=None,
        enhancer_assumptions=[],
        prompt_gate=None,
        plan=None,
        plan_hash=None,
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
