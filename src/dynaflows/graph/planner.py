"""Planning: draft, validate, and hash. ADR-001, ADR-014.

Kept apart from the node so every rule here is testable without a graph, a
checkpointer or an event loop.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dynaflows.contracts.state import MAX_FANOUT, Plan, PlanTask
from dynaflows.graph.capabilities import CATALOGUE_VERSION
from dynaflows.graph.prompts import PlanDraft
from dynaflows.store.catalogue import SourceCatalogue, unknown_paths
from dynaflows.store.sources import read_sources

# Per-task token estimate, used for the G2 summary. PLACEHOLDER (playbook 4.5):
# the pack budget plus room for the objective and the worker's answer. Step 1.8
# replaces it with LangSmith's real per-call counts.
TOKENS_PER_TASK_ESTIMATE = 3_500

_ID_CLEAN = re.compile(r"[^a-z0-9_-]+")


def normalise_task_id(raw: str, index: int) -> str:
    """Make an id usable without rejecting a plan over punctuation.

    A model that returns "Task 1: Review auth" has not made a planning error;
    it has made a formatting one, and re-planning for that would spend a
    frontier call to fix a hyphen.
    """
    cleaned = _ID_CLEAN.sub("-", raw.strip().lower()).strip("-")
    return cleaned or f"task-{index + 1}"


def draft_to_plan(draft: PlanDraft) -> Plan:
    """Strict contract from a loose draft. Raises ValidationError on a bad plan.

    The caller turns that into ONE re-plan attempt, then a halt -- never a
    silent trim (ADR-014).
    """
    return Plan(
        rationale=draft.rationale,
        tasks=[
            PlanTask(
                task_id=normalise_task_id(task.task_id, index),
                capability=task.capability.strip(),
                objective=task.objective.strip(),
                inputs=list(task.inputs),
                playbook_anchors=list(task.playbook_anchors),
            )
            for index, task in enumerate(draft.tasks)
        ],
    )


def violation_of(draft: PlanDraft, catalogue: SourceCatalogue | None = None) -> str | None:
    """Why this draft cannot become a plan, phrased for the model to act on.

    Returned to the planner verbatim on the re-plan. A message like
    "ValidationError" teaches it nothing; naming the bound and the offending
    value gives it something to change.

    `catalogue` is optional so the planning rules stay testable without a
    filesystem, and its absence disables only the rules that need it -- a
    check that silently passes when its input is missing is worse than one
    that is not there (ADR-018).
    """
    if not draft.tasks:
        return "The plan had no tasks. Emit at least one."
    if catalogue is not None and catalogue.total:
        # ADR-018, the half that matters: the catalogue makes good plans
        # likely, this makes a bad one impossible to execute silently.
        for task in draft.tasks:
            if not task.inputs:
                return (
                    f"Task '{task.task_id}' named no inputs. Every task must name at least "
                    "one file or directory from the source catalogue -- a worker cannot "
                    "search, and a task with nothing to examine produces guesswork."
                )
            missing = unknown_paths(list(task.inputs), catalogue)
            if missing:
                return (
                    f"Task '{task.task_id}' named {', '.join(repr(m) for m in missing)}, "
                    "which is not in the source catalogue. Copy paths exactly from the "
                    "catalogue; do not infer or abbreviate them."
                )
    if len(draft.tasks) > MAX_FANOUT:
        return (
            f"The plan had {len(draft.tasks)} tasks; the hard limit is {MAX_FANOUT}. "
            "Merge the overlapping ones rather than dropping any work."
        )
    try:
        draft_to_plan(draft)
    except ValidationError as exc:
        return f"The plan was rejected: {exc.errors()[0].get('msg', exc)}"
    return None


def plan_hash(plan: Plan, models_version: str) -> str:
    """Identity of a plan, including what it was planned AGAINST.

    The catalogue and model-registry versions are in the hash because a plan
    means something different under a different catalogue or a different model
    set -- comparing two runs across either would be comparing nothing
    (ADR-001, ADR-006).
    """
    material = {
        "catalogue_version": CATALOGUE_VERSION,
        "models_version": models_version,
        "tasks": [
            {
                "task_id": t.task_id,
                "capability": t.capability,
                "objective": t.objective,
                "inputs": t.inputs,
                "playbook_anchors": t.playbook_anchors,
            }
            for t in plan.tasks
        ],
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def measure_context(plan: Plan, repository: Any, root: Path, budget: int) -> Plan:
    """Fill in what each task's worker will actually be given, by building it.

    Not an estimate. The old `estimate_tokens` multiplied the task count by a
    constant I invented, so gate G2 -- the gate that authorises the entire
    fan-out spend -- showed a number with no relationship to what the workers
    would receive. Run `s1` displayed "~10,500 tokens estimated" for three
    tasks whose inputs came to 40,000, and the workers were handed a fraction
    of their files.

    This reads the same files the dispatcher will read, through the same
    resolver, so the number at the gate is the number that happens. It costs
    one extra read of each named file per run, at the point where a human is
    about to approve spending real money on them.
    """
    for task in plan.tasks:
        chunks, _ = read_sources(list(task.inputs), root)
        anchors = repository.by_anchor(list(task.playbook_anchors))
        task.context_tokens = sum(c.tokens for c in (*anchors, *chunks))
    plan.context_budget = budget
    plan.estimated_tokens = sum(min(t.context_tokens, budget) for t in plan.tasks)
    return plan


def over_budget(plan: Plan) -> list[str]:
    """Tasks whose inputs do not fit, named so the human can act at G2.

    A task over budget is not an error -- `pack_sections` will fill what fits
    and the worker will be told what it did not get. It IS something the person
    approving the spend has to see, because the remedy is theirs: split the
    task, or name fewer files.
    """
    if not plan.context_budget:
        return []
    return [t.task_id for t in plan.tasks if t.context_tokens > plan.context_budget]


def estimate_tokens(plan: Plan) -> int:
    """Superseded by `measure_context`. Kept for the zero-input case only."""
    return len(plan.tasks) * TOKENS_PER_TASK_ESTIMATE
