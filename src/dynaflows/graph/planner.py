"""Planning: draft, validate, and hash. ADR-001, ADR-014.

Kept apart from the node so every rule here is testable without a graph, a
checkpointer or an event loop.
"""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import ValidationError

from dynaflows.contracts.state import MAX_FANOUT, Plan, PlanTask
from dynaflows.graph.capabilities import CATALOGUE_VERSION
from dynaflows.graph.prompts import PlanDraft

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


def violation_of(draft: PlanDraft) -> str | None:
    """Why this draft cannot become a plan, phrased for the model to act on.

    Returned to the planner verbatim on the re-plan. A message like
    "ValidationError" teaches it nothing; naming the bound and the offending
    value gives it something to change.
    """
    if not draft.tasks:
        return "The plan had no tasks. Emit at least one."
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


def estimate_tokens(plan: Plan) -> int:
    return len(plan.tasks) * TOKENS_PER_TASK_ESTIMATE
