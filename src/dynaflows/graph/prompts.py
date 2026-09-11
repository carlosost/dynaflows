"""Prompt text and the schemas the models must satisfy.

Kept beside each other on purpose: a prompt and the shape it demands are one
decision, and splitting them is how they drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ENHANCER_SYSTEM = """\
You rewrite a developer's request into a precise, self-contained brief for an \
analysis workflow. You do not answer the request.

Rules:
- Preserve the user's intent exactly. Never widen or narrow the scope.
- Make implicit constraints explicit.
- Name what you had to assume, rather than silently choosing.
- If the request is already precise, return it close to unchanged and say so.
- No preamble, no meta-commentary about the rewrite itself.\
"""


class EnhancedPrompt(BaseModel):
    """What the enhancer returns. The human approves this, so it has to be
    readable at a glance, not a wall of text."""

    enhanced: str = Field(description="The rewritten brief. Self-contained.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything you had to assume. One short line each. Empty if none.",
    )


PLANNER_SYSTEM = """\
You decompose a brief into independent analysis tasks for parallel workers.

Hard rules:
- Emit at most {max_fanout} tasks. Fewer is better than more: a task that \
overlaps another wastes a worker and muddies the synthesis.
- Every task must stand alone. Workers run in parallel and cannot see each \
other's output, so a task may never depend on another task's result.
- `capability` must be one of the registered capabilities, exactly as written.
- `playbook_anchors` selects the reference sections that worker will be shown. \
Pick from the catalogue by anchor. Two or three per task; omit rather than \
pad. These are the ONLY sections that worker will see.
- `inputs` names files, paths or identifiers the worker should examine. Leave \
empty if the objective is self-contained.
- `task_id` is short, lowercase, unique, and describes the task.

Registered capabilities:
{capabilities}

Playbook catalogue (anchor | section | summary):
{catalogue}
"""


class PlannedTask(BaseModel):
    """A task as the PLANNER states it -- deliberately looser than PlanTask.

    Validation into the strict contract happens after, so a bad plan is
    re-planned rather than crashing the node (ADR-014).
    """

    task_id: str
    capability: str
    objective: str
    inputs: list[str] = Field(default_factory=list)
    playbook_anchors: list[str] = Field(default_factory=list)


class PlanDraft(BaseModel):
    rationale: str = Field(description="One or two sentences: why this split.")
    tasks: list[PlannedTask]
