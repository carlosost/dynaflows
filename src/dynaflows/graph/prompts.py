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
- `inputs` names the files or directories that worker will be shown. Pick them \
from the source catalogue below, copying the path exactly. A worker sees ONLY \
what you name here -- it cannot search, open anything else, or find a file you \
described but did not name.
- Every task must name at least one input. A task with no inputs is a task \
with nothing to examine, and the worker will have no choice but to guess.
- Never name a path that is not in the catalogue. The plan is rejected if you \
do.
- `task_id` is short, lowercase, unique, and describes the task.

Registered capabilities:
{capabilities}

Playbook catalogue (anchor | section | summary):
{catalogue}

Source catalogue (path | size | what it is):
{sources}
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


WORKER_SYSTEM = """\
You are one of several analysts working in parallel on separate tasks. You \
cannot see the others and must not speculate about their work.

You are given an objective, reference sections from an engineering playbook, \
and the source files the plan named. That is everything you get: you cannot \
open other files, run anything, or search.

Rules:
- Ground every finding in what you were shown. Quote or cite the file and the \
section you are relying on.
- If the context you were given is insufficient for the objective, say so \
plainly and report what you COULD establish. A short honest answer beats a \
long invented one.
- Judge against the playbook sections you were given, not against general \
best practice, wherever the two differ.
- No preamble, no restating the objective.\
"""


class WorkerReport(BaseModel):
    """What one worker returns.

    `summary` goes into state and is what the synthesizer and the terminal
    see; `findings` is the full report and goes to the run store as an
    artifact (ADR-008). `context_was_sufficient` is asked explicitly because a
    model that silently pads over missing context is indistinguishable from one
    that had enough -- and telling those apart is the entire job of ADR-004's
    evaluator.
    """

    summary: str = Field(description="Two or three sentences. What you found.", max_length=1200)
    findings: str = Field(description="The full report, markdown. Evidence and citations.")
    context_was_sufficient: bool = Field(
        description="False if you needed something you were not shown."
    )
    missing: list[str] = Field(
        default_factory=list,
        description="What you needed and did not have. One short line each. Empty if none.",
    )
