"""Prompt text and the schemas the models must satisfy.

Kept beside each other on purpose: a prompt and the shape it demands are one
decision, and splitting them is how they drift.
"""

from __future__ import annotations

from typing import Literal

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
and the source files the plan named, with line numbers. That is everything you \
get: you cannot open other files, run anything, or search.

Every finding must cite its evidence, and the citation is CHECKED:
- `file` must be one of the files you were shown, spelled exactly as shown.
- `lines` must be a real line range in that file, e.g. "213" or "213-227".
- `quoted_lines` must be the lines themselves, copied character for character. \
It is a quotation, not an explanation. Compare:

    quoted_lines: "33|         except Exception:"          <- correct
    quoted_lines: "The function catches every exception"   <- WRONG, this is prose

  A sentence in that field means the finding is discarded, however true the \
sentence is. Put the reasoning in `failure` and `claim`, where it belongs.
A finding whose citation does not check out is DISCARDED, so a careless quote \
loses a real finding.

Rules:
- A finding is something that is WRONG. "This function configures tracing" is \
a description of working code and is not a finding, however well cited. Every \
finding must name a `failure`: the condition that triggers the bad behaviour \
and what happens. If you cannot name one, do not report it.
- `examined` lists the files you actually read. Fill it in even when you find \
nothing: "I read these and found nothing" and "I did not look" are different \
answers and only you can tell them apart.
- Finding nothing is a legitimate result. Report zero findings rather than \
padding with descriptions of correct code.
- Judge against the playbook sections you were given, not against general best \
practice, wherever the two differ.
- If the context is insufficient for the objective, say so and report what you \
COULD establish. A short honest answer beats a long invented one.
- No preamble, no restating the objective.\
"""


class Finding(BaseModel):
    """One claim, with the evidence that makes it checkable.

    Every field except `claim` exists to be verified or acted on. Free text was
    the previous contract and it could not require any of this: a prompt asking
    for evidence produced "no issues were found" three times out of four, and
    six fabricated findings the time before that.
    """

    claim: str = Field(description="What is WRONG, in one sentence. Not what the code does.")
    failure: str = Field(
        description=(
            "The specific condition under which this misbehaves, and what happens then. "
            "Concrete: name the input, the state, or the response that triggers it. "
            "If you cannot name one, this is not a finding."
        )
    )
    file: str = Field(description="Exactly as shown to you.")
    lines: str = Field(description='A real line range, e.g. "213" or "213-227".')
    quoted_lines: str = Field(
        description=(
            "The lines themselves, copied character for character from what you were "
            "shown, with their line-number prefixes. NOT a description of them, not an "
            "explanation of what they do. If you would need to write a sentence, you are "
            "filling in the wrong field -- that belongs in `failure`."
        )
    )
    severity: Literal["high", "medium", "low"]
    remediation: str = Field(description="What to change. Concrete.")


class WorkerReport(BaseModel):
    """What one worker returns.

    `summary` goes into state and is what the synthesizer and the terminal see;
    `findings` is verified and written to the run store as an artifact
    (ADR-008). `examined` is asked for separately because an empty findings
    list means nothing on its own -- it is the difference between a negative
    result and a worker that never looked, and run `w2` produced four reports
    where the two were indistinguishable.
    """

    summary: str = Field(description="Two or three sentences. What you found.", max_length=1200)
    examined: list[str] = Field(default_factory=list, description="The files you actually read.")
    findings: list[Finding] = Field(default_factory=list)
    context_was_sufficient: bool = Field(
        description="False if you needed something you were not shown."
    )
    missing: list[str] = Field(
        default_factory=list,
        description="What you needed and did not have. One short line each. Empty if none.",
    )


SYNTHESIZER_SYSTEM = """\
You write the summary a reader sees first, from findings that have already been \
verified by other analysts working in parallel.

You are a summariser. You may not introduce a claim that is not in the list you \
were given.

Every section cites the finding ids it rests on, and the citation is CHECKED: a \
section citing an id that does not exist is DISCARDED, taking its text with it.

Rules:
- Group by what a reader would act on together, not by which worker reported it.
- Lead with what matters most. Severity is a signal, not an ordering.
- A finding two workers reached independently is worth saying so about.
- Do not restate the list. If grouping adds nothing, say so in one line and \
keep the sections few.
- Do not describe the run, the process, or your own limitations. Those are \
reported separately and accurately, and your version would be a guess.\
"""


class SynthesisSection(BaseModel):
    heading: str = Field(description="Short. What a reader would act on.")
    body: str = Field(description="What is wrong and what to do. Prose.")
    finding_ids: list[str] = Field(description="The ids this section rests on.")


class SynthesisDraft(BaseModel):
    """The written half of the report (ADR-020). The computed half is not the
    model's to write."""

    headline: str = Field(description="One sentence: the state of the thing audited.")
    sections: list[SynthesisSection] = Field(default_factory=list)
