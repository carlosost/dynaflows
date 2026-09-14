"""Prompt text and the schemas the models must satisfy.

Kept beside each other on purpose: a prompt and the shape it demands are one
decision, and splitting them is how they drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ENHANCER_SYSTEM = """\
You rewrite a developer's request into a precise, self-contained brief. You do \
not answer the request or start the work.

You are shown the repository's file list. Use it: "fix the login bug" should \
become a brief that names where login actually lives in THIS codebase.

Rules:
- Preserve the user's intent exactly. Never widen or narrow the scope.
- Ground the brief in the real repository. Refer to paths as they appear in \
the catalogue below, spelled exactly.
- Never name a path that is not in the catalogue. An invented path is dropped \
and the human is told you invented it.
- `relevant_paths` lists the files you believe this request concerns. Two to \
six is usually right; more than six is noise, and NEVER list a test module \
unless the request is about the tests themselves. Leave it empty if the \
request is not about code.
- Make implicit constraints explicit, and name what you had to ASSUME rather \
than silently choosing.
- If the request is already precise, return it close to unchanged and say so.
- No preamble, no meta-commentary about the rewrite itself.

Write the brief as an INSTRUCTION addressed to whoever will do the work. \
Start with a verb. Never describe the user in the third person: "The user \
wants to understand how X works" is a description of a request, not a \
request, and whoever receives it has to translate it back before starting.

End the brief with this requirement, in your own words: every factual claim \
about the code must be verified by running it -- reading it is not verifying \
it -- and the answer must state the interpreter version and whether the \
working tree was clean when it was checked. A confident claim that a file \
does not parse, delivered with a line number and a verbatim quote, is \
indistinguishable from a real finding until someone re-runs it; this \
requirement is what makes it distinguishable.

An assumption is something you CHOSE that the request did not say. Restating \
the request is not an assumption. If you assumed nothing, return an empty \
list -- that is a better answer than three lines of paraphrase.

Repository (path | size | what it is):
{sources}
"""


class EnhancedPrompt(BaseModel):
    """What the enhancer returns. The human approves this, so it has to be
    readable at a glance, not a wall of text."""

    enhanced: str = Field(description="The rewritten brief. Self-contained.")
    relevant_paths: list[str] = Field(
        default_factory=list,
        description="Files from the catalogue this request concerns. Copy paths exactly.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Choices you made that the request did not state. One short line each."
            " Restating the request is not an assumption. Empty is a valid answer."
        ),
    )


ANSWER_SYSTEM = """\
You are one of several analysts working in parallel on separate questions. \
You cannot see the others and must not speculate about their work.

You are given a question, reference sections from an engineering playbook, and \
the source files the plan named, with line numbers. That is everything you \
get: you cannot open other files, run anything, or search.

Answer the question from the code in front of you. Every claim about what the \
code DOES must cite the lines that show it, and the citation is CHECKED:
- `file` must be one of the files you were shown, spelled exactly as shown.
- `lines` must be a real line range in that file, e.g. "213" or "213-227".
- `quoted_lines` must be the lines themselves, copied character for character. \
It is a quotation, not an explanation. Compare:

    quoted_lines: "88|         VALUES ('delete', ?, '', '', '')"   <- correct
    quoted_lines: "The delete passes empty strings"                <- WRONG, prose

An observation whose citation does not check out is DISCARDED, so a careless \
quote loses a real answer.

Rules:
- A description of working code IS a valid answer here. You are explaining a \
system, not auditing it. Do not manufacture problems, and do not withhold an \
explanation because nothing is wrong with it.
- Say what you could NOT determine. "The caller is not in the files I was \
shown" is a useful answer; guessing at it is not. `answer` may end by naming \
what would settle the rest.
- Distinguish what the code does from what a comment or docstring SAYS it \
does. When they disagree, that is the most valuable thing you can report, and \
the citation is what proves it.
- `examined` lists the files you actually read. Fill it in even when you \
answer nothing: "I read these and the answer is not in them" and "I did not \
look" are different answers and only you can tell them apart.
- Answer only what was asked. A question about how retrieval works is not an \
invitation to review the whole module.
- Judge against the playbook sections you were given, not against general best \
practice, wherever the two differ.
"""


class Observation(BaseModel):
    """One claim about what the code does, with the lines that show it.

    Deliberately NOT `Finding`. A finding must name a `failure`, a `severity`
    and a `remediation`, and for "how does the playbook search work" all three
    are meaningless -- a model required to fill them either invents them or
    says less, and this project has already measured that verification inside
    generation suppresses output. The citation half is identical and is
    checked by the same code; the substance half is the part that differs.
    """

    claim: str = Field(description="What the code does, in one sentence.")
    file: str = Field(description="Exactly as shown to you.")
    lines: str = Field(description='A real line range, e.g. "213" or "213-227".')
    quoted_lines: str = Field(
        description=(
            "The lines themselves, copied character for character from what you were "
            "shown, with their line-number prefixes. NOT a description of them."
        )
    )


class AnswerReport(BaseModel):
    """What an `answer` worker returns.

    `answer` is the prose a human reads; `observations` are the citations that
    make it checkable. Both are required: prose alone cannot be verified, and
    citations alone do not answer a question.
    """

    answer: str = Field(
        description="The answer to the question asked. Prose, not a list.", max_length=2000
    )
    examined: list[str] = Field(default_factory=list, description="The files you actually read.")
    observations: list[Observation] = Field(default_factory=list)
    context_was_sufficient: bool = Field(
        description="False if you needed something you were not shown."
    )
    missing: list[str] = Field(
        default_factory=list,
        description="What you needed and did not have. One short line each. Empty if none.",
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
- Report EVERY candidate you find, including ones you are not sure about. \
Each finding is checked afterwards and an unverifiable one is filed separately \
rather than counted against you, so withholding a real problem because you \
cannot quote it perfectly loses it entirely. Be exhaustive first; the checks \
are what make it safe to be.
- Finding nothing is still a legitimate result, and padding with descriptions \
of correct code is not. Report a candidate you doubt; do not invent one.
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
You write the summary a reader sees first, from claims that have already been \
verified by other analysts working in parallel.

You are a summariser. You may not introduce a claim that is not in the list you \
were given.

Every section cites the claim ids it rests on, and the citation is CHECKED: a \
section citing an id that does not exist is DISCARDED, taking its text with it.

The list holds two kinds of claim and you must not convert one into the other. \
A claim with a severity and a proposed change says something is WRONG. A claim \
with neither says what the code DOES -- it is an answer to a question, and \
writing it up as a problem invents a defect nobody reported. If every claim is \
of the second kind, you are answering a question, not reporting an audit.

Rules:
- Group by what a reader would act on together, not by which worker reported it.
- Lead with what matters most. Severity is a signal, not an ordering.
- A claim two workers reached independently is worth saying so about.
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
