"""Prompt text and the schemas the models must satisfy.

Kept beside each other on purpose: a prompt and the shape it demands are one
decision, and splitting them is how they drift.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

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

The brief is written in the IMPERATIVE MOOD, as a task someone is handed.

- "Explain how X works and whether it scales to N documents." -- correct.
- "The user wants to understand how X works." -- WRONG. That describes a \
request instead of making one, and the reader has to translate it back.
- "I'll examine X. Let me start by reading Y." -- WRONG, and worse. You are \
not the one doing this work. Announcing what you would do produces no brief \
at all.

Never write "I", "I'll", "I need to", "Let me", or "we". There is no first \
person in a brief.

An assumption is something you CHOSE that the request did not say. Restating \
the request is not an assumption. If you assumed nothing, return an empty \
list -- that is a better answer than three lines of paraphrase.

Decide `intent`, and decide it from what the user wants BACK, not from what \
the request is about.

- "how does the playbook search work" -> question.
- "would it scale to a few thousand documents" -> question.
- "what would I have to change to index something other than Markdown" -> \
QUESTION. Its subject is a change and it is still a question: the user asked \
to be told, not to be given a diff. "Explain what would have to change..." is \
the brief; "Modify the pipeline so that..." is a different request that \
nobody made.
- "add support for .txt" / "fix the login bug" -> work.

The grammar is usually the tell: "what would I", "how does", "can I", "would \
it" are interrogative. "Add", "fix", "make", "refactor" are imperative.

**When the two readings are both defensible, choose question.** Answering a \
question when work was wanted costs one more turn. Doing work when a question \
was asked costs a diff nobody asked for, against code the user was still \
deciding about.

A question brief is still written in the imperative -- "Explain...", \
"Identify...", "Trace..." -- because the brief instructs a reader. Imperative \
MOOD is not a mandate to CHANGE anything.

Repository (path | size | what it is):
{sources}
"""


# Appended to every brief by `compose_brief`, never generated.
#
# This was a paragraph in ENHANCER_SYSTEM telling the model to write the
# requirement "in your own words", and it was ignored by both models that saw
# it -- neither brief contained a trace of it. That is the predictable outcome:
# it is the SAME SENTENCE every time, so asking a model to reproduce it pays
# tokens for a constant and then depends on a weak model's instruction
# following for whether a standing policy appears at all.
#
# The rule this is an instance of: if you can enforce it, do not ask for it.
# `relevant_paths` is trimmed in code for exactly this reason; putting the
# executor's policy in the generated half was the same mistake in the other
# direction, in the same commit.
STANDING_REQUIREMENTS = """\
Verify every factual claim about the code by RUNNING it -- reading it is not \
verifying it. A confident claim that a file does not parse, delivered with a \
line number and a verbatim quote, is indistinguishable from a real finding \
until somebody re-runs it.

Run everything in a scratch directory outside the repository. Write nothing \
inside it, take no lock on it, and run no command that does -- `git status` \
takes one. Leave the working tree exactly as you found it.

State the interpreter version you verified against and whether the tree was \
clean. Every number states how it was obtained; a figure with no method is \
worth less than no figure."""


# A polite imperative, not a question. "Can you add .txt support" is a request
# for work wearing a question mark, and treating it as interrogative would
# make the override below fire on exactly the case it must not.
_POLITE = re.compile(r"^\s*(?:please\b|(?:can|could|would|will)\s+you\b)", re.IGNORECASE)

# Interrogative openings. The user asking to be TOLD something, in the
# grammar they actually use.
_INTERROGATIVE = re.compile(
    r"^\s*(?:how|what|why|when|where|which|who|whose"
    r"|does|do|did|is|are|was|were|can|could|would|should|will|has|have)\b",
    re.IGNORECASE,
)


def asks_a_question(raw_prompt: str) -> bool:
    """Is this request interrogative, by its grammar alone?

    Deterministic, and it OVERRIDES the model when the two disagree. The
    first live `run` asked "how does the playbook index avoid returning stale
    results after a file changes" and a 2.6B model labelled it `work` -- which
    would have sent the planner to audit the retrieval code for defects
    instead of explaining it, at frontier-tier prices.

    `intent` is a field the model fills, and a field a model fills is a
    request. Whether a sentence opens with "how does" is a fact about the
    sentence, so it is checked rather than asked. Same rule as trimming
    `relevant_paths` and rejecting narration in code.

    The override runs ONE WAY on purpose. Grammar that says "question" wins,
    because answering when work was wanted costs a turn and working when an
    answer was wanted costs a diff nobody asked for. Grammar that says
    nothing leaves the model's judgement alone: "add support for .txt" is not
    interrogative, and neither is "the login page 500s on a bad token".
    """
    text = raw_prompt.strip()
    if _POLITE.match(text):
        return False
    return bool(_INTERROGATIVE.match(text)) or text.endswith("?")


def compose_brief(enhanced: str) -> str:
    """The brief as it is handed over: the sharpened request, then the policy.

    Two halves with different authors. The first is what the model was asked
    for -- this request, made precisely. The second is constant, and a
    constant belongs in the program.
    """
    return f"{enhanced.strip()}\n\n{STANDING_REQUIREMENTS}"


# How a model announces it is about to do the work instead of writing the
# brief. A fixed shape, so it is detected rather than asked against: two live
# runs produced "I'll examine the playbook search implementation... Let me
# start by looking at" and "I need to read the playbook indexing files... Let
# me start by reading the key files". Both are role-play, neither is a brief,
# and both passed every structural check there was.
# Two shapes, because the first pattern here caught only one of them and the
# very next live run produced the other.
#
# First person: "I'll examine X. Let me start by reading Y."
_NARRATION = re.compile(
    r"^\s*(?:i['’]?(?:ll|m|\s+will|\s+need|\s+should|\s+am|\s+can)"
    r"|let(?:'|’)?s?\s+(?:me|us)?|we(?:['’]ll|\s+will|\s+need|\s+should)"
    r"|first,?\s+i|to\s+answer\s+this,?\s+i)\b",
    re.IGNORECASE,
)

# Bare gerund: "Looking at the playbook index mechanism to understand how
# stale results are avoided." No first person at all, so the pattern above
# walked straight past it -- and it is the same failure. A participle names
# an activity in progress; a brief names a task to be done. The difference
# is whether the reader is being told what someone is doing or what to do.
#
# Only ONE clause is inspected: a brief may legitimately contain "...,
# checking each caller" further in. This fires on the opening word, where a
# verb belongs.
_GERUND_OPENING = re.compile(
    r"^\s*(?:look|examin|review|analy[sz]|investigat|explor|check|inspect|read"
    r"|trac|stud|consider|assess|evaluat|search|start|begin)\w*ing\b",
    re.IGNORECASE,
)


class EnhancedPrompt(BaseModel):
    """What the enhancer returns. The human approves this, so it has to be
    readable at a glance, not a wall of text."""

    enhanced: str = Field(description="The rewritten brief. Self-contained.")
    intent: Literal["question", "work"] = Field(
        default="question",
        description=(
            "What the user wants back. 'question' if they asked to be told something, "
            "even when the subject is a hypothetical change. 'work' only if they asked "
            "for the codebase to be different afterwards. When both readings are "
            "defensible, 'question'."
        ),
    )

    @field_validator("enhanced")
    @classmethod
    def _must_be_a_brief_not_a_plan_to_write_one(cls, value: str) -> str:
        """Reject agent narration at the contract, not in a review.

        The prompt already forbids the first person. It was ignored by both
        models that saw it, which is the ordinary outcome for a weak model and
        a long instruction -- and a rule that only exists in a prompt is a
        request. This one is enforceable by inspection, so it is enforced:
        `SCHEMA_INVALID` here buys a repair call that names the actual problem,
        and a model that narrates twice loses its turn to the next in the
        chain.

        Same rule as trimming `relevant_paths` in code rather than asking
        nicely. If you can enforce it, do not ask for it.
        """
        if _NARRATION.match(value) or _GERUND_OPENING.match(value):
            opening = " ".join(value.split()[:8])
            raise ValueError(
                "a brief is a task, not an announcement that you are about to do it. "
                f"This began {opening!r}. Rewrite it in the imperative mood, addressed "
                "to the person who will do the work, with no first person."
            )
        return value

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

    # 1200, matching `WorkerReport.summary` -- and both match
    # `WorkerResult.summary`, which is what they are assigned to. This was
    # 2000, and a live `answer` worker wrote 1,300 characters: pydantic
    # refused the WorkerResult, the exception escaped a Send branch, and the
    # whole superstep died. "One task failed" became "the run is gone", which
    # this node's docstring calls the worst outcome available in the design.
    #
    # The cap belongs on the MODEL's contract, not only at the seam, so an
    # overrun becomes a schema failure with a repair attempt instead of a
    # silent truncation. The full text is in the artifact either way; this
    # field is the summary state carries (ADR-008).
    answer: str = Field(
        description=(
            "The answer to the question asked. Prose, not a list."
            " Keep it under 1200 characters -- detail belongs in the observations."
        ),
        max_length=1200,
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

The brief opens by stating whether the user asked a QUESTION or asked for \
WORK. That line decides the capability: a question is answered by `answer` \
tasks, work is examined by `analyse` tasks. Do not override it from the \
wording -- "what would I have to change" is a question about a change, and \
auditing the code for defects is not an answer to it.

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
{section_map}

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
