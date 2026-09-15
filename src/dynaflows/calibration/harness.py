"""Loading the manifest and scoring a worker's findings against it. ADR-022.

Deterministic and model-free, like every other check in this project that
decides whether something is true.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dynaflows.graph.grounding import parse_range
from dynaflows.graph.prompts import Finding, Observation

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True, slots=True)
class Defect:
    id: str
    start: int
    end: int
    kind: str
    why: str

    def covers(self, start: int, end: int) -> bool:
        """Any overlap counts.

        A model citing the `except` line and one citing the `return None`
        beneath it have found the same defect, and scoring them differently
        would measure citation style rather than detection.
        """
        return start <= self.end and end >= self.start


@dataclass(frozen=True, slots=True)
class Scorecard:
    findings: tuple[Finding, ...]
    found: tuple[str, ...]
    missed: tuple[Defect, ...]
    unplanted: tuple[Finding, ...]
    discarded: int
    reported: int

    @property
    def planted(self) -> int:
        return len(self.found) + len(self.missed)

    @property
    def recall(self) -> float:
        return len(self.found) / self.planted if self.planted else 0.0

    @property
    def discard_rate(self) -> float:
        """The §4.5 number ADR-019 said would decide whether the evidence match
        is too strict. It has had no measurement behind it until now."""
        return self.discarded / self.reported if self.reported else 0.0


def fixture_paths(name: str = "broken_client") -> tuple[Path, Path]:
    return FIXTURES / f"{name}.py", FIXTURES / f"{name}.manifest.toml"


def load_manifest(path: Path) -> list[Defect]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Defect(
            id=str(entry["id"]),
            start=int(entry["lines"][0]),
            end=int(entry["lines"][1]),
            kind=str(entry.get("kind", "")),
            why=str(entry.get("why", "")),
        )
        for entry in raw.get("defect", [])
    ]


def score(
    findings: list[Finding], defects: list[Defect], *, reported: int, discarded: int
) -> Scorecard:
    """Match verified findings to planted defects by line overlap.

    `reported` and `discarded` come from the grounding check, not from the list
    above: a finding rejected as ungrounded still tells us the worker tried,
    and counting only what survived would hide a worker that is inventing.
    """
    found: list[str] = []
    unplanted: list[Finding] = []

    for finding in findings:
        span = parse_range(finding.lines)
        if span is None:
            unplanted.append(finding)
            continue
        hit = next((d for d in defects if d.covers(*span)), None)
        if hit is None:
            unplanted.append(finding)
        elif hit.id not in found:
            found.append(hit.id)

    return Scorecard(
        findings=tuple(findings),
        found=tuple(found),
        missed=tuple(d for d in defects if d.id not in found),
        unplanted=tuple(unplanted),
        discarded=discarded,
        reported=reported,
    )


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Several runs of the same model on the same fixture.

    One run is a sample, not a measurement. The same model scored 4/5 and then
    3/5 on this fixture with nothing changed between them, and a tier decision
    had already been made on the first number. A defect found in every run and
    one found in a third of them are different facts about a model, and a mean
    alone hides which is which.
    """

    cards: tuple[Scorecard, ...]
    hits: dict[str, int]
    planted: int

    @property
    def runs(self) -> int:
        return len(self.cards)

    @property
    def mean_recall(self) -> float:
        return sum(c.recall for c in self.cards) / self.runs if self.runs else 0.0

    @property
    def always(self) -> list[str]:
        return sorted(k for k, v in self.hits.items() if v == self.runs)

    @property
    def sometimes(self) -> list[str]:
        return sorted(k for k, v in self.hits.items() if 0 < v < self.runs)

    @property
    def never(self) -> list[str]:
        return sorted(k for k, v in self.hits.items() if v == 0)

    def summary(self) -> str:
        scores = ", ".join(f"{len(c.found)}/{c.planted}" for c in self.cards)
        return f"{scores} (mean {self.mean_recall:.0%} over {self.runs} run(s))"


def aggregate(cards: list[Scorecard], defects: list[Defect]) -> Aggregate:
    hits = {d.id: 0 for d in defects}
    for card in cards:
        for found in card.found:
            hits[found] = hits.get(found, 0) + 1
    return Aggregate(cards=tuple(cards), hits=hits, planted=len(defects))


# --- the `answer` half (ADR-022 for ADR-024) -----------------------------
#
# Scoring prose is the problem this has to solve without an LLM judge, which
# ADR-004 rules out. Two mechanical signals, kept apart because they fail
# independently and only the PAIR is informative:
#
#   grounded  -- did it cite the lines that hold the answer?
#   correct   -- does the answer contain the fact those lines establish?
#
# Grounded-and-wrong is a model that looked and misread. Correct-but-
# ungrounded is worse and is the thing this fixture exists to catch: a model
# answering from priors, fluently, about a file it did not read. Collapsing
# them into one score would hide exactly that.


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    ask: str
    kind: str
    why: str
    start: int = 0
    end: int = 0
    contains: str = ""
    must_mention: tuple[str, ...] = ()
    answerable: bool = True

    def covers(self, start: int, end: int) -> bool:
        return bool(self.start) and start <= self.end and end >= self.start

    def satisfied_by(self, answer: str) -> bool:
        """Every `must_mention` pattern appears in the answer.

        Regex and case-insensitive, and each pattern is written to be
        satisfiable by any correct phrasing. A pattern a plausible WRONG
        answer could also match measures nothing, which is why they are
        reviewed in the manifest beside the reason for the question.
        """
        return all(re.search(pattern, answer, re.IGNORECASE) for pattern in self.must_mention)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: Question
    grounded: bool
    correct: bool
    declared_insufficient: bool

    @property
    def passed(self) -> bool:
        if not self.question.answerable:
            # The only right answer is to say the file does not contain one.
            # Citing something anyway is the `w1` failure, and it is scored as
            # a failure here however well-formed the citation is.
            return self.declared_insufficient and not self.grounded
        return self.grounded and self.correct


@dataclass(frozen=True, slots=True)
class AnswerScorecard:
    results: tuple[AnswerResult, ...]

    @property
    def asked(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def answered_from_priors(self) -> tuple[AnswerResult, ...]:
        """Right answer, no citation to support it.

        The headline number. It is not a scoring detail: a tool whose answers
        are right when the model already knew and unchecked when it did not
        is a tool that cannot be trusted on the questions that matter.
        """
        return tuple(
            r for r in self.results if r.question.answerable and r.correct and not r.grounded
        )

    @property
    def looked_but_misread(self) -> tuple[AnswerResult, ...]:
        return tuple(
            r for r in self.results if r.question.answerable and r.grounded and not r.correct
        )

    @property
    def invented(self) -> tuple[AnswerResult, ...]:
        """Cited something for a question the file cannot answer."""
        return tuple(r for r in self.results if not r.question.answerable and r.grounded)


def load_questions(path: Path) -> list[Question]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for entry in raw.get("question", []):
        lines = entry.get("lines") or [0, 0]
        questions.append(
            Question(
                id=str(entry["id"]),
                ask=str(entry["ask"]),
                kind=str(entry.get("kind", "")),
                why=str(entry.get("why", "")),
                start=int(lines[0]),
                end=int(lines[1]),
                contains=str(entry.get("contains", "")),
                must_mention=tuple(str(p) for p in entry.get("must_mention", [])),
                answerable=bool(entry.get("answerable", True)),
            )
        )
    return questions


def score_answer(
    question: Question,
    answer: str,
    observations: list[Observation],
    *,
    context_was_sufficient: bool,
) -> AnswerResult:
    """One question, scored against what the worker actually returned.

    `observations` are the VERIFIED ones -- whatever survived the citation
    check. An observation quoting a line the worker was never shown has
    already been dropped by then, so this cannot reward a fabricated cite.
    """
    if question.answerable:
        grounded = any(
            (span := parse_range(o.lines)) is not None and question.covers(*span)
            for o in observations
        )
        correct = question.satisfied_by(answer)
    else:
        # An unanswerable question has no anchor, so "did it cite the right
        # lines" is the wrong test and the first version of this silently
        # returned False for every observation -- which made `invented`
        # unreachable, defeating the one detection this fixture exists for.
        #
        # Here ANY surviving citation is the finding: the file does not
        # contain the answer, so a citation into it is the worker inventing
        # support for something it made up.
        grounded = bool(observations)
        # `must_mention` is empty for these, and `all([])` is True -- so a
        # meaningless `correct` would read as a pass in any summary that
        # printed it. There is no right prose here, only a right refusal.
        correct = False
    return AnswerResult(
        question=question,
        grounded=grounded,
        correct=correct,
        declared_insufficient=not context_was_sufficient,
    )
