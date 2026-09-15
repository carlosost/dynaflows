"""The known-answer fixture for `answer`. ADR-022 applied to ADR-024.

"No findings" was unfalsifiable and got a fixture. "A plausible answer" is
unfalsifiable in exactly the same way, and this is its fixture.

The scoring problem is that prose cannot be diffed and ADR-004 forbids an LLM
judge. Two mechanical signals instead -- did it cite the lines that hold the
answer, and does the answer carry the fact those lines establish -- kept
apart, because the interesting failure is the one where they disagree.
"""

from __future__ import annotations

import re

import pytest

from dynaflows.calibration.harness import (
    FIXTURES,
    AnswerScorecard,
    load_questions,
    score_answer,
)
from dynaflows.graph.prompts import Observation

pytestmark = pytest.mark.deterministic

SOURCE = FIXTURES / "token_bucket.py"
MANIFEST = FIXTURES / "token_bucket.manifest.toml"


def questions() -> dict[str, object]:
    return {q.id: q for q in load_questions(MANIFEST)}


def an_observation(lines: str) -> Observation:
    return Observation(claim="c", file="token_bucket.py", lines=lines, quoted_lines="q")


# --- the manifest describes the file it claims to describe ---------------


@pytest.mark.parametrize("question", [q for q in load_questions(MANIFEST) if q.start])
def test_every_anchor_points_at_what_it_claims(question: object) -> None:
    """The recorded failure, guarded.

    The last manifest in this project was written from memory and all five
    line ranges were wrong; two of them overlapped. A range nobody checks is
    a measurement nobody can trust, so every range carries a `contains`
    string and this asserts it against the source.
    """
    body = "\n".join(SOURCE.read_text().splitlines()[question.start - 1 : question.end])  # type: ignore[attr-defined]

    assert question.contains in body, (  # type: ignore[attr-defined]
        f"{question.id}: lines {question.start}-{question.end} do not contain "  # type: ignore[attr-defined]
        f"{question.contains!r}"  # type: ignore[attr-defined]
    )


def test_the_fixture_has_a_question_the_file_cannot_answer() -> None:
    """Without one, the fixture cannot detect the `w1` failure -- a worker
    that invents an answer rather than reporting it lacked the context. That
    is worth more than any correct answer here."""
    assert any(not q.answerable for q in load_questions(MANIFEST))


def test_the_fixture_has_a_question_priors_get_wrong() -> None:
    """The control/measurement pair. A fixture whose every question is also
    guessable from the method names measures fluency, not reading."""
    assert any(q.kind == "contradiction" for q in load_questions(MANIFEST))


def test_the_contradiction_is_really_in_the_source() -> None:
    """`retry_after` says seconds everywhere and returns milliseconds. If
    someone 'fixes' the fixture, the instrument is gone and the suite should
    say so rather than quietly scoring everything as correct."""
    text = SOURCE.read_text()

    assert "Seconds until `tokens` would be available" in text
    assert "* 1000" in text


# --- scoring -------------------------------------------------------------


def test_the_right_answer_with_the_right_citation_passes() -> None:
    question = questions()["retry-after-units"]

    result = score_answer(
        question,
        "It returns milliseconds, despite the docstring saying seconds.",
        [an_observation("87-89")],
        context_was_sufficient=True,
    )

    assert result.passed


def test_a_fluent_wrong_answer_fails() -> None:
    """The default outcome for a model that read the method name and the
    docstring and stopped. Every word of it is plausible."""
    question = questions()["retry-after-units"]

    result = score_answer(
        question,
        "It returns the number of seconds to wait, rounded up, for a Retry-After header.",
        [an_observation("78-80")],
        context_was_sufficient=True,
    )

    assert not result.passed
    assert not result.correct


def test_being_right_without_a_citation_is_recorded_separately() -> None:
    """The headline failure, and the reason the two signals are not one score.

    A model that answers correctly from priors looks identical to one that
    read the file -- until the question is one its priors get wrong. Counting
    these apart is what makes that distinguishable at all.
    """
    question = questions()["retry-after-units"]

    card = AnswerScorecard(
        (
            score_answer(
                question,
                "It returns milliseconds.",
                [],
                context_was_sufficient=True,
            ),
        )
    )

    assert card.passed == 0
    assert len(card.answered_from_priors) == 1


def test_looking_and_misreading_is_recorded_separately() -> None:
    question = questions()["retry-after-units"]

    card = AnswerScorecard(
        (
            score_answer(
                question,
                "It returns seconds.",
                [an_observation("84-89")],
                context_was_sufficient=True,
            ),
        )
    )

    assert len(card.looked_but_misread) == 1
    assert not card.answered_from_priors


def test_saying_the_file_cannot_answer_is_the_pass() -> None:
    question = questions()["callers"]

    result = score_answer(
        question,
        "The callers are not in the file I was shown.",
        [],
        context_was_sufficient=False,
    )

    assert result.passed


def test_inventing_an_answer_to_an_unanswerable_question_fails() -> None:
    """Run `w1`: a worker with no source invented four filenames, cited line
    numbers in them and reported a HIGH severity finding. A fixture whose
    every question has an answer cannot see that happen."""
    question = questions()["callers"]

    card = AnswerScorecard(
        (
            score_answer(
                question,
                "The gateway and the planner both call it on every request.",
                [an_observation("51-60")],
                context_was_sufficient=True,
            ),
        )
    )

    assert card.passed == 0
    assert len(card.invented) == 1


def test_an_out_of_range_citation_does_not_count_as_grounded() -> None:
    question = questions()["refill"]

    result = score_answer(
        question,
        "It computes the elapsed time and caps at capacity.",
        [an_observation("91-93")],
        context_was_sufficient=True,
    )

    assert result.correct
    assert not result.grounded


@pytest.mark.parametrize("question", [q for q in load_questions(MANIFEST) if q.must_mention])
def test_no_must_mention_pattern_matches_the_question_itself(question: object) -> None:
    """A pattern satisfied by echoing the question back is not a test of the
    answer. Cheap to write by accident and it would silently inflate every
    score from here on."""
    for pattern in question.must_mention:  # type: ignore[attr-defined]
        assert not re.search(pattern, question.ask, re.IGNORECASE), (  # type: ignore[attr-defined]
            f"{question.id}: {pattern!r} matches the question text"  # type: ignore[attr-defined]
        )
