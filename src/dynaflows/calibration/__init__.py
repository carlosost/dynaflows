"""Known-answer calibration for the worker. ADR-022.

"No findings" is unfalsifiable on its own: a clean codebase and a worker that
cannot find anything produce identical output. This package plants defects in
a fixture, records where they are, and scores what comes back.

It measures the WORKER, not the truth. Five planted defects say nothing about
the defects nobody planted, and full recall here must never be read as "the
pipeline finds bugs". It is a floor.

`token_bucket.py` is the same idea for `answer` (ADR-024), where the
unfalsifiable thing is not "no findings" but "a plausible answer". It scores
two signals apart -- did the worker cite the lines that hold the answer, and
does the answer carry the fact those lines establish -- because the failure
worth catching is the one where they disagree: a fluent, correct answer with
nothing behind it is a model reciting priors about a file it did not read.
"""

from dynaflows.calibration.harness import (
    Aggregate,
    AnswerResult,
    AnswerScorecard,
    Defect,
    Question,
    Scorecard,
    aggregate,
    fixture_paths,
    load_manifest,
    load_questions,
    score,
    score_answer,
)

__all__ = [
    "Aggregate",
    "AnswerResult",
    "AnswerScorecard",
    "Defect",
    "Question",
    "Scorecard",
    "aggregate",
    "fixture_paths",
    "load_manifest",
    "load_questions",
    "score",
    "score_answer",
]
