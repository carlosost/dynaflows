"""ADR-022: scoring a worker against defects whose location is known.

The scorer is deterministic and must stay that way: it is the only thing in
this project that can tell a clean codebase from a blind worker, and a scorer
that needs judgement to interpret answers neither question.
"""

from __future__ import annotations

import pytest

from dynaflows.calibration import fixture_paths, load_manifest, score
from dynaflows.graph.prompts import Finding

pytestmark = pytest.mark.deterministic


def a_finding(lines: str, **kwargs: object) -> Finding:
    return Finding(
        claim=kwargs.pop("claim", "swallowed"),  # type: ignore[arg-type]
        file=kwargs.pop("file", "broken_client.py"),  # type: ignore[arg-type]
        lines=lines,
        evidence=kwargs.pop("evidence", "except Exception:"),  # type: ignore[arg-type]
        severity=kwargs.pop("severity", "high"),  # type: ignore[arg-type]
        remediation=kwargs.pop("remediation", "Re-raise it."),  # type: ignore[arg-type]
    )


@pytest.fixture
def defects() -> list:
    return load_manifest(fixture_paths()[1])


# --- the fixture and its manifest must agree -----------------------------


def test_every_planted_defect_points_at_real_lines(defects: list) -> None:
    """The manifest is the answer key. A wrong line number here scores a
    correct worker as a failure -- and the first version of this manifest had
    five wrong ranges, written from memory instead of from the file."""
    source = fixture_paths()[0].read_text(encoding="utf-8").splitlines()

    assert defects
    for defect in defects:
        assert 1 <= defect.start <= defect.end <= len(source), defect.id
        assert any(source[i].strip() for i in range(defect.start - 1, defect.end)), defect.id


def test_the_fixture_carries_no_marker_a_model_could_read(defects: list) -> None:
    """Otherwise this is a reading-comprehension test, not an analysis one."""
    source = fixture_paths()[0].read_text(encoding="utf-8").lower()

    for word in ("bug", "defect", "planted", "deliberate", "wrong", "fixme", "todo"):
        assert word not in source, word
    for defect in defects:
        assert defect.id not in source


def test_the_planted_ranges_do_not_overlap(defects: list) -> None:
    """Overlapping ranges would let one finding claim two defects and inflate
    recall."""
    ordered = sorted(defects, key=lambda d: d.start)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        assert earlier.end < later.start, f"{earlier.id} overlaps {later.id}"


# --- scoring --------------------------------------------------------------


def test_a_finding_inside_a_planted_range_counts(defects: list) -> None:
    card = score([a_finding("28-33")], defects, reported=1, discarded=0)

    assert card.found == ("swallowed-fetch",)
    assert card.recall == pytest.approx(1 / card.planted)


def test_a_finding_that_overlaps_the_range_counts(defects: list) -> None:
    """A model citing the `except` line and one citing the `return None`
    beneath it found the same defect. Scoring them differently measures
    citation style, not detection."""
    assert score([a_finding("33")], defects, reported=1, discarded=0).found == ("swallowed-fetch",)
    assert score([a_finding("25-30")], defects, reported=1, discarded=0).found == (
        "swallowed-fetch",
    )


def test_two_findings_on_one_defect_are_one_hit(defects: list) -> None:
    card = score([a_finding("28"), a_finding("33")], defects, reported=2, discarded=0)

    assert card.found == ("swallowed-fetch",)


def test_a_finding_outside_every_range_is_neither_a_hit_nor_a_failure(defects: list) -> None:
    """It may be a real defect nobody planted. Counting it as a miss would
    punish a worker for being right about something else."""
    card = score([a_finding("20-21")], defects, reported=1, discarded=0)

    assert card.found == ()
    assert len(card.unplanted) == 1


def test_an_unparseable_range_is_not_credited(defects: list) -> None:
    card = score([a_finding("around the top")], defects, reported=1, discarded=0)

    assert card.found == ()
    assert len(card.unplanted) == 1


def test_recall_is_zero_when_nothing_is_reported(defects: list) -> None:
    """The `s2` shape. It must score as zero rather than divide by nothing."""
    card = score([], defects, reported=0, discarded=0)

    assert card.recall == 0.0
    assert card.discard_rate == 0.0
    assert len(card.missed) == card.planted


def test_the_discard_rate_counts_claims_not_survivors(defects: list) -> None:
    """A finding rejected as ungrounded still says the worker tried. Counting
    only what survived would hide a worker that is inventing."""
    card = score([a_finding("28-33")], defects, reported=4, discarded=3)

    assert card.discard_rate == pytest.approx(0.75)
    assert card.recall > 0
