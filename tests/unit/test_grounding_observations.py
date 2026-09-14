"""The one rule that differs between `analyse` and `answer`.

Everything about a citation -- the file was shown, the range exists, the quote
is really there -- is wrong whatever was asked, and is the same code. The
substance rule is not: NOT_A_DEFECT drops a claim that merely describes the
code, which is right for an audit and exactly backwards for a question.

These tests exist because that inversion is the entire reason `answer` is a
separate capability rather than a flag on `analyse`.
"""

from __future__ import annotations

import pytest

from dynaflows.contracts.playbook import Chunk
from dynaflows.graph import grounding
from dynaflows.graph.prompts import Finding, Observation

pytestmark = pytest.mark.deterministic

_BODY = "\n".join(
    [
        "def replace_source(connection, path, relative, chunks):",
        "    with connection:",
        "        for row in old:",
        "            connection.execute(",
        '                "INSERT INTO chunks_fts(chunks_fts, rowid, heading_path) "',
        "                \"VALUES ('delete', ?, '')\",",
        "            )",
    ]
)


def a_chunk() -> Chunk:
    return Chunk(
        id="c1",
        source_path="src/store.py",
        heading_path="src/store.py",
        anchors=("store",),
        body=_BODY,
        tokens=40,
    )


def an_observation(**over: str) -> Observation:
    base = {
        "claim": "The FTS5 delete passes an empty string for heading_path.",
        "file": "src/store.py",
        "lines": "5-6",
        "quoted_lines": "\"VALUES ('delete', ?, '')\",",
    }
    return Observation(**{**base, **over})


def test_a_description_of_working_code_is_kept_for_an_answer() -> None:
    """The inversion. As a finding this is dropped NOT_A_DEFECT; as an answer
    to "how does the delete work" it is the answer."""
    result = grounding.verify_observations([an_observation()], [a_chunk()])

    assert len(result.kept) == 1
    assert not result.dropped


def test_the_same_claim_as_a_finding_is_dropped() -> None:
    """Pinned side by side, because the two rules are one decision and reading
    either alone makes the other look like an oversight."""
    finding = Finding(
        claim="The FTS5 delete passes an empty string for heading_path.",
        failure="none",
        file="src/store.py",
        lines="5-6",
        quoted_lines="\"VALUES ('delete', ?, '')\",",
        severity="low",
        remediation="No remediation needed",
    )

    result = grounding.verify([finding], [a_chunk()])

    assert not result.kept
    assert result.dropped[0].reason is grounding.Ungrounded.NOT_A_DEFECT


def test_an_invented_file_is_still_rejected() -> None:
    """Relaxing the substance rule must not relax the citation rules. Run `w1`
    invented four filenames; that is wrong whatever the question was."""
    result = grounding.verify_observations(
        [an_observation(file="src/dynaflows/rag/store.py")], [a_chunk()]
    )

    assert result.dropped[0].reason is grounding.Ungrounded.UNKNOWN_FILE


def test_an_invented_quote_is_still_rejected() -> None:
    result = grounding.verify_observations(
        [an_observation(quoted_lines="connection.execute('DROP TABLE chunks_fts')")],
        [a_chunk()],
    )

    assert result.dropped[0].reason is grounding.Ungrounded.EVIDENCE_NOT_FOUND


def test_a_line_range_past_the_end_is_still_rejected() -> None:
    result = grounding.verify_observations([an_observation(lines="900")], [a_chunk()])

    assert result.dropped[0].reason is grounding.Ungrounded.OUT_OF_RANGE


def test_all_ungrounded_still_distinguishes_from_silence() -> None:
    """A worker whose every claim was thrown out is not a worker that made
    none, and the difference decides `ok` versus `degraded` (ADR-004)."""
    answered_badly = grounding.verify_observations([an_observation(file="nope.py")], [a_chunk()])
    said_nothing = grounding.verify_observations([], [a_chunk()])

    assert answered_badly.all_ungrounded
    assert not said_nothing.all_ungrounded
