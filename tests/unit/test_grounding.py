"""ADR-019: a citation is checkable by string search.

Run `w1` is the subject. A worker given no source invented `gateway/logger.py`,
`gateway/middleware.py`, `gateway/handlers.py` and `gateway/metrics.py`, cited
"Lines 70-80" in them, and reported a HIGH severity finding. Every one of those
citations fails the first check here, with no model and no second call.

The opposite risk is the one that gets a check deleted: if this drops findings
that are true but paraphrased, people will turn it off. Half the tests below
are about NOT firing.
"""

from __future__ import annotations

import pytest

from dynaflows.contracts.playbook import Chunk
from dynaflows.graph.grounding import Ungrounded, parse_range, verify
from dynaflows.graph.prompts import Finding
from dynaflows.store.sources import number_lines

pytestmark = pytest.mark.deterministic

_SOURCE = """def login(user, password):
    if not user:
        return None
    return check(user, password)
"""


def a_chunk(path: str = "src/auth.py", text: str = _SOURCE) -> Chunk:
    numbered = number_lines(text)
    return Chunk(
        id=f"src:{path}",
        source_path=path,
        heading_path=path,
        anchors=(path,),
        body=numbered,
        tokens=len(numbered) // 4,
    )


def a_finding(**kwargs: object) -> Finding:
    return Finding(
        claim=kwargs.pop("claim", "no password check"),  # type: ignore[arg-type]
        file=kwargs.pop("file", "src/auth.py"),  # type: ignore[arg-type]
        lines=kwargs.pop("lines", "2-3"),  # type: ignore[arg-type]
        evidence=kwargs.pop("evidence", "if not user:"),  # type: ignore[arg-type]
        severity=kwargs.pop("severity", "high"),  # type: ignore[arg-type]
        remediation=kwargs.pop("remediation", "Validate the password."),  # type: ignore[arg-type]
    )


# --- what it must catch --------------------------------------------------


def test_a_file_the_worker_never_saw_is_rejected() -> None:
    """`w1`, exactly: gateway/logger.py did not exist and was not in any pack."""
    grounding = verify([a_finding(file="gateway/logger.py")], [a_chunk()])

    assert grounding.kept == ()
    assert grounding.dropped[0].reason is Ungrounded.UNKNOWN_FILE


def test_a_line_past_the_end_of_the_file_is_rejected() -> None:
    """`w1` cited "Lines 70-80" of a four-line world."""
    grounding = verify([a_finding(lines="70-80")], [a_chunk()])

    assert grounding.dropped[0].reason is Ungrounded.OUT_OF_RANGE


def test_evidence_that_is_not_in_the_file_is_rejected() -> None:
    grounding = verify([a_finding(evidence="log.info(user.password)")], [a_chunk()])

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND


def test_an_unparseable_range_is_rejected_rather_than_guessed() -> None:
    grounding = verify([a_finding(lines="somewhere near the top")], [a_chunk()])

    assert grounding.dropped[0].reason is Ungrounded.BAD_RANGE


def test_a_section_dropped_for_budget_is_not_citable() -> None:
    """Checked against what the worker ACTUALLY saw. A claim about a chunk that
    never made it into the pack cannot have been read."""
    grounding = verify([a_finding()], [])

    assert grounding.dropped[0].reason is Ungrounded.UNKNOWN_FILE


def test_all_ungrounded_is_not_the_same_as_no_findings() -> None:
    """A worker whose every claim was thrown out is a very different thing
    from one that claimed nothing, and the status depends on telling them
    apart."""
    invented = verify([a_finding(file="nope.py")], [a_chunk()])
    silent = verify([], [a_chunk()])

    assert invented.all_ungrounded is True
    assert silent.all_ungrounded is False


# --- what it must NOT catch ----------------------------------------------


def test_a_correct_citation_survives() -> None:
    grounding = verify([a_finding()], [a_chunk()])

    assert len(grounding.kept) == 1
    assert grounding.dropped == ()


def test_quoting_with_the_line_number_prefix_survives() -> None:
    """The worker sees `2|     if not user:`. Copying it verbatim -- which is
    what it was told to do -- must not be a grounding failure."""
    grounding = verify([a_finding(evidence="2|     if not user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_reindented_evidence_survives() -> None:
    """Whitespace is collapsed on both sides. A model that re-indents a quote
    has not invented it, and treating that as fabrication is how a check gets
    switched off (playbook 5.2, Pattern 5)."""
    grounding = verify([a_finding(evidence="if   not  user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_a_single_line_citation_survives() -> None:
    grounding = verify([a_finding(lines="2")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_a_slightly_wrong_line_number_with_real_evidence_survives() -> None:
    """Deliberate. Evidence is checked against the whole chunk, not the cited
    span: an off-by-two line reference with a real quote is a citation error,
    and discarding a true finding over it trades a visible problem for an
    invisible one."""
    grounding = verify([a_finding(lines="1", evidence="if not user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_counts_are_kept_apart() -> None:
    """AP-20. Claims made and claims that survived are two facts; equal numbers
    mean a careful worker and a large gap means one that is inventing, and one
    number cannot say which."""
    grounding = verify(
        [a_finding(), a_finding(file="ghost.py"), a_finding(evidence="nope")], [a_chunk()]
    )

    assert grounding.reported == 3
    assert len(grounding.kept) == 1
    assert len(grounding.dropped) == 2


def test_a_dropped_finding_says_what_it_claimed() -> None:
    """Written to the artifact rather than discarded silently: a reader who
    cannot see six thrown-out claims will read the silence as diligence."""
    grounding = verify([a_finding(file="ghost.py")], [a_chunk()])

    rendered = grounding.dropped[0].render()
    assert "ghost.py" in rendered
    assert "not in its context" in rendered


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4", (4, 4)), ("4-9", (4, 9)), (" 4 - 9 ", (4, 9)), ("4–9", (4, 9)), ("4:9", (4, 9))],
)
def test_ranges_people_actually_write(raw: str, expected: tuple[int, int]) -> None:
    assert parse_range(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "0", "9-4", "-3"])
def test_ranges_that_are_not_ranges(raw: str) -> None:
    assert parse_range(raw) is None


# --- the check was brittle in one direction, and s4 found it -------------


def test_a_real_quote_with_an_added_delimiter_survives() -> None:
    """Run `s4`: five of eight findings discarded, every one citing a line that
    existed. A model quoting the first line of a MULTI-line docstring closes it
    with a delimiter the source does not have at that point. That is a real
    quote with a syntactic completion attached, not a fabrication."""
    chunk = a_chunk(text='"""One line of a docstring.\n\nMore prose.\n"""\nx = 1\n')

    grounding = verify([a_finding(lines="1", evidence='"""One line of a docstring."""')], [chunk])

    assert len(grounding.kept) == 1


def test_a_quote_with_a_trailing_ellipsis_survives() -> None:
    grounding = verify([a_finding(evidence="if not user:  ...")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_an_invented_line_still_fails() -> None:
    """The loosening must not become an opening. `w1` is the reason this file
    exists."""
    grounding = verify(
        [a_finding(evidence='log.audit("this line was never written")')], [a_chunk()]
    )

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND


def test_a_quote_too_thin_to_prove_anything_fails() -> None:
    """`)` is in every Python file. Accepting it would make the check a
    formality."""
    for thin in (")", "...", "else:", "   "):
        grounding = verify([a_finding(evidence=thin)], [a_chunk()])
        assert grounding.dropped, thin
        assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND, thin


def test_one_real_line_and_one_invented_line_fails() -> None:
    """Every substantial line must be present, not one of them -- otherwise a
    true quote becomes cover for a false one."""
    grounding = verify(
        [a_finding(evidence="if not user:\n    self.audit_log.write(password)")], [a_chunk()]
    )

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND


# --- grounded, and worthless ---------------------------------------------


def test_a_finding_proposing_no_fix_is_not_a_finding() -> None:
    """Run `s4` verified three of these: real file, real lines, verbatim quote,
    severity "low", remediation "No remediation needed" -- and the synthesis
    summarised what the code does. Verification that validates form and not
    substance is a rubber stamp with extra steps."""
    for remedy in ("No remediation needed", "none", "N/A", "No action required.", ""):
        grounding = verify([a_finding(remediation=remedy)], [a_chunk()])
        assert grounding.dropped, remedy
        assert grounding.dropped[0].reason is Ungrounded.NOT_A_DEFECT, remedy


def test_a_real_remediation_is_not_mistaken_for_a_non_finding() -> None:
    """Deliberately narrow: it matches an explicit "nothing to do here", not a
    short fix. A gate that fires on ordinary work gets disabled."""
    for remedy in ("Re-raise it.", "Validate the password", "Use a typed error", "Log and raise"):
        grounding = verify([a_finding(remediation=remedy)], [a_chunk()])
        assert len(grounding.kept) == 1, remedy
