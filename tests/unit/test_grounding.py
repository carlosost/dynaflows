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
        quoted_lines=kwargs.pop("quoted_lines", "if not user:"),  # type: ignore[arg-type]
        severity=kwargs.pop("severity", "high"),  # type: ignore[arg-type]
        failure=kwargs.pop(  # type: ignore[arg-type]
            "failure", "A request with no user returns None and the caller cannot tell why."
        ),
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
    grounding = verify([a_finding(quoted_lines="log.info(user.password)")], [a_chunk()])

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
    grounding = verify([a_finding(quoted_lines="2|     if not user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_reindented_evidence_survives() -> None:
    """Whitespace is collapsed on both sides. A model that re-indents a quote
    has not invented it, and treating that as fabrication is how a check gets
    switched off (playbook 5.2, Pattern 5)."""
    grounding = verify([a_finding(quoted_lines="if   not  user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_a_single_line_citation_survives() -> None:
    grounding = verify([a_finding(lines="2")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_a_slightly_wrong_line_number_with_real_evidence_survives() -> None:
    """Deliberate. Evidence is checked against the whole chunk, not the cited
    span: an off-by-two line reference with a real quote is a citation error,
    and discarding a true finding over it trades a visible problem for an
    invisible one."""
    grounding = verify([a_finding(lines="1", quoted_lines="if not user:")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_counts_are_kept_apart() -> None:
    """AP-20. Claims made and claims that survived are two facts; equal numbers
    mean a careful worker and a large gap means one that is inventing, and one
    number cannot say which."""
    grounding = verify(
        [a_finding(), a_finding(file="ghost.py"), a_finding(quoted_lines="nope")], [a_chunk()]
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

    grounding = verify(
        [a_finding(lines="1", quoted_lines='"""One line of a docstring."""')], [chunk]
    )

    assert len(grounding.kept) == 1


def test_a_quote_with_a_trailing_ellipsis_survives() -> None:
    grounding = verify([a_finding(quoted_lines="if not user:  ...")], [a_chunk()])

    assert len(grounding.kept) == 1


def test_an_invented_line_still_fails() -> None:
    """The loosening must not become an opening. `w1` is the reason this file
    exists."""
    grounding = verify(
        [a_finding(quoted_lines='log.audit("this line was never written")')], [a_chunk()]
    )

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND


def test_a_quote_too_thin_to_prove_anything_fails() -> None:
    """`)` is in every Python file. Accepting it would make the check a
    formality."""
    for thin in (")", "...", "else:", "   "):
        grounding = verify([a_finding(quoted_lines=thin)], [a_chunk()])
        assert grounding.dropped, thin
        assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND, thin


def test_one_real_line_and_one_invented_line_fails() -> None:
    """Every substantial line must be present, not one of them -- otherwise a
    true quote becomes cover for a false one."""
    grounding = verify(
        [a_finding(quoted_lines="if not user:\n    self.audit_log.write(password)")], [a_chunk()]
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


# --- a citation proves it read the file, not that it found anything ------


def test_a_finding_with_no_failure_scenario_is_a_description() -> None:
    """Run `s5` claimed fifteen findings. "Telemetry errors are detected and
    classified", "Settings are properly configured for telemetry" -- correct
    statements about working code, correctly cited, worth nothing. Citation
    checking proves the model READ the file; only this asks whether it found
    anything."""
    for failure in ("", "none", "N/A", "bad", "   "):
        grounding = verify([a_finding(failure=failure)], [a_chunk()])
        assert grounding.dropped, repr(failure)
        assert grounding.dropped[0].reason is Ungrounded.NOT_A_DEFECT, repr(failure)


def test_a_real_failure_scenario_passes() -> None:
    """Both fields are checked because either alone is easy to satisfy by
    accident. This must not fire on a concise, real one."""
    for failure in (
        "Returns None on a 404 and the caller retries",
        "A 401 is retried three times and sleeps between each",
        "The API key reaches the log at INFO on every call",
    ):
        grounding = verify([a_finding(failure=failure)], [a_chunk()])
        assert len(grounding.kept) == 1, failure


def test_an_elided_quote_survives() -> None:
    """Run `s5`: `def f(...) -> T: ... return T(...)` joins two real but
    non-adjacent fragments. Both are in the file; the joined string is not. An
    elision is the model saying "these two real fragments, with something
    between", which is a true statement about the file."""
    grounding = verify(
        [a_finding(quoted_lines="def login(user, password): ... return check(user, password)")],
        [a_chunk()],
    )

    assert len(grounding.kept) == 1


def test_an_elision_cannot_smuggle_an_invented_fragment() -> None:
    """Splitting on the ellipsis must not weaken the check: every fragment is
    still required to be present."""
    grounding = verify(
        [a_finding(quoted_lines="def login(user, password): ... audit.write(password)")],
        [a_chunk()],
    )

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND


def test_prose_in_the_quote_field_is_rejected() -> None:
    """Run `s6`: all three claims were real defect claims and all three put an
    EXPLANATION where the quote belongs -- "The function configure_tracing
    checks if LANGSMITH_TRACING and LANGSMITH_API_KEY are set. However, it does
    not provide a default value." True, and not a quotation.

    In an audit report "evidence" conventionally means the reasoning that
    supports a claim, so the field name was doing the opposite of its job. It
    is `quoted_lines` now, and this asserts the check still catches the prose
    that name is meant to prevent."""
    prose = (
        "The function configure_tracing checks if LANGSMITH_TRACING is set. "
        "However, it does not provide a default value for the unset case."
    )

    grounding = verify([a_finding(quoted_lines=prose)], [a_chunk()])

    assert grounding.dropped[0].reason is Ungrounded.EVIDENCE_NOT_FOUND
