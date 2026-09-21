"""Did the brief keep what the request said?

Built from a real failure: on 2026-09-17 a 200-word request with four explicit
constraints came back from the enhancer as one sentence, and the constraint
that vanished was "do not choose a budget value" -- the one whose absence
would have made the coding agent invent an unmeasured constant (§4.5).

The enhancer's prompt had said "Preserve the user's intent exactly" since it
was written. It was ignored. This is the check that replaces the asking.
"""

from __future__ import annotations

import pytest

from dynaflows.graph.fidelity import concerns_about

pytestmark = pytest.mark.deterministic

# The actual request and the actual brief from the run that motivated this.
REQUEST = (
    "Give repository.section_map() a token budget. It is 5,191 tokens, 54% of the "
    "planner prompt, and uncapped.\n"
    "Do NOT rely on filtering to formal anchors to get under a cap: 90 of 105 rows "
    "already carry one (repository._FORMAL_RE), so it saves about 14%.\n"
    "Make truncation visible at gate G2 rather than silent, the way build_source_map "
    "already does via SourceMap.truncated and render().\n"
    "Do not choose a budget value. Surface the trade-off and leave the number to a human."
)
BRIEF = "Modify repository.section_map() to enforce a token budget for the planner prompt."


def test_the_real_failure_is_caught() -> None:
    found = concerns_about(REQUEST, BRIEF)
    assert found, "the run that motivated this module must not pass silently"


def test_the_dropped_prohibition_is_named() -> None:
    """The one that mattered. Without it the agent picks a number, and §4.5
    calls an unmeasured constant a placeholder."""
    joined = " ".join(concerns_about(REQUEST, BRIEF))
    assert "do not choose a budget value" in joined.lower()


def test_the_dropped_identifiers_are_named() -> None:
    joined = " ".join(concerns_about(REQUEST, BRIEF))
    assert "_FORMAL_RE" in joined
    assert "5,191" in joined


def test_severe_shrinkage_is_reported_as_a_percentage() -> None:
    joined = " ".join(concerns_about(REQUEST, BRIEF))
    assert "shorter than your request" in joined


# --------------------------------------------------------------------------
# It must stay quiet on ordinary work. A gate that fires on correct output is
# a gate people stop reading (playbook 5.2, Pattern 5).
# --------------------------------------------------------------------------


def test_an_unchanged_brief_raises_nothing() -> None:
    assert concerns_about(REQUEST, REQUEST) == []


def test_a_light_edit_raises_nothing() -> None:
    """What a good enhancement of a precise request looks like: a qualified
    name, a hedge on a measured figure, everything else intact."""
    edited = REQUEST.replace(
        "repository.section_map()", "PlaybookRepository.section_map()"
    ).replace("5,191 tokens", "~5,191 tokens")
    assert concerns_about(REQUEST, edited) == []


def test_a_short_request_is_not_judged_on_length() -> None:
    """"fix the login bug" expanded into a real brief is the enhancer working.
    Flagging the reverse case on a two-line request would fire constantly."""
    assert not any(
        "shorter" in c for c in concerns_about("fix the login bug", "Fix the bug in login.")
    )


def test_an_expanded_brief_raises_nothing() -> None:
    brief = (
        "Fix the authentication bug in src/auth/session.py where expired tokens "
        "are accepted. Add a test covering expiry at the boundary."
    )
    assert concerns_about("fix the login bug", brief) == []


def test_empty_input_is_not_a_concern() -> None:
    assert concerns_about("", "something") == []
    assert concerns_about("something", "") == []


# --------------------------------------------------------------------------
# What it does and does not claim.
# --------------------------------------------------------------------------


def test_a_prohibition_kept_in_other_words_is_not_flagged() -> None:
    """It reads tokens, not meaning. A negation that survives with its
    identifiers intact is left alone -- the check is a tripwire, and a
    tripwire that fires on a correct paraphrase is noise."""
    raw = "Do not use the ADR-009 anchors for this. " + "x" * 300
    brief = "Never use the ADR-009 anchors for this. " + "x" * 300
    assert not any("constraint" in c for c in concerns_about(raw, brief))


def test_it_reports_rather_than_rejects() -> None:
    """The contract: a list for a human at G1, never a verdict. A brief that
    drops something may still be right."""
    found = concerns_about(REQUEST, BRIEF)
    assert isinstance(found, list)
    assert all(isinstance(item, str) for item in found)


def test_many_dropped_identifiers_are_summarised_not_dumped() -> None:
    raw = " ".join(f"ADR-{n:03d}" for n in range(1, 30)) + " and more text " * 20
    joined = " ".join(concerns_about(raw, "Do the thing."))
    assert "more)" in joined, "a gate line must not be 29 identifiers long"
