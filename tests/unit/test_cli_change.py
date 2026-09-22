"""The `change` command's rendering and its one write. ADR-023, ADR-026.

The graph itself is covered by `test_write_pipeline.py`. What is tested here
is the part a human actually experiences: whether the gate says the dangerous
thing FIRST, and whether an approval is required before anything reaches the
user's files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from dynaflows import cli
from dynaflows.contracts.state import (
    ArtifactRef,
    ChangeSet,
    GateDecision,
    GateOutcome,
    WorkspaceRef,
)

pytestmark = pytest.mark.deterministic


def _render(payload: dict[str, Any]) -> str:
    buffer = Console(record=True, width=100)
    cli._render_change_gate(payload, out=buffer)
    return buffer.export_text()


def _payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gate": "change",
        "empty": False,
        "files": ["a.py"],
        "stat": " a.py | 2 +-",
        "branch": "dynaflows/chg-1",
        "comparable": True,
        "newly_failing": [],
        "newly_passing": [],
        "still_failing": [],
        "dirty_paths": [],
        "dirty_checked": True,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# The order is the design.
# --------------------------------------------------------------------------


def test_an_empty_diff_is_the_first_thing_said_and_nothing_else_is(
    ) -> None:
    """Every line below `empty` reads as success when the diff is empty.

    Measured live: `acceptEdits` exits 0 having silently refused a command,
    reports itself done, and leaves no diff.
    """
    text = _render(_payload(empty=True, agent_said="All done!", newly_passing=["t::x"]))

    assert "changed nothing" in text
    assert "t::x" not in text, "no verdict may be shown beside an empty diff"
    assert "Approving applies nothing" in text


def test_an_uncomparable_suite_is_called_unverified_not_clean() -> None:
    """An empty `newly_failing` from a suite that collected no tests looks
    exactly like a clean change."""
    text = _render(_payload(comparable=False, suite_tail="INTERNALERROR"))

    assert "could not be compared" in text
    assert "unverified" in text.lower()
    assert "Nothing that passed before fails now" not in text


def test_a_clean_change_says_so_in_terms_of_what_passed_before() -> None:
    text = _render(_payload())
    assert "Nothing that passed before fails now" in text


def test_broken_tests_are_named_not_counted() -> None:
    text = _render(_payload(newly_failing=["tests/a.py::test_one"]))
    assert "tests/a.py::test_one" in text


def test_pre_existing_failures_are_not_blamed_on_this_change() -> None:
    text = _render(_payload(still_failing=["tests/old.py::red"]))
    assert "already failing before this change" in text


def test_uncommitted_files_the_agent_never_saw_are_warned_about() -> None:
    """The worktree was cut from HEAD; this patch is about to land on top of
    edits the agent could not see (ADR-025)."""
    text = _render(_payload(dirty_paths=["module.py"]))
    assert "never saw" in text
    assert "module.py" in text


def test_after_a_resume_an_empty_dirty_list_says_it_was_not_rechecked() -> None:
    """AP-20: the list describes the moment the worktree was cut."""
    text = _render(_payload(dirty_paths=[], dirty_checked=False))
    assert "not re-checked" in text


def test_the_agents_cost_is_never_shown_as_api_spend() -> None:
    """Two currencies in one number is AP-20 in a cost line."""
    text = _render(_payload(agent_cost_equivalent=0.1734, agent_turns=6))
    assert "equivalent" in text
    assert "quota" in text


def test_the_gate_says_what_approving_will_do() -> None:
    assert "into your working tree, unstaged" in _render(_payload())


# --------------------------------------------------------------------------
# The write.
# --------------------------------------------------------------------------


def _values(tmp_path: Path, decision: GateDecision, patch_text: str) -> dict[str, Any]:
    patch_file = tmp_path / "p.diff"
    patch_file.write_text(patch_text, encoding="utf-8")
    return {
        "change_gate": GateOutcome(decision=decision),
        "changes": ChangeSet(
            files=["module.py"],
            stat=" module.py | 2 +-",
            patch=ArtifactRef(sha="abc", path=str(patch_file), kind="diff"),
            captured=True,
        ),
        "workspace": WorkspaceRef(path="/w", branch="dynaflows/chg-1", base="deadbeef"),
    }


def test_a_rejected_gate_applies_nothing_and_says_where_the_work_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli._finish_change(_values(tmp_path, GateDecision.REJECT, "diff"), tmp_path)
    out = capsys.readouterr().out

    assert "Nothing was applied" in out
    assert "dynaflows/chg-1" in out


def test_an_empty_change_is_not_reported_as_applied(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = _values(tmp_path, GateDecision.APPROVE, "diff")
    values["changes"] = ChangeSet(files=[], captured=True)
    cli._finish_change(values, tmp_path)

    assert "no diff" in capsys.readouterr().out


def test_a_missing_patch_is_a_loud_failure_not_a_silent_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = _values(tmp_path, GateDecision.APPROVE, "diff")
    values["changes"] = ChangeSet(files=["module.py"], captured=True, patch=None)
    cli._finish_change(values, tmp_path)

    assert "no patch was recorded" in capsys.readouterr().out


# --------------------------------------------------------------------------
# G2 must not describe a fan-out the WRITE pipeline cannot perform.
# --------------------------------------------------------------------------


def _plan_payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gate": "plan",
        "pipeline": "write",
        "rationale": "one change",
        "plan_hash": "abc123",
        "estimated_tokens": 10_000,
        "context_budget": 21_000,
        "over_budget": [],
        "tasks": [
            {
                "task_id": "section-map-budget",
                "capability": "implement",
                "objective": "cap the catalogue",
                "inputs": ["src/dynaflows/playbook/repository.py"],
                "anchors": ["ADR-009", "AP-20"],
                "context_tokens": 10_000,
            }
        ],
    }
    base.update(over)
    return base


def _render_plan(payload: dict[str, Any]) -> str:
    buffer = Console(record=True, width=120)
    cli._render_plan_gate(payload, out=buffer)
    return buffer.export_text()


def test_a_write_plan_is_not_described_as_parallel_workers() -> None:
    """The first live `change` run showed four workers, 68,806 context tokens
    and a worker-budget warning -- for a pipeline with no worker node."""
    text = _render_plan(_plan_payload())

    assert "parallel worker" not in text
    assert "1 change" in text
    assert "isolated worktree" in text


def test_a_write_plan_says_how_many_playbook_sections_the_agent_will_see() -> None:
    """The planner's one irreplaceable contribution here, so the gate names it."""
    assert "2 playbook section(s)" in _render_plan(_plan_payload())


def test_a_write_plan_with_no_anchors_says_so_rather_than_saying_nothing() -> None:
    """AP-20: 'the planner chose none' and 'the planner was not asked' are two
    facts, and the first one means the agent arrives without the rules."""
    payload = _plan_payload()
    payload["tasks"][0]["anchors"] = []
    text = _render_plan(payload)

    assert "no playbook sections were selected" in text


def test_a_read_plan_still_describes_its_fan_out() -> None:
    payload = _plan_payload(pipeline="read")
    payload["tasks"][0]["capability"] = "analyse"
    text = _render_plan(payload)

    assert "1 parallel worker(s)" in text
    assert "isolated worktree" not in text


def test_a_read_plan_still_warns_about_over_budget_tasks() -> None:
    payload = _plan_payload(pipeline="read", over_budget=["section-map-budget"])
    text = _render_plan(payload)

    assert "worker" in text
    assert "budget" in text


# --------------------------------------------------------------------------
# A cut-off change looks exactly like a finished one.
# --------------------------------------------------------------------------


def test_an_unfinished_change_says_so_above_the_diffstat() -> None:
    """Live, 2026-09-22: the agent hit a spend limit at turn 80 having done
    most of the work, and its final message was an error. The gate showed a
    diffstat and a green verdict -- both true -- and never said the run was
    cut off. A partial change compiles and passes the suite, which is why the
    verdict cannot carry this and a separate field must."""
    text = _render(
        _payload(finished=False, agent_stopped_because="You've hit your monthly spend limit")
    )

    assert "PARTIAL" in text
    assert "monthly spend limit" in text
    position = text.index("PARTIAL")
    assert position < text.index("file(s) changed"), "the warning must come first"


def test_a_finished_change_carries_no_partial_warning() -> None:
    assert "PARTIAL" not in _render(_payload(finished=True))


def test_a_payload_without_the_field_does_not_cry_wolf() -> None:
    """Older checkpoints have no `finished` key. Defaulting to "unfinished"
    would fire the loudest warning on every resumed run (playbook 5.2,
    Pattern 5: a gate that fires on ordinary work gets ignored)."""
    payload = _payload()
    payload.pop("finished", None)
    assert "PARTIAL" not in _render(payload)
