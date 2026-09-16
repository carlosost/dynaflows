"""The test oracle: what a change did to the suite, or why that cannot be said.

The subprocess half is exercised with real processes rather than a patched
`subprocess.run`, because every claim here is about how a process behaves --
that a missing binary raises OSError rather than returning a code, that a
timeout loses the output, that pytest's exit 5 means nothing was collected.
A patched runner would assert we handle the return values we already chose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dynaflows.contracts.state import SuiteRun
from dynaflows.executor import verify

pytestmark = pytest.mark.deterministic


def _run(
    *,
    failing: set[str],
    parsed: bool = True,
    exit_code: int = 1,
) -> SuiteRun:
    return SuiteRun(
        command=["fake"],
        exit_code=exit_code,
        failing=sorted(failing),
        parsed=parsed,
        duration_s=0.0,
        timed_out=False,
        tail="",
    )


# --------------------------------------------------------------------------
# Comparison. The reason the oracle is a set and not a count.
# --------------------------------------------------------------------------


def test_the_same_number_of_failures_can_be_a_different_set() -> None:
    """Three before and three after, and the change broke one and fixed one.

    A count-based oracle reports 'no change' here. It is the exact case this
    module exists for.
    """
    verdict = verify.compare(
        before=_run(failing={"a", "b", "c"}),
        after=_run(failing={"a", "b", "d"}),
    )
    assert verdict.newly_failing == ["d"]
    assert verdict.newly_passing == ["c"]
    assert verdict.still_failing == ["a", "b"]
    assert verdict.clean is False


def test_a_suite_that_was_already_red_does_not_block_a_clean_change() -> None:
    """`clean` is 'nothing that passed now fails', not 'the suite is green'.

    A project with pre-existing failures would otherwise be unable to accept
    any change at all, and a gate nobody can pass is a gate people route
    around (playbook 5.2, Pattern 5).
    """
    verdict = verify.compare(before=_run(failing={"a", "b"}), after=_run(failing={"a", "b"}))
    assert verdict.clean is True
    assert verdict.newly_failing == []
    assert verdict.still_failing == ["a", "b"]


def test_a_fix_shows_up_as_newly_passing() -> None:
    after = _run(failing=set(), exit_code=0)
    verdict = verify.compare(before=_run(failing={"test_the_bug"}), after=after)
    assert verdict.newly_passing == ["test_the_bug"]
    assert verdict.clean is True
    # `green` is a fact about the RUN, not about the comparison. The verdict
    # deliberately does not carry the two runs it was computed from: state
    # would then hold them twice, which is the payload-in-state rule (ADR-008)
    # broken by a convenience property.
    assert after.green is True


def test_an_unreadable_run_is_not_a_clean_one() -> None:
    """The most dangerous output this module could produce.

    An empty `newly_failing` from a run that never collected a test is
    indistinguishable from a clean change unless `comparable` is checked, so
    `clean` checks it.
    """
    verdict = verify.compare(before=_run(failing=set()), after=_run(failing=set(), parsed=False))
    assert verdict.comparable is False
    assert verdict.clean is False
    assert verdict.newly_failing == []
    assert verdict.newly_passing == []
    assert verdict.still_failing == []


def test_an_unreadable_baseline_is_not_comparable_either() -> None:
    verdict = verify.compare(before=_run(failing=set(), parsed=False), after=_run(failing={"a"}))
    assert verdict.comparable is False
    assert verdict.clean is False


def test_a_run_that_did_not_parse_is_never_green() -> None:
    assert _run(failing=set(), parsed=False, exit_code=0).green is False


# --------------------------------------------------------------------------
# Execution, against real processes.
# --------------------------------------------------------------------------


class _Space:
    """The one attribute `_execute` reads. A real Workspace needs a git repo,
    and none of these tests are about git."""

    def __init__(self, path: Path) -> None:
        self.path = path


def test_a_passing_command_parses_as_green(tmp_path: Path) -> None:
    result = verify.run_tests(
        _Space(tmp_path),  # type: ignore[arg-type]
        (sys.executable, "-c", "print('1 passed')"),
    )
    assert result.exit_code == 0
    assert result.parsed is True
    assert result.green is True
    assert result.failing == []


def test_failed_lines_are_read_as_test_identities(tmp_path: Path) -> None:
    script = (
        "print('short test summary info')\n"
        "print('FAILED tests/unit/test_a.py::test_one - AssertionError')\n"
        "print('FAILED tests/unit/test_b.py::test_two')\n"
        "print('ERROR tests/unit/test_c.py')\n"
        "raise SystemExit(1)\n"
    )
    result = verify.run_tests(_Space(tmp_path), (sys.executable, "-c", script))  # type: ignore[arg-type]

    assert result.parsed is True
    assert result.failing == [
        "tests/unit/test_a.py::test_one",
        "tests/unit/test_b.py::test_two",
        "tests/unit/test_c.py",
    ]
    assert result.green is False


def test_nothing_collected_is_not_a_clean_run(tmp_path: Path) -> None:
    """pytest exit 5. An empty failure set here means 'no tests ran'."""
    result = verify.run_tests(
        _Space(tmp_path),  # type: ignore[arg-type]
        (sys.executable, "-c", "raise SystemExit(5)"),
    )
    assert result.exit_code == 5
    assert result.parsed is False
    assert result.green is False


def test_an_internal_crash_is_not_a_clean_run(tmp_path: Path) -> None:
    result = verify.run_tests(
        _Space(tmp_path),  # type: ignore[arg-type]
        (sys.executable, "-c", "raise SystemExit(3)"),
    )
    assert result.parsed is False


def test_a_missing_command_is_a_configuration_problem_not_a_red_suite(
    tmp_path: Path,
) -> None:
    result = verify.run_tests(
        _Space(tmp_path),  # type: ignore[arg-type]
        ("definitely-not-a-real-binary-9f3a",),
    )
    assert result.parsed is False
    assert result.timed_out is False
    assert result.failing == []
    assert result.tail


def test_a_hanging_suite_times_out_rather_than_hanging_the_run(tmp_path: Path) -> None:
    result = verify.run_tests(
        _Space(tmp_path),  # type: ignore[arg-type]
        (sys.executable, "-c", "import time; time.sleep(30)"),
        timeout_s=1,
    )
    assert result.timed_out is True
    assert result.parsed is False
    assert result.green is False


def test_stderr_is_read_too_because_pytest_crashes_there(tmp_path: Path) -> None:
    script = "import sys; print('FAILED tests/x.py::t', file=sys.stderr); raise SystemExit(1)"
    result = verify.run_tests(_Space(tmp_path), (sys.executable, "-c", script))  # type: ignore[arg-type]
    assert result.failing == ["tests/x.py::t"]


def test_the_command_runs_in_the_worktree_not_the_users_tree(tmp_path: Path) -> None:
    """The whole point. A test run in the wrong directory tests the wrong code."""
    work = tmp_path / "work"
    work.mkdir()
    result = verify.run_tests(
        _Space(work),  # type: ignore[arg-type]
        (sys.executable, "-c", "import os; print(os.getcwd())"),
    )
    assert str(work.resolve()) in result.tail


def test_the_tail_is_bounded_so_a_gate_stays_readable(tmp_path: Path) -> None:
    script = "for i in range(500): print(f'line {i}')"
    result = verify.run_tests(_Space(tmp_path), (sys.executable, "-c", script))  # type: ignore[arg-type]
    assert len(result.tail.splitlines()) == verify._TAIL_LINES
    assert "line 499" in result.tail
    assert "line 100" not in result.tail
