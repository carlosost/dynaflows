"""Running the project's own tests as the oracle, before and after. ADR-025.

Two decisions carry this module, and both were reached by being wrong first.

**The venv cannot be shared with the worktree.** The obvious optimisation is
to symlink the project's `.venv` into the worktree, because `uv sync` per run
sounds slow. It is also silently, completely wrong. An editable install puts
an absolute path in a `.pth` file:

    .venv/lib/python3.14/site-packages/dynaflows.pth
      -> /Users/.../Projects/dynaflows/src

Tests run in the worktree against that venv import the package from the
ORIGINAL source tree. They exercise the user's code, not the agent's changes;
they pass; and the verdict means nothing. No error, no warning, no way to tell
from the output. So the worktree gets its own environment, `uv`'s cache makes
that cheap, and the saving was never worth a green light that does not refer
to the change it claims to be about.

**Counting failures is not comparing them.** A suite that was already red
before the agent touched it stays red afterwards, and "3 failures" before and
"3 failures" after is the same number describing possibly disjoint sets of
tests. So the comparison is over SETS of test ids, and it yields three facts
that get three fields -- newly failing, newly passing, still failing -- rather
than one number that hides two of them (AP-20).

**What this module refuses to guess.** If the test ids cannot be read -- a
collection error, an internal crash, a command that is not pytest -- then
`parsed` is False and the comparison is not offered at all. An empty
`newly_failing` set from a run that never collected a test looks exactly like
a clean change, and that is the single most dangerous output this module could
produce.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from dynaflows.executor.workspace import Workspace

__all__ = ["SuiteRun", "Verdict", "compare", "prepare", "run_tests"]

# pytest's short summary, which `-q` still prints. Matched rather than the
# progress line because a node id is a stable, comparable identity and a dot
# is not. ERROR covers collection failures, which are failures of the change
# just as much as an assertion is.
_OUTCOME = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)

# pytest: 0 = all passed, 1 = tests failed. Every other code (2 interrupted,
# 3 internal error, 4 usage error, 5 nothing collected) means the run did not
# produce a trustworthy set of results, whatever it printed.
_TRUSTWORTHY_EXITS = frozenset({0, 1})

DEFAULT_COMMAND = ("uv", "run", "pytest", "-q", "--tb=no")
DEFAULT_TIMEOUT_S = 900
_TAIL_LINES = 40


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """One execution of the suite, with its identity-bearing result.

    Not `TestRun`: pytest collects any class whose name starts with `Test`,
    fails to instantiate this one, and emits a PytestCollectionWarning on
    every run of the suite. A warning that fires on correct work is a warning
    people learn to scroll past, which is the same failure mode as a gate
    that fires on ordinary work (playbook 5.2, Pattern 5) -- just quieter.
    """

    command: tuple[str, ...]
    exit_code: int
    failing: frozenset[str]
    # Whether `failing` is a real answer. False after a crash, a usage error,
    # or an empty collection -- cases where an empty set would otherwise read
    # as "nothing is broken".
    parsed: bool
    duration_s: float
    timed_out: bool
    tail: str

    @property
    def green(self) -> bool:
        return self.parsed and self.exit_code == 0


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the change did to the suite, or why that cannot be said."""

    before: SuiteRun
    after: SuiteRun

    @property
    def comparable(self) -> bool:
        """Both runs produced a trustworthy set of test ids.

        Read this before reading anything else. The three sets below are all
        empty when it is False, and empty means 'unknown' there, not 'clean'.
        """
        return self.before.parsed and self.after.parsed

    @property
    def newly_failing(self) -> frozenset[str]:
        """Broken by the change. The only set that should block a merge."""
        if not self.comparable:
            return frozenset()
        return self.after.failing - self.before.failing

    @property
    def newly_passing(self) -> frozenset[str]:
        """Fixed by the change -- which for a bug fix is the point of it."""
        if not self.comparable:
            return frozenset()
        return self.before.failing - self.after.failing

    @property
    def still_failing(self) -> frozenset[str]:
        """Red before, red after. Not this change's doing, and not hidden."""
        if not self.comparable:
            return frozenset()
        return self.after.failing & self.before.failing

    @property
    def clean(self) -> bool:
        """Nothing that passed before fails now.

        Deliberately NOT 'the suite is green': a project whose suite was
        already red would then be unable to accept any change at all, and the
        gate would be one people route around.
        """
        return self.comparable and not self.newly_failing


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-_TAIL_LINES:])


def prepare(workspace: Workspace, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> SuiteRun:
    """Give the worktree its own environment.

    Returns a `SuiteRun` rather than raising so that a failed sync is reported
    through the same shape as a failed suite: the caller has one thing to
    render at the gate, and a dependency resolution failure is a legitimate
    outcome of a change that edited `pyproject.toml`.
    """
    return _execute(workspace.path, ("uv", "sync", "--quiet"), timeout_s=timeout_s)


def run_tests(
    workspace: Workspace,
    command: tuple[str, ...] = DEFAULT_COMMAND,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> SuiteRun:
    """Run the suite inside the worktree."""
    return _execute(workspace.path, command, timeout_s=timeout_s)


def _execute(cwd: Path, command: tuple[str, ...], *, timeout_s: int) -> SuiteRun:
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as expired:
        return SuiteRun(
            command=command,
            exit_code=-1,
            failing=frozenset(),
            parsed=False,
            duration_s=time.monotonic() - started,
            timed_out=True,
            tail=_tail(expired.stdout.decode() if isinstance(expired.stdout, bytes) else ""),
        )
    except OSError as error:
        # The command does not exist. A missing `uv` is a configuration
        # problem, not a red suite, and must not read as one.
        return SuiteRun(
            command=command,
            exit_code=-1,
            failing=frozenset(),
            parsed=False,
            duration_s=time.monotonic() - started,
            timed_out=False,
            tail=str(error),
        )

    output = completed.stdout + completed.stderr
    return SuiteRun(
        command=command,
        exit_code=completed.returncode,
        failing=frozenset(_OUTCOME.findall(output)),
        parsed=completed.returncode in _TRUSTWORTHY_EXITS,
        duration_s=time.monotonic() - started,
        timed_out=False,
        tail=_tail(output),
    )


def compare(before: SuiteRun, after: SuiteRun) -> Verdict:
    return Verdict(before=before, after=after)
