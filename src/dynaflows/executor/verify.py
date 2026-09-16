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
from pathlib import Path

from dynaflows.contracts.state import SuiteRun, Verdict
from dynaflows.executor.workspace import Workspace

__all__ = ["compare", "prepare", "run_tests"]

# `SuiteRun` and `Verdict` are the contract types (contracts/state.py), not
# copies of them. They are checkpointed, so there can be exactly one shape for
# each: a live dataclass here and a serialisable twin there would be two
# hand-written lists of the same fields, and this project has already paid for
# that mistake once (`merge_cost` and `_merge_delta` drifting apart).

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
            command=list(command),
            exit_code=-1,
            failing=[],
            parsed=False,
            duration_s=time.monotonic() - started,
            timed_out=True,
            tail=_tail(expired.stdout.decode() if isinstance(expired.stdout, bytes) else ""),
        )
    except OSError as error:
        # The command does not exist. A missing `uv` is a configuration
        # problem, not a red suite, and must not read as one.
        return SuiteRun(
            command=list(command),
            exit_code=-1,
            failing=[],
            parsed=False,
            duration_s=time.monotonic() - started,
            timed_out=False,
            tail=str(error),
        )

    output = completed.stdout + completed.stderr
    return SuiteRun(
        command=list(command),
        exit_code=completed.returncode,
        failing=sorted(set(_OUTCOME.findall(output))),
        parsed=completed.returncode in _TRUSTWORTHY_EXITS,
        duration_s=time.monotonic() - started,
        timed_out=False,
        tail=_tail(output),
    )


def compare(before: SuiteRun, after: SuiteRun) -> Verdict:
    """The three facts, computed once, in the shape state will hold.

    Computed here rather than as properties on `Verdict` because the verdict
    is checkpointed and read back by the gate and the CLI: a property would
    recompute from `before`/`after` that state does not carry, and carrying
    them twice to keep the property working is how the payload-in-state rule
    gets broken by accident (ADR-008).
    """
    comparable = before.parsed and after.parsed
    if not comparable:
        # Every set empty AND comparable False. An empty `newly_failing` from
        # a run that collected nothing is indistinguishable from a clean
        # change unless the caller checks `comparable`, so `clean` checks it.
        return Verdict(comparable=False)
    was, now = set(before.failing), set(after.failing)
    return Verdict(
        comparable=True,
        newly_failing=sorted(now - was),
        newly_passing=sorted(was - now),
        still_failing=sorted(now & was),
    )
