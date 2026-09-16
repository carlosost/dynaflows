"""Invoking the coding agent on an approved brief. ADR-025.

This is the seam. Everything either side of it is ours; the agent is a
separate process with its own model, its own permissions and its own planner,
and the only things that cross are a brief going in and a diff coming out.

**Why the CLI and not the SDK.** `claude-agent-sdk` would give typed events
and streaming, at the cost of a hard Python dependency and a second way for
this project to hold an agent's credentials. The CLI needs neither: it is
already installed and already authenticated for the user who will run it, and
the contract we need from it -- run this brief in this directory, tell me what
happened -- is three flags wide. The Protocol below is what makes the choice
reversible; an SDK adapter satisfies the same shape.

**Billing, which is not a detail.** `claude --bare` reads auth strictly from
ANTHROPIC_API_KEY and never from OAuth or the keychain. So `--bare` means
pay-per-token and its absence means the user's own subscription. This module
does NOT pass `--bare`: a change run should cost what the user already pays
for, and the only metered calls in the shape are the enhancer at G1 and the
planner at G2. `--bare` would also skip CLAUDE.md discovery, which is exactly
the project convention a change ought to follow.

**What is deliberately not decided here.** `--permission-mode` takes six
values and the help text documents none of their non-interactive behaviour.
A mode that prompts in a non-interactive run does not error -- it HANGS, and
a hung run looks like a slow one until the timeout fires. So the mode is a
field with no confident default, set from a probe against a throwaway
worktree rather than from a guess (§4.5: a constant nobody measured is a
PLACEHOLDER, and this one is worse than most because being wrong costs the
whole timeout).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from dynaflows.executor.workspace import Workspace

__all__ = [
    "Agent",
    "AgentOptions",
    "AgentRun",
    "ClaudeCodeAgent",
    "build_command",
    "check_agent_auth",
    "MEASURED_PERMISSION_MODE",
    "session_id_for",
]

# Derived from the run id rather than random, so the same run always addresses
# the same agent session: `claude --resume <id>` then continues the work
# instead of starting a second one, and a resumed dynaflows run cannot orphan
# the session it already paid for.
_SESSION_NAMESPACE = uuid.UUID("6f1b6c6e-0a2f-4f1a-9a3e-0b5c9d2a7e41")

# Matched against the agent's own `result` text rather than an exit code,
# because a not-logged-in run reports `"subtype":"success"` alongside
# `"is_error":true`. Anything keyed on `subtype` would have read that as a
# successful change. Recorded because the trap is in the agent's contract,
# not in ours.
_NOT_AUTHENTICATED = re.compile(r"not logged in|please run /login|invalid api key", re.IGNORECASE)

# Measured 2026-09-16, two tasks per mode: one benign (`git rev-parse`, very
# likely auto-allowed) and one arbitrary (`python3 -c ...`, which no allowlist
# covers). Round one used only the benign task and therefore measured nothing.
#
#   mode                benign   arbitrary
#   acceptEdits         OK       SILENT NO-OP  -- exit 0, empty diff
#   auto                OK       OK
#   bypassPermissions   OK       OK
#   dontAsk             NO-OP    (not re-run)
#
# `acceptEdits` is disqualified, and not for the reason expected. It does not
# hang: it returns in seven seconds, exit 0, `is_error` false, having refused
# the command, with the refusal only in the result text -- "Command needs your
# approval". A run that reports success with an empty diff is worse than one
# that blocks, because a block is loud once the timeout fires.
#
# `auto` passed both, and is still not the choice: it routes through a
# classifier (`claude auto-mode`), so the same command can be allowed on one
# run and refused on the next. An intermittent silent no-op is the worst
# version of the defect above.
#
# `bypassPermissions` is deterministic, which is the property the executor
# needs. It is not where the safety comes from and this project should not
# pretend otherwise (ADR-025): the boundary is the worktree, the explicit
# `disallowed_tools`, and the human approving at G2 before anything runs.
# Choosing a mode that feels safer while silently skipping work would be
# theatre that costs correctness.
MEASURED_PERMISSION_MODE = "bypassPermissions"

DEFAULT_TIMEOUT_S = 1800
_TAIL_CHARS = 4000


@dataclass(frozen=True, slots=True)
class AgentOptions:
    """How the agent is invoked. Every field is a decision, not a knob."""

    # No default. See the module docstring: the wrong value hangs the run
    # until the timeout, which is the most expensive way to be wrong here.
    permission_mode: str
    model: str | None = None
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    # `--max-budget-usd` is documented as bounding "API calls". Under
    # subscription auth it may be a no-op; that is unmeasured, so a caller
    # setting it must not treat it as a guarantee.
    max_budget_usd: float | None = None
    append_system_prompt: str | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    binary: str = "claude"


@dataclass(frozen=True, slots=True)
class AgentRun:
    """What came back. Parsed where possible, kept raw where not.

    The JSON shape of `--output-format json` is the agent's contract, not
    ours, and it can change between versions of a tool this project does not
    release. So every parsed field is optional, `raw` always holds the tail,
    and `parsed` says which of the two the caller is looking at -- rather
    than a zero cost and an empty result being indistinguishable from a
    successful cheap run (AP-20).
    """

    session_id: str
    exit_code: int
    timed_out: bool
    duration_s: float
    raw: str
    parsed: bool = False
    result_text: str = ""
    cost_usd: float | None = None
    num_turns: int | None = None
    # The agent's own verdict on its own run, which is not the same as the
    # exit code and not the same as the tests passing.
    reported_error: bool = False
    fields: dict[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """It ran to completion and did not report failing at its own task.

        Says nothing about whether the change is correct -- that is what
        `verify` is for, and conflating the two would let an agent that
        cheerfully edited the wrong file report success.
        """
        return self.exit_code == 0 and not self.timed_out and not self.reported_error

    @property
    def unavailable(self) -> bool:
        """The agent could not be used at all, as opposed to trying and failing.

        Met on the first live probe: every permission mode returned in under
        two seconds at $0.0000 with `terminal_reason: api_error`, and the
        cause was `Not logged in`. Without this distinction the write pipeline
        would tell the user their change failed -- sending them to read a
        diff that does not exist -- when what it needed to say was "log in".
        The same shape as ADR-025's missing-binary case, and the same rule
        the gateway learned about 402s: a configuration problem wearing an
        error code is still a configuration problem.
        """
        return self.exit_code != 0 and (
            not self.parsed or _NOT_AUTHENTICATED.search(self.result_text) is not None
        )


class Agent(Protocol):
    """The seam. An SDK adapter or a recorded transcript satisfies this too."""

    def run(self, workspace: Workspace, brief: str) -> AgentRun: ...


def session_id_for(run_id: str) -> str:
    """A stable UUID for this run's agent session."""
    return str(uuid.uuid5(_SESSION_NAMESPACE, run_id))


def build_command(
    brief: str, options: AgentOptions, session: str, *, resume: bool = False
) -> list[str]:
    """The argv, as a pure function, because this is the part worth testing.

    argv rather than a shell string: the brief is user text containing quotes,
    backticks and newlines, and there is no shell here to interpret any of it.

    Note `--bare` is absent on purpose (see the module docstring) and so is
    `--worktree`: Claude Code can cut its own, but ours is already cut, lives
    under `.dynaflows/` where ADR-016 can see it, and is the thing `verify`
    and the G3 diff both read.
    """
    command = [options.binary, "--print", "--output-format", "json"]
    if resume:
        command += ["--resume", session]
    else:
        command += ["--session-id", session]
    command += ["--permission-mode", options.permission_mode]
    if options.model:
        command += ["--model", options.model]
    if options.allowed_tools:
        command += ["--allowedTools", *options.allowed_tools]
    if options.disallowed_tools:
        command += ["--disallowedTools", *options.disallowed_tools]
    if options.max_budget_usd is not None:
        command += ["--max-budget-usd", str(options.max_budget_usd)]
    if options.append_system_prompt:
        command += ["--append-system-prompt", options.append_system_prompt]
    # Last, so a brief beginning with a hyphen cannot be read as a flag.
    command.append(brief)
    return command


def _parse(payload: str) -> dict[str, object]:
    """Read the result object, or return nothing. Never raises.

    `--output-format json` emits one object, but a wrapper, a warning line or
    a future version emitting a list would all break a strict read -- and a
    crash here would lose a completed, paid-for agent run over a parsing
    detail.
    """
    text = payload.strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                loaded = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            return {}
    if isinstance(loaded, list):
        loaded = next(
            (item for item in reversed(loaded) if isinstance(item, dict)),
            {},
        )
    return loaded if isinstance(loaded, dict) else {}


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class ClaudeCodeAgent:
    """Runs `claude -p` with the worktree as its working directory."""

    def __init__(self, options: AgentOptions) -> None:
        self._options = options

    def run(self, workspace: Workspace, brief: str) -> AgentRun:
        session = session_id_for(workspace.run_id)
        command = build_command(brief, self._options, session)
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=str(workspace.path),
                capture_output=True,
                text=True,
                check=False,
                timeout=self._options.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return AgentRun(
                session_id=session,
                exit_code=-1,
                timed_out=True,
                duration_s=time.monotonic() - started,
                raw=f"timed out after {self._options.timeout_s}s",
            )
        except OSError as error:
            return AgentRun(
                session_id=session,
                exit_code=-1,
                timed_out=False,
                duration_s=time.monotonic() - started,
                raw=f"could not start {self._options.binary!r}: {error}",
            )

        payload = _parse(completed.stdout)
        return AgentRun(
            session_id=str(payload.get("session_id") or session),
            exit_code=completed.returncode,
            timed_out=False,
            duration_s=time.monotonic() - started,
            raw=(completed.stdout + completed.stderr)[-_TAIL_CHARS:],
            parsed=bool(payload),
            result_text=str(payload.get("result") or ""),
            cost_usd=_as_float(payload.get("total_cost_usd")),
            num_turns=_as_int(payload.get("num_turns")),
            reported_error=payload.get("is_error") is True,
            fields=payload,
        )


def check_agent_auth(binary: str = "claude", *, timeout_s: int = 30) -> tuple[bool, str]:
    """Is the coding agent installed and logged in? Returns (ok, what was found).

    Lives here rather than in `doctor.py` because ADR-025 bans importing
    `subprocess` outside this package, and the ban is worth more than the
    convenience of putting the check where the other checks are.

    Costs one trivial turn when logged in, and nothing at all when not: a
    not-logged-in agent answers in about 60ms having called no model. This
    exists because a change run that reaches the agent and discovers the
    problem there has already cut a worktree, run a baseline suite and spent
    two metered calls at G1 and G2 -- to arrive at "log in". **If you can
    check it early, do not discover it late.**
    """
    try:
        completed = subprocess.run(  # noqa: S603
            [binary, "--print", "--output-format", "json", "reply with OK"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"{binary} did not answer within {timeout_s}s"
    except OSError as error:
        return False, f"not installed or not on PATH: {error}"

    payload = _parse(completed.stdout)
    reported = str(payload.get("result") or "")
    if _NOT_AUTHENTICATED.search(reported):
        return False, f"installed but not logged in -- run `{binary}` and `/login`"
    if completed.returncode != 0 or payload.get("is_error") is True:
        return False, reported or (completed.stdout + completed.stderr)[-200:] or "unknown failure"
    return True, "installed and authenticated"
