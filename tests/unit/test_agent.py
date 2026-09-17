"""The seam to the coding agent.

`build_command` gets the most attention because it is the part that is pure,
and because every mistake in it is silent: a flag in the wrong order, a brief
read as an option, a `--bare` that quietly moves the bill from a subscription
to a token meter. The subprocess half is exercised against a real fake binary
rather than a patched `subprocess.run`, so the argv, the working directory
and the exit code are the real ones.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from dynaflows.executor.agent import (
    AgentOptions,
    ClaudeCodeAgent,
    build_command,
    session_id_for,
)

pytestmark = pytest.mark.deterministic

OPTIONS = AgentOptions(permission_mode="acceptEdits")


class _Space:
    def __init__(self, path: Path, run_id: str = "r1") -> None:
        self.path = path
        self.run_id = run_id


def _fake_claude(directory: Path, *, body: str) -> str:
    """A stand-in binary, so the argv and cwd under test are the real ones."""
    script = directory / "fake-claude"
    script.write_text(f"#!{sys.executable}\nimport sys, json, os\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


# --------------------------------------------------------------------------
# The argv.
# --------------------------------------------------------------------------


def test_the_brief_is_the_last_argument_so_a_leading_hyphen_is_not_a_flag() -> None:
    command = build_command("--help me understand this", OPTIONS, "s1")
    assert command[-1] == "--help me understand this"


def test_the_brief_is_never_shell_interpreted() -> None:
    """argv, not a shell string. Quotes and backticks are just characters."""
    brief = 'fix `rm -rf /` and the "quoted" $PATH bug\nsecond line'
    assert build_command(brief, OPTIONS, "s1")[-1] == brief


def test_bare_is_never_passed_because_it_moves_the_bill() -> None:
    """`--bare` reads auth strictly from ANTHROPIC_API_KEY, never OAuth.

    Passing it would silently switch a change run from the user's
    subscription to pay-per-token, and would also skip CLAUDE.md -- the
    project conventions a change is supposed to follow.
    """
    assert "--bare" not in build_command("b", OPTIONS, "s1")


def test_the_agent_does_not_cut_its_own_worktree() -> None:
    """Ours is already cut, lives under .dynaflows/ where ADR-016 can see it,
    and is what `verify` and the G3 diff both read."""
    command = build_command("b", OPTIONS, "s1")
    assert "--worktree" not in command
    assert "-w" not in command


def test_print_and_json_are_always_present() -> None:
    command = build_command("b", OPTIONS, "s1")
    assert "--print" in command
    assert command[command.index("--output-format") + 1] == "json"


def test_the_session_id_is_set_on_a_first_run_and_resumed_after() -> None:
    fresh = build_command("b", OPTIONS, "s1")
    assert fresh[fresh.index("--session-id") + 1] == "s1"
    assert "--resume" not in fresh

    again = build_command("b", OPTIONS, "s1", resume=True)
    assert again[again.index("--resume") + 1] == "s1"
    assert "--session-id" not in again


def test_the_session_id_is_derived_from_the_run_so_a_resume_finds_it() -> None:
    assert session_id_for("q7") == session_id_for("q7")
    assert session_id_for("q7") != session_id_for("q8")
    import uuid

    uuid.UUID(session_id_for("q7"))  # --session-id requires a valid UUID


def test_optional_flags_are_absent_rather_than_empty() -> None:
    command = build_command("b", OPTIONS, "s1")
    for flag in ("--model", "--allowedTools", "--disallowedTools", "--max-budget-usd"):
        assert flag not in command


def test_every_option_reaches_the_command_line() -> None:
    options = AgentOptions(
        permission_mode="acceptEdits",
        model="sonnet",
        allowed_tools=("Read", "Edit"),
        disallowed_tools=("WebFetch",),
        max_budget_usd=2.5,
        append_system_prompt="follow the playbook",
    )
    command = build_command("b", options, "s1")
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--allowedTools") + 1 : command.index("--allowedTools") + 3] == [
        "Read",
        "Edit",
    ]
    assert command[command.index("--disallowedTools") + 1] == "WebFetch"
    assert command[command.index("--max-budget-usd") + 1] == "2.5"
    assert command[command.index("--append-system-prompt") + 1] == "follow the playbook"


def test_the_permission_mode_has_no_default_and_must_be_chosen() -> None:
    """A mode that prompts in a non-interactive run hangs until the timeout,
    so there is no safe value to assume."""
    with pytest.raises(TypeError):
        AgentOptions()  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Running it.
# --------------------------------------------------------------------------


def test_a_successful_run_is_parsed(tmp_path: Path) -> None:
    binary = _fake_claude(
        tmp_path,
        body=(
            "print(json.dumps({'type':'result','is_error':False,'result':'edited two files',"
            "'session_id':'abc','total_cost_usd':0.42,'num_turns':7}))"
        ),
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.parsed is True
    assert run.usable is True
    assert run.result_text == "edited two files"
    assert run.cost_usd == 0.42
    assert run.num_turns == 7
    assert run.session_id == "abc"


def test_the_agents_own_error_flag_is_not_the_exit_code(tmp_path: Path) -> None:
    """An agent can exit 0 having reported it could not do the task."""
    binary = _fake_claude(
        tmp_path,
        body="print(json.dumps({'is_error':True,'result':'I could not find the module'}))",
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.exit_code == 0
    assert run.reported_error is True
    assert run.usable is False


def test_unparseable_output_keeps_the_run_and_says_it_is_unparsed(tmp_path: Path) -> None:
    """A completed, paid-for agent run must not be lost to a parsing detail."""
    binary = _fake_claude(tmp_path, body="print('not json at all')")
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.parsed is False
    assert run.cost_usd is None
    assert "not json at all" in run.raw


def test_a_trailing_result_object_is_found_among_preamble_lines(tmp_path: Path) -> None:
    binary = _fake_claude(
        tmp_path,
        body=(
            "print('warning: something')\n"
            "print(json.dumps({'is_error':False,'result':'done','total_cost_usd':1.0}))"
        ),
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.parsed is True
    assert run.result_text == "done"


def test_a_missing_binary_is_reported_not_raised(tmp_path: Path) -> None:
    run = ClaudeCodeAgent(
        AgentOptions(permission_mode="acceptEdits", binary="no-such-agent-7b2")
    ).run(_Space(tmp_path), "fix it")  # type: ignore[arg-type]
    assert run.usable is False
    assert run.timed_out is False
    assert "could not start" in run.raw


def test_a_hanging_agent_times_out(tmp_path: Path) -> None:
    """The failure a wrong --permission-mode produces: a prompt nobody answers."""
    binary = _fake_claude(tmp_path, body="import time; time.sleep(30)")
    run = ClaudeCodeAgent(
        AgentOptions(permission_mode="manual", binary=binary, timeout_s=1)
    ).run(_Space(tmp_path), "fix it")  # type: ignore[arg-type]
    assert run.timed_out is True
    assert run.usable is False


def test_the_agent_runs_inside_the_worktree(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    binary = _fake_claude(tmp_path, body="print(json.dumps({'result': os.getcwd()}))")
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(work), "fix it"  # type: ignore[arg-type]
    )
    assert os.path.realpath(run.result_text) == os.path.realpath(str(work))


def test_the_brief_reaches_the_agent_intact(tmp_path: Path) -> None:
    binary = _fake_claude(tmp_path, body="print(json.dumps({'result': sys.argv[-1]}))")
    brief = 'multi\nline "brief" with `backticks`'
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), brief  # type: ignore[arg-type]
    )
    assert run.result_text == brief


# --------------------------------------------------------------------------
# Unavailable is not failed. Found on the first live probe.
# --------------------------------------------------------------------------


def test_a_not_logged_in_agent_is_unavailable_not_a_failed_change(tmp_path: Path) -> None:
    """The real payload from the first live probe, shape for shape.

    Note `subtype` says "success" while `is_error` is true. Anything keyed on
    `subtype` would have called this a successful change and sent the user to
    read a diff that does not exist.
    """
    binary = _fake_claude(
        tmp_path,
        body=(
            "print(json.dumps({'type':'result','subtype':'success','is_error':True,"
            "'result':'Not logged in \\u00b7 Please run /login','total_cost_usd':0,"
            "'num_turns':1,'terminal_reason':'api_error'}))\n"
            "sys.exit(1)"
        ),
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.parsed is True
    assert run.usable is False
    assert run.unavailable is True


def test_a_missing_binary_is_also_unavailable(tmp_path: Path) -> None:
    run = ClaudeCodeAgent(
        AgentOptions(permission_mode="acceptEdits", binary="no-such-agent-7b2")
    ).run(_Space(tmp_path), "fix it")  # type: ignore[arg-type]
    assert run.unavailable is True


def test_an_agent_that_tried_and_failed_is_not_unavailable(tmp_path: Path) -> None:
    """The distinction that matters: this one DID work, and could not do it.

    The user should read what it said, not be told to log in.
    """
    binary = _fake_claude(
        tmp_path,
        body="print(json.dumps({'is_error':True,'result':'the module does not exist'}))",
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.usable is False
    assert run.unavailable is False


def test_a_successful_run_is_neither_failed_nor_unavailable(tmp_path: Path) -> None:
    binary = _fake_claude(tmp_path, body="print(json.dumps({'is_error':False,'result':'done'}))")
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "fix it"  # type: ignore[arg-type]
    )
    assert run.usable is True
    assert run.unavailable is False


def test_the_measured_mode_is_the_deterministic_one() -> None:
    """Measured 2026-09-16, not chosen. See the constant's comment.

    Asserted so that a later "this looks unsafe, use acceptEdits" edit has to
    confront the measurement: `acceptEdits` exits 0 with an empty diff when it
    meets a command it will not run, and `auto` decides via a classifier that
    can answer differently on two identical runs.
    """
    from dynaflows.executor.agent import MEASURED_PERMISSION_MODE

    assert MEASURED_PERMISSION_MODE == "bypassPermissions"


def test_context_reaches_the_system_prompt_and_not_the_brief(tmp_path: Path) -> None:
    """ADR-025: the brief is the exact string a human approved at G1."""
    binary = _fake_claude(
        tmp_path, body="print(json.dumps({'result': ' '.join(sys.argv[1:])}))"
    )
    run = ClaudeCodeAgent(AgentOptions(permission_mode="acceptEdits", binary=binary)).run(
        _Space(tmp_path), "THE BRIEF", context="these files matter"  # type: ignore[arg-type]
    )
    assert "--append-system-prompt these files matter" in run.result_text
    assert run.result_text.endswith("THE BRIEF")


def test_context_is_appended_to_a_configured_system_prompt_not_replacing_it(
    tmp_path: Path,
) -> None:
    binary = _fake_claude(
        tmp_path, body="print(json.dumps({'result': ' '.join(sys.argv[1:])}))"
    )
    options = AgentOptions(
        permission_mode="acceptEdits", binary=binary, append_system_prompt="follow the playbook"
    )
    run = ClaudeCodeAgent(options).run(_Space(tmp_path), "b", context="and these files")  # type: ignore[arg-type]

    assert "follow the playbook" in run.result_text
    assert "and these files" in run.result_text


def test_per_run_context_does_not_mutate_the_shared_options(tmp_path: Path) -> None:
    """The agent is constructed once and used for every task in a run. An
    options object edited in place would accumulate every previous task's
    context."""
    binary = _fake_claude(tmp_path, body="print(json.dumps({'result': 'ok'}))")
    options = AgentOptions(permission_mode="acceptEdits", binary=binary)
    agent = ClaudeCodeAgent(options)

    agent.run(_Space(tmp_path), "b", context="first")  # type: ignore[arg-type]
    agent.run(_Space(tmp_path), "b", context="second")  # type: ignore[arg-type]

    assert options.append_system_prompt is None
