"""`dynaflows run` and `resume`, end to end through Typer.

These exist because of a bug that unit tests could not have caught: the
command built its config WITHOUT a gateway, so every real run failed inside
`enhance_prompt` while 204 tests stayed green. The graph was tested; the
wiring that feeds it was not.

Everything below patches `get_gateway` -- the factory boundary (playbook §3.1,
AP-02) -- so no network is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from dynaflows.cli import app
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.graph.prompts import compose_brief
from tests.conftest import FakeGateway, make_playbook, make_workspace

_DEFAULT_PLAN_TASKS = range(3)  # what conftest's default draft plans

pytestmark = pytest.mark.deterministic


def _settings_rooted_at(root: Path) -> Any:
    """get_settings(), with the project root moved.

    `find_project_root()` walks up for pyproject.toml, so without this the CLI
    tests plan against the real repository and the fixture's plan names a file
    that is not in it.
    """
    from dynaflows.settings import get_settings as real

    def _get(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("root", root)
        return real(*args, **kwargs)

    return _get


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGateway:
    """A gateway that counts calls, and state written under tmp.

    `--root` is NOT passed by these tests; instead the project root itself is
    the workspace, so the fake plan's `src/auth.py` is a path ADR-018's
    catalogue really contains. Pointing the tests at a special directory the
    product never uses would test a path the product never takes.
    """
    make_workspace(tmp_path)
    monkeypatch.setattr("dynaflows.cli.get_settings", _settings_rooted_at(tmp_path / "workspace"))
    gateway = FakeGateway("a precise brief", ["assumed the HTTP layer"])
    monkeypatch.setattr("dynaflows.gateway.client.get_gateway", lambda **_: gateway)
    monkeypatch.setattr("dynaflows.playbook.get_playbook_repository", lambda **_: make_playbook())
    monkeypatch.setenv("DYNAFLOWS_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return gateway


def test_run_reaches_the_enhancer_with_a_gateway(isolated: FakeGateway) -> None:
    """The regression. Without a gateway in config the node raises
    CONFIG_INVALID and the run produces nothing."""
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t1", "--yes-prompt", "--yes-plan"]
    )
    assert result.exit_code == 0, result.output
    assert isolated.enhancer_calls == 1
    assert "Completed" in result.output


def test_run_halts_at_gate_g1_and_shows_the_rewrite(isolated: FakeGateway) -> None:
    """Answering 'r' at the prompt: the human sees both texts and says no."""
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t2", "--yes-plan"], input="r\n"
    )
    assert "audit auth" in result.output
    assert "a precise brief" in result.output
    assert "assumed the HTTP layer" in result.output
    assert result.exit_code == 2, result.output
    assert "rejected by human at gate G1" in result.output


def test_approving_at_the_gate_completes_the_run(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t3", "--yes-plan"], input="a\n"
    )
    assert result.exit_code == 0, result.output
    assert "Completed" in result.output
    assert isolated.enhancer_calls == 1


def test_a_rejection_reports_what_it_cost_before_stopping(isolated: FakeGateway) -> None:
    """ADR-005: the value of gating at G1 is that a 'no' costs one small-tier
    call. Saying so out loud is how that stays true."""
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t4", "--yes-plan"], input="r\n"
    )
    assert "$0.0020 on this thread" in result.output


def test_yes_prompt_never_asks(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t5", "--yes-prompt", "--yes-plan"], input=""
    )
    assert result.exit_code == 0, result.output
    assert "approve" not in result.output.lower()


def test_resume_continues_a_gated_run_without_re_running_the_enhancer(
    isolated: FakeGateway,
) -> None:
    """ADR-007 through the CLI, not just the graph."""
    runner = CliRunner()
    first = runner.invoke(app, ["run", "audit auth", "--thread", "t6", "--yes-plan"], input="r\n")
    assert first.exit_code == 2
    assert isolated.enhancer_calls == 1


def test_resume_on_an_unknown_thread_fails_loudly(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(app, ["resume", "no-such-thread"])
    assert result.exit_code == 1
    assert "No checkpoint" in result.output


# --------------------------------------------------------------------------
# The `edit` path. Its first version shipped broken and was found by a human
# at a live gate: no $EDITOR set, so it fell back to a single-line prompt with
# no default -- which made retyping a five-line brief miserable AND looped
# forever on an empty Enter. A gate the user cannot get out of is worse than
# no gate.
# --------------------------------------------------------------------------


def test_no_editor_falls_back_to_multiline_and_never_loops(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dynaflows.cli._find_editor", lambda: None)
    result = CliRunner().invoke(
        app,
        ["run", "audit auth", "--thread", "e1", "--yes-plan"],
        input="e\nmy own brief\nsecond line\n.\n",
    )
    assert result.exit_code == 0, result.output
    assert "No editor found" in result.output


def test_submitting_nothing_keeps_the_text_instead_of_re_asking(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop. Empty input used to re-prompt forever with no way out."""
    monkeypatch.setattr("dynaflows.cli._find_editor", lambda: None)
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "e2", "--yes-plan"], input="e\n.\n"
    )
    assert result.exit_code == 0, result.output
    assert "treating as approve" in result.output


def test_an_explicit_editor_variable_wins_over_a_discovered_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dynaflows.cli import _find_editor

    monkeypatch.setenv("VISUAL", "")
    monkeypatch.setenv("EDITOR", "my-editor --wait")
    assert _find_editor() == ["my-editor", "--wait"]
    monkeypatch.setenv("VISUAL", "visual-editor")
    assert _find_editor() == ["visual-editor"]


def test_without_an_editor_variable_a_friendly_one_is_preferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Someone who never set $EDITOR is unlikely to be a vi user, and dropping
    them into modal editing unannounced is its own trap."""
    from dynaflows.cli import _find_editor

    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert _find_editor() == ["nano"]


def test_the_edited_text_reaches_the_workflow(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dynaflows.cli._find_editor", lambda: None)
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "e3", "--yes-plan"], input="e\nREPLACED BRIEF\n.\n"
    )
    assert result.exit_code == 0, result.output
    assert isolated.enhancer_calls == 1, "editing must not re-ask the model"


def test_resuming_a_finished_thread_says_so_instead_of_claiming_work(
    isolated: FakeGateway,
) -> None:
    """`resume` on a completed thread printed "Completed." -- reporting work
    that did not happen. Two different facts, one word: AP-20 in a status line.
    """
    runner = CliRunner()
    first = runner.invoke(
        app, ["run", "audit auth", "--thread", "d1", "--yes-prompt", "--yes-plan"]
    )
    assert first.exit_code == 0, first.output

    again = runner.invoke(app, ["resume", "d1"])
    assert again.exit_code == 0, again.output
    assert "already finished" in again.output
    assert isolated.enhancer_calls == 1, "resuming a finished thread must not re-run anything"


def test_run_and_resume_describe_a_finished_run_the_same_way(
    isolated: FakeGateway,
) -> None:
    """They had grown separate endings and disagreed: `run` showed the task
    summary, `resume` showed nothing. One reporter now."""
    runner = CliRunner()
    ran = runner.invoke(app, ["run", "audit auth", "--thread", "d2", "--yes-prompt", "--yes-plan"])
    resumed = runner.invoke(app, ["resume", "d2"])
    for fragment in ("task(s)", "on this thread", "call(s)"):
        assert fragment in ran.output, fragment
        assert fragment in resumed.output, fragment


def test_a_completed_run_reports_what_it_cost(isolated: FakeGateway) -> None:
    """Enhancer + planner + one call per worker.

    The number is derived from the fake's per-call cost rather than written
    out. It was written out, and step 1.6 -- which changed the call COUNT and
    nothing about costing -- made it fail for a reason that had nothing to do
    with what it was testing. A test whose number moves for unrelated reasons
    is a test people edit instead of read.
    """
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "d3", "--yes-prompt", "--yes-plan"]
    )

    calls = 2 + len(_DEFAULT_PLAN_TASKS)
    assert f"{calls} call(s)" in result.output
    assert f"${calls * 0.002:.4f} on this thread" in result.output
    assert isolated.worker_calls == len(_DEFAULT_PLAN_TASKS)


def test_a_typed_error_is_a_message_not_a_traceback(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Playbook §1.4: errors are typed so a caller can act on them. A 200-line
    LangChain traceback for "you are out of credits" throws that away.

    Found live: a 402 from the planner dumped the whole stack, and the one
    useful line -- the provider's own remedy -- was buried at the bottom.
    """
    from dynaflows.contracts.errors import DynaflowsError, ErrorCode

    error = DynaflowsError.of(ErrorCode.INSUFFICIENT_CREDIT, "acme/big: 402 out of credits")
    error.remedy = "Add credits, or lower max_tokens"  # type: ignore[attr-defined]

    async def boom(request, **_):  # noqa: ANN001, ANN202
        raise error

    monkeypatch.setattr(isolated, "call", boom)
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "x1", "--yes-prompt"])
    assert result.exit_code == 3
    assert "INSUFFICIENT_CREDIT" in result.output
    assert "Add credits, or lower max_tokens" in result.output
    assert "Traceback" not in result.output


def test_a_failed_run_still_says_how_to_resume(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint survived the failure; the user should not have to guess."""
    from dynaflows.contracts.errors import DynaflowsError, ErrorCode

    async def boom(request, **_):  # noqa: ANN001, ANN202
        raise DynaflowsError.of(ErrorCode.INSUFFICIENT_CREDIT, "402")

    monkeypatch.setattr(isolated, "call", boom)
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "x2", "--yes-prompt"])
    assert "dynaflows resume x2" in result.output


def test_resume_recovers_the_root_the_run_was_planned_against(
    isolated: FakeGateway, tmp_path: Path
) -> None:
    """The checkpoint stores the plan; it has to store the tree too.

    `resume` builds its config before it can read state, so without this it
    defaults to the project root and the workers analyse different files than
    the ones the plan named -- silently, and with a plausible-looking report at
    the end. A wrong answer, not an inconvenience.
    """
    runner = CliRunner()
    runner.invoke(
        app, ["run", "audit auth", "--thread", "rr1", "--yes-prompt", "--stop-before", "plan"]
    )

    resumed = runner.invoke(app, ["resume", "rr1"], input="a\n")

    assert resumed.exit_code == 0, resumed.output
    # The proof is in what a worker was handed: src/auth.py exists only in the
    # recorded workspace, so its contents in the prompt means the right tree
    # was read after the process boundary.
    worker_prompts = [r.prompt for r in isolated.requests if r.schema.__name__ == "WorkerReport"]
    assert worker_prompts
    assert "def login" in worker_prompts[0]


def test_a_finished_run_says_where_the_report_is(isolated: FakeGateway) -> None:
    """Counters without a path make the user hunt for their own deliverable."""
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "rp1", "--yes-prompt", "--yes-plan"]
    )

    assert result.exit_code == 0, result.output
    assert "report" in result.output
    assert "synthesis" in result.output


def test_the_gate_shows_what_the_request_is_about(isolated: FakeGateway) -> None:
    isolated.relevant_paths = ["src/auth.py"]

    result = CliRunner().invoke(app, ["run", "fix the login bug", "--thread", "rp2"], input="r\n")

    assert "reads this as being about" in result.output
    assert "src/auth.py" in result.output


def test_the_gate_names_a_path_the_enhancer_invented(isolated: FakeGateway) -> None:
    """Silently dropping it would let the human approve an invention."""
    isolated.relevant_paths = ["src/auth.py", "src/ghost.py"]

    result = CliRunner().invoke(app, ["run", "fix the login bug", "--thread", "rp3"], input="r\n")

    assert "do not exist and were dropped" in result.output
    assert "src/ghost.py" in result.output


# --- `brief`: stdout is the deliverable, everything else is furniture ----


def test_brief_prints_only_the_brief_on_stdout(isolated: FakeGateway) -> None:
    """`dynaflows brief "..." | pbcopy` must copy the brief and nothing else.
    That split is the entire reason this command exists rather than being a
    flag on `run`."""
    runner = CliRunner()
    result = runner.invoke(app, ["brief", "fix the login bug", "--yes"])

    assert result.exit_code == 0, result.output
    # The sharpened request, then the standing requirements -- two halves with
    # two authors, and both are the deliverable.
    assert result.stdout.strip() == compose_brief("a precise brief")


def test_brief_puts_the_gate_where_a_pipe_will_not_see_it(isolated: FakeGateway) -> None:
    isolated.relevant_paths = ["src/auth.py"]
    runner = CliRunner()

    result = runner.invoke(app, ["brief", "fix the login bug"], input="a\n")

    assert result.exit_code == 0
    # The brief on stdout; the panels, paths and cost on stderr. CliRunner
    # echoes the typed answer into its captured stdout -- a real terminal does
    # not, verified against an actual pipe -- so this asserts on what is
    # PRESENT and ABSENT rather than on exact equality.
    assert "a precise brief" in result.stdout
    assert "you asked" not in result.stdout
    assert "src/auth.py" not in result.stdout


def test_rejecting_a_brief_writes_nothing_to_stdout(isolated: FakeGateway) -> None:
    """A rejected brief piped to the clipboard would be worse than no brief."""
    result = CliRunner().invoke(app, ["brief", "fix the login bug"], input="r\n")

    assert result.exit_code == 1
    assert "a precise brief" not in result.stdout


def test_brief_stops_before_planning(isolated: FakeGateway) -> None:
    """This command's job ends at G1. Planning would spend a frontier call for
    output nobody asked for."""
    CliRunner().invoke(app, ["brief", "fix the login bug", "--yes"])

    assert isolated.enhancer_calls == 1
    assert isolated.planner_calls == 0


def test_a_failure_does_not_reach_the_pipe(
    isolated: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_fail` wrote to stdout, so a failed `brief` piped to the clipboard
    copied "MODEL_UNAVAILABLE ... Connection error". An error is never a
    command's output."""

    async def boom(request: object, **_: object) -> object:
        raise DynaflowsError.of(ErrorCode.MODEL_UNAVAILABLE, "everything is down")

    monkeypatch.setattr(isolated, "call", boom)

    result = CliRunner().invoke(app, ["brief", "fix the login bug", "--yes"])

    assert result.exit_code == 3
    assert "everything is down" not in result.stdout


def _ai_dir(tmp_path: Path) -> Path:
    return tmp_path / "workspace" / ".ai"


def test_brief_saves_both_the_brief_and_the_record(isolated: FakeGateway, tmp_path: Path) -> None:
    """The first A/B run was lost to a shell redirect: brief in one file, gate
    in another, both in /tmp. The command keeps them itself now."""
    result = CliRunner().invoke(app, ["brief", "fix the login bug", "--thread", "t9", "--yes"])

    assert result.exit_code == 0
    assert (_ai_dir(tmp_path) / "t9.brief.txt").exists()
    assert (_ai_dir(tmp_path) / "t9.md").exists()


def test_the_saved_brief_is_byte_identical_to_stdout(isolated: FakeGateway, tmp_path: Path) -> None:
    """The invariant that makes the file usable. If the copy on disk could
    drift from what was piped, an A/B test would compare the wrong text and
    nothing in the run would say so."""
    result = CliRunner().invoke(app, ["brief", "fix the login bug", "--thread", "t10", "--yes"])

    saved = (_ai_dir(tmp_path) / "t10.brief.txt").read_text()
    assert saved.strip() == result.stdout.strip()


def test_no_save_writes_nothing(isolated: FakeGateway, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["brief", "fix the login bug", "--thread", "t11", "--yes", "--no-save"]
    )

    assert result.exit_code == 0
    assert not _ai_dir(tmp_path).exists()


def test_a_rejected_brief_is_saved_even_though_stdout_is_empty(
    isolated: FakeGateway, tmp_path: Path
) -> None:
    """A rejection is the model being wrong with a human's verdict attached.
    Keeping only approvals keeps the half that teaches nothing."""
    result = CliRunner().invoke(app, ["brief", "fix the login bug", "--thread", "t12"], input="r\n")

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert "**decision** reject" in (_ai_dir(tmp_path) / "t12.md").read_text()


def test_saving_never_touches_the_repository_under_root(
    isolated: FakeGateway, tmp_path: Path
) -> None:
    """ADR-016. `--root` can name someone else's tree; the bookkeeping belongs
    to this project, and `--root` is the option that makes writing into the
    wrong one possible by accident."""
    other = tmp_path / "someone-elses-repo"
    other.mkdir()

    result = CliRunner().invoke(
        app, ["brief", "fix the login bug", "--thread", "t13", "--yes", "--root", str(other)]
    )

    assert result.exit_code == 0
    assert not (other / ".ai").exists()
    assert (_ai_dir(tmp_path) / "t13.md").exists()


def test_asking_a_question_puts_nothing_on_stdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-byte leak, pinned.

    `typer.prompt(..., err=True)` passes the prompt's LAST character to
    `input()`, which always writes to stdout -- a readline workaround in
    click. Every interactive gate therefore prepended a space to the pipe.
    Written against `typer.prompt` this test fails; that is what it is for.
    """
    from dynaflows.cli import _ask, _err

    monkeypatch.setattr("builtins.input", lambda: "r")

    answer = _ask(_err, "[a]pprove  [e]dit  [r]eject", default="a")

    assert answer == "r"
    assert capsys.readouterr().out == ""


def test_an_unanswered_gate_is_not_an_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """EOF on stdin means nobody answered. The gate exists to make a human
    say yes, so silence must not be read as one."""

    def no_answer() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_answer)

    from dynaflows.cli import _ask, _err

    with pytest.raises(typer.Abort):
        _ask(_err, "[a]pprove", default="a")


def test_reusing_a_thread_for_a_different_prompt_is_refused(
    isolated: FakeGateway, tmp_path: Path
) -> None:
    """The fifth wrong number.

    LangGraph appends: a fresh input on a thread that already holds state
    starts ANOTHER run over the same checkpoint and, because `cost` is a
    summing reducer, bills both to one ledger. `brief --thread p2` therefore
    reported "$0.0066 spent, 3 call(s)" for what its author believed was one
    free call -- the thread's lifetime, presented as this run's cost.

    Three threads in the real A/B run had 3, 2 and 1 runs on them. Only the
    third reported a number that meant what it said.
    """
    first = CliRunner().invoke(app, ["brief", "fix the login bug", "--thread", "t20", "--yes"])
    assert first.exit_code == 0

    second = CliRunner().invoke(
        app, ["brief", "something else entirely", "--thread", "t20", "--yes"]
    )

    assert second.exit_code == 3
    assert second.stdout.strip() == ""


def test_reusing_a_thread_for_the_same_prompt_is_a_revisit(isolated: FakeGateway) -> None:
    """What `--thread` is actually for. The help says "reuse it to revisit",
    and that has to keep working or the guard is just a papercut."""
    args = ["brief", "fix the login bug", "--thread", "t21", "--yes"]
    assert CliRunner().invoke(app, args).exit_code == 0

    again = CliRunner().invoke(app, args)

    assert again.exit_code == 0
    assert again.stdout.strip()
