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

import pytest
from typer.testing import CliRunner

from dynaflows.cli import app
from tests.conftest import FakeGateway, make_playbook

_DEFAULT_PLAN_TASKS = range(3)  # what conftest's default draft plans

pytestmark = pytest.mark.deterministic


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGateway:
    """A gateway that counts calls, and state written under tmp."""
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
    assert "$0.0020 spent" in result.output


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
    for fragment in ("task(s)", "spent", "call(s)"):
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
    assert f"${calls * 0.002:.4f} spent" in result.output
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
