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
from tests.conftest import FakeGateway

pytestmark = pytest.mark.deterministic


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGateway:
    """A gateway that counts calls, and state written under tmp."""
    gateway = FakeGateway("a precise brief", ["assumed the HTTP layer"])
    monkeypatch.setattr("dynaflows.gateway.client.get_gateway", lambda **_: gateway)
    monkeypatch.setenv("DYNAFLOWS_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return gateway


def test_run_reaches_the_enhancer_with_a_gateway(isolated: FakeGateway) -> None:
    """The regression. Without a gateway in config the node raises
    CONFIG_INVALID and the run produces nothing."""
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "t1", "--yes-prompt"])
    assert result.exit_code == 0, result.output
    assert isolated.calls == 1
    assert "Completed" in result.output


def test_run_halts_at_gate_g1_and_shows_the_rewrite(isolated: FakeGateway) -> None:
    """Answering 'r' at the prompt: the human sees both texts and says no."""
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "t2"], input="r\n")
    assert "audit auth" in result.output
    assert "a precise brief" in result.output
    assert "assumed the HTTP layer" in result.output
    assert result.exit_code == 2, result.output
    assert "rejected by human at gate G1" in result.output


def test_approving_at_the_gate_completes_the_run(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "t3"], input="a\n")
    assert result.exit_code == 0, result.output
    assert "Completed" in result.output
    assert isolated.calls == 1


def test_a_rejection_reports_what_it_cost_before_stopping(isolated: FakeGateway) -> None:
    """ADR-005: the value of gating at G1 is that a 'no' costs one small-tier
    call. Saying so out loud is how that stays true."""
    result = CliRunner().invoke(app, ["run", "audit auth", "--thread", "t4"], input="r\n")
    assert "spent $0.0020" in result.output


def test_yes_prompt_never_asks(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(
        app, ["run", "audit auth", "--thread", "t5", "--yes-prompt"], input=""
    )
    assert result.exit_code == 0, result.output
    assert "approve" not in result.output.lower()


def test_resume_continues_a_gated_run_without_re_running_the_enhancer(
    isolated: FakeGateway,
) -> None:
    """ADR-007 through the CLI, not just the graph."""
    runner = CliRunner()
    first = runner.invoke(app, ["run", "audit auth", "--thread", "t6"], input="r\n")
    assert first.exit_code == 2
    assert isolated.calls == 1


def test_resume_on_an_unknown_thread_fails_loudly(isolated: FakeGateway) -> None:
    result = CliRunner().invoke(app, ["resume", "no-such-thread"])
    assert result.exit_code == 1
    assert "No checkpoint" in result.output
