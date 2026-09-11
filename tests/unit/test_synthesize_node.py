"""The synthesize node. ADR-020, through the graph.

The property that matters: the user learns what the run did in every case --
when the model writes a good summary, when it invents citations, when the call
fails, and when there is nothing to summarise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.state import EvaluationReport, initial_state
from dynaflows.graph import build_graph, nodes, open_checkpointer
from dynaflows.graph.prompts import SynthesisDraft, SynthesisSection, WorkerReport
from tests.conftest import FakeGateway, a_finding, cfg_factory, draft_with

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cfg(workspace: Path) -> Any:
    return cfg_factory(workspace, ["prompt", "plan"])


def reporting_findings(n: int = 1) -> Any:
    def _report(request: Any) -> WorkerReport:
        return WorkerReport(
            summary="found things",
            examined=["src/auth.py"],
            findings=[a_finding() for _ in range(n)],
            context_was_sufficient=True,
        )

    return _report


async def run_graph(tmp_path: Path, cfg: Any, gateway: FakeGateway, thread: str) -> dict:
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        return await graph.ainvoke(initial_state("r", thread, "audit"), cfg(thread, gateway))


def synthesis_text(final: dict) -> str:
    return Path(final["synthesis"].path).read_text(encoding="utf-8")


async def test_a_run_with_findings_produces_a_synthesis(tmp_path: Path, cfg: Any) -> None:
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(2)
    gateway.worker_report = reporting_findings()

    final = await run_graph(tmp_path, cfg, gateway, "s1")

    assert gateway.synthesis_calls == 1
    text = synthesis_text(final)
    assert "the thing has problems" in text
    assert "## What this run actually did" in text
    assert "## Summary" in text


async def test_the_computed_half_comes_first(tmp_path: Path, cfg: Any) -> None:
    """A reader who stops after the first screen has seen what the run did,
    not a confident summary of part of it."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(2)
    gateway.worker_report = reporting_findings()

    text = synthesis_text(await run_graph(tmp_path, cfg, gateway, "s2"))

    assert text.index("What this run actually did") < text.index("## Summary")


async def test_no_verified_findings_means_no_model_call(tmp_path: Path, cfg: Any) -> None:
    """Paying a frontier model to write prose about an empty list produces
    confident prose about nothing -- run w1's failure at the last step."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(2)
    # The default report finds nothing.

    final = await run_graph(tmp_path, cfg, gateway, "s3")

    assert gateway.synthesis_calls == 0
    text = synthesis_text(final)
    assert "No verified findings" in text
    assert "What this run actually did" in text


async def test_a_section_citing_an_invented_id_is_discarded(tmp_path: Path, cfg: Any) -> None:
    """ADR-019 one level up: the synthesizer is a summariser and may not
    introduce claims."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(1)
    gateway.worker_report = reporting_findings()
    gateway.synthesis = lambda request: SynthesisDraft(
        headline="h",
        sections=[
            SynthesisSection(heading="Real", body="ok", finding_ids=["task-1#1"]),
            SynthesisSection(heading="Invented", body="no", finding_ids=["ghost#9"]),
        ],
    )

    final = await run_graph(tmp_path, cfg, gateway, "s4")

    text = synthesis_text(final)
    assert "Real" in text
    assert "Invented" not in text
    assert "1 section(s) were discarded" in text
    assert final["degraded"] is True


async def test_a_section_citing_nothing_is_discarded(tmp_path: Path, cfg: Any) -> None:
    """Prose with no citation is prose the reader cannot check."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(1)
    gateway.worker_report = reporting_findings()
    gateway.synthesis = lambda request: SynthesisDraft(
        headline="h", sections=[SynthesisSection(heading="Vibes", body="feels bad", finding_ids=[])]
    )

    text = synthesis_text(await run_graph(tmp_path, cfg, gateway, "s5"))

    assert "Vibes" not in text


async def test_a_failed_synthesis_call_still_reports_the_run(tmp_path: Path, cfg: Any) -> None:
    """Losing the summary is a smaller loss than losing the report. Every
    finding is still in the artifact."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(1)
    gateway.worker_report = reporting_findings()

    def boom(request: Any) -> Any:
        raise DynaflowsError.of(ErrorCode.RATE_LIMIT, "429 everywhere")

    gateway.synthesis = boom

    final = await run_graph(tmp_path, cfg, gateway, "s6")

    text = synthesis_text(final)
    assert "Synthesis unavailable" in text
    assert "429 everywhere" in text
    assert "login always returns True" in text
    assert final["degraded"] is True


async def test_the_findings_always_appear_verbatim(tmp_path: Path, cfg: Any) -> None:
    """The summary is a convenience. The findings are the deliverable, and a
    reader must never have to trust the summary to reach them."""
    gateway = FakeGateway()
    gateway.plan_draft = lambda: draft_with(1)
    gateway.worker_report = reporting_findings()

    text = synthesis_text(await run_graph(tmp_path, cfg, gateway, "s7"))

    assert "## Verified findings" in text
    assert "src/auth.py:4-5" in text


async def test_a_degraded_run_says_so_in_the_synthesis(tmp_path: Path) -> None:
    """ADR-004's requirement, which ADR-020 implements."""
    evaluation = EvaluationReport(
        task_count=2,
        ok_count=0,
        failed_count=1,
        empty_count=0,
        degraded_count=1,
        passed=False,
        reasons=["1 task(s) failed"],
    )
    text = nodes.synth.render_accounting(
        nodes.synth.Accounting(tasks=2, ok=0, degraded=1, failed=1), evaluation
    )

    assert "did not pass its own checks" in text
    assert "1 task(s) failed" in text
