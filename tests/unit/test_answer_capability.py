"""The `answer` capability. ADR-023, use case 3.

A question about a codebase is not an audit of it, and the difference is not
cosmetic: the audit path DISCARDS a grounded claim that merely describes the
code, which is the correct rule for finding defects and the exact opposite of
what "how does the playbook search work" needs.

So `answer` is a registered capability with its own schema, its own prompt and
its own substance rule -- and the SAME citation checking, because a fabricated
file is fabricated whatever was asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynaflows.contracts.state import PlanTask, initial_state
from dynaflows.graph import nodes
from dynaflows.graph.prompts import AnswerReport, Observation, WorkerReport
from tests.conftest import FakeGateway, graph_config

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def an_answer_task(**kwargs: Any) -> PlanTask:
    return PlanTask(
        task_id=kwargs.pop("task_id", "q1"),
        capability="answer",
        objective=kwargs.pop("objective", "how does login decide success"),
        **kwargs,
    )


async def run_worker(task: PlanTask, gateway: FakeGateway, root: Path) -> dict:
    config = graph_config("w", gateway, root)
    state = {**initial_state("r1", "w", "ask"), "task": task}
    return await nodes.worker(state, config)  # type: ignore[arg-type]


async def test_an_answer_task_is_asked_the_answer_question(workspace: Path) -> None:
    """The dispatch. Same node, different contract -- and if the schema did
    not change, the model would be asked for a severity and a remediation for
    "how does login work", which it would invent."""
    gateway = FakeGateway()

    await run_worker(an_answer_task(inputs=["src/auth.py"]), gateway, workspace)

    request = gateway.requests[-1]
    assert request.schema is AnswerReport
    assert "A description of working code IS a valid answer here" in request.system


async def test_an_analyse_task_is_still_asked_for_findings(workspace: Path) -> None:
    """The other half of the dispatch, asserted in the same file. Testing only
    the new branch is how the old one quietly stops being reached."""
    gateway = FakeGateway()
    task = PlanTask(task_id="t1", capability="analyse", objective="review login")

    await run_worker(task, gateway, workspace)

    assert gateway.requests[-1].schema is WorkerReport


async def test_the_answer_becomes_the_summary(workspace: Path) -> None:
    """`WorkerResult.summary` is what the synthesizer and the terminal see. An
    answer worker has no `summary` field -- it has `answer` -- and a silent
    getattr miss here would produce an empty string and no error."""
    gateway = FakeGateway()
    gateway.answer_report = lambda _: AnswerReport(
        answer="Login returns True unconditionally.",
        examined=["src/auth.py"],
        observations=[],
        context_was_sufficient=True,
    )

    out = await run_worker(an_answer_task(inputs=["src/auth.py"]), gateway, workspace)

    assert out["results"][0].summary == "Login returns True unconditionally."


async def test_a_description_survives_verification(workspace: Path) -> None:
    """The whole point. As a `Finding` this claim is dropped NOT_A_DEFECT; as
    an answer it is the thing that was asked for."""
    gateway = FakeGateway()
    gateway.answer_report = lambda _: AnswerReport(
        answer="It always succeeds.",
        examined=["src/auth.py"],
        observations=[
            Observation(
                claim="login() returns True with no checks.",
                file="src/auth.py",
                lines="4-5",
                quoted_lines="def login():\n    return True",
            )
        ],
        context_was_sufficient=True,
    )

    out = await run_worker(an_answer_task(inputs=["src/auth.py"]), gateway, workspace)

    result = out["results"][0]
    assert result.findings_grounded == 1
    assert result.status == "ok"


async def test_a_fabricated_citation_is_still_dropped(workspace: Path) -> None:
    """Relaxing the substance rule must not relax the citation rules. Run `w1`
    invented four filenames and reported a HIGH severity finding about them;
    that is a fabrication whatever the question was."""
    gateway = FakeGateway()
    gateway.answer_report = lambda _: AnswerReport(
        answer="Sessions are stored in Redis.",
        examined=["src/auth.py"],
        observations=[
            Observation(
                claim="The session store is Redis.",
                file="src/session_store.py",
                lines="10-12",
                quoted_lines="redis.Redis(host=settings.REDIS_HOST)",
            )
        ],
        context_was_sufficient=True,
    )

    out = await run_worker(an_answer_task(inputs=["src/auth.py"]), gateway, workspace)

    result = out["results"][0]
    assert result.findings_reported == 1
    assert result.findings_grounded == 0
    # Every claim it made failed. Not the same as making none (ADR-004).
    assert result.status == "degraded"


async def test_the_artifact_leads_with_the_answer(workspace: Path) -> None:
    """A findings report is a list and reads as one. An answer is prose, and
    burying it under a table of evidence makes the reader rebuild it."""
    gateway = FakeGateway()
    gateway.answer_report = lambda _: AnswerReport(
        answer="Login returns True unconditionally.",
        examined=["src/auth.py"],
        observations=[],
        context_was_sufficient=True,
    )

    out = await run_worker(an_answer_task(inputs=["src/auth.py"]), gateway, workspace)

    document = (workspace / ".dynaflows" / out["results"][0].artifact.path).read_text()
    assert document.startswith("# Login returns True unconditionally.")
    assert "_None verified. The answer above is unsupported._" in document


def test_observations_survive_the_round_trip_to_the_synthesizer() -> None:
    """Where `answer` would have died silently.

    `collect` validates each stored row against a model. It validated every
    row as a `Finding`; an observation has no `failure`, `severity` or
    `remediation`, so validation raised, a bare `except ... continue` swallowed
    it, and the report said "no verified findings" for a run that verified
    several. No error, no counter, no way to tell it from a worker that found
    nothing.
    """
    from dynaflows.contracts.state import ArtifactRef, WorkerResult
    from dynaflows.graph import synthesis

    rows = [
        Observation(
            claim="login() returns True with no checks.",
            file="src/auth.py",
            lines="4-5",
            quoted_lines="def login():\n    return True",
        ).model_dump()
    ]
    result = WorkerResult(
        task_id="q1",
        capability="answer",
        status="ok",
        summary="It always succeeds.",
        findings_reported=1,
        findings_grounded=1,
        findings_ref=ArtifactRef(sha="s", path="q1.json", kind="json"),
    )

    claims, accounting = synthesis.collect([result], lambda _: rows)

    assert len(claims) == 1
    assert accounting.unreadable == 0
    assert claims[0].finding.claim == "login() returns True with no checks."


def test_an_unreadable_row_is_counted_rather_than_dropped() -> None:
    """AP-20. A row a worker verified and the synthesizer could not read back
    is work that was done and paid for and then lost, and the reader cannot
    infer it from anything else on the page."""
    from dynaflows.contracts.state import ArtifactRef, WorkerResult
    from dynaflows.graph import synthesis

    result = WorkerResult(
        task_id="q1",
        capability="answer",
        status="ok",
        summary="s",
        findings_reported=1,
        findings_grounded=1,
        findings_ref=ArtifactRef(sha="s", path="q1.json", kind="json"),
    )

    claims, accounting = synthesis.collect([result], lambda _: [{"nonsense": True}])

    assert not claims
    assert accounting.unreadable == 1
    assert "could not be read back" in synthesis.render_accounting(accounting, None)


def test_an_observation_is_not_rendered_as_a_defect() -> None:
    """The synthesizer is shown severity and a proposed change for a finding.
    An observation has neither, and inventing placeholders would tell the
    model something untrue about a claim that never made that statement."""
    from dynaflows.graph import synthesis

    rendered = synthesis.render_findings(
        [
            synthesis.Corroborated(
                id="q1#1",
                finding=Observation(
                    claim="login() returns True with no checks.",
                    file="src/auth.py",
                    lines="4-5",
                    quoted_lines="def login():\n    return True",
                ),
                reported_by=("q1",),
            )
        ]
    )

    assert "proposed:" not in rendered
    assert "(high)" not in rendered and "(low)" not in rendered
    assert "login() returns True with no checks." in rendered
