"""ADR-020: the synthesis is two documents, one computed and one written.

The computed half exists because a model asked to describe the limits of its
own output understates them, and an understatement reads exactly like an
accurate summary. Every number in it comes from state.
"""

from __future__ import annotations

from typing import Any

import pytest

from dynaflows.contracts.errors import ErrorCode, ErrorEnvelope
from dynaflows.contracts.state import ArtifactRef, EvaluationReport, WorkerResult
from dynaflows.graph import synthesis as synth
from dynaflows.graph.prompts import Finding

pytestmark = pytest.mark.deterministic


def a_finding(**kwargs: Any) -> Finding:
    return Finding(
        claim=kwargs.pop("claim", "no password check"),
        file=kwargs.pop("file", "src/auth.py"),
        lines=kwargs.pop("lines", "2-3"),
        quoted_lines=kwargs.pop("quoted_lines", "if not user:"),
        severity=kwargs.pop("severity", "high"),
        failure=kwargs.pop("failure", "An empty password authenticates any caller."),
        remediation=kwargs.pop("remediation", "Validate it."),
    )


def a_result(task_id: str, findings: list[Finding], **kwargs: Any) -> WorkerResult:
    return WorkerResult(
        task_id=task_id,
        status=kwargs.pop("status", "ok"),
        summary=kwargs.pop("summary", "done"),
        findings_reported=kwargs.pop("reported", len(findings)),
        findings_grounded=len(findings),
        findings_ref=ArtifactRef(sha=task_id, path=f"/tmp/{task_id}.json", tokens=10),
        **kwargs,
    )


def loader(mapping: dict[str, list[Finding]]) -> Any:
    def _load(ref: ArtifactRef) -> Any:
        return [f.model_dump() for f in mapping.get(ref.sha, [])]

    return _load


# --- collecting and collapsing -------------------------------------------


def test_findings_from_every_worker_are_collected() -> None:
    results = [a_result("t1", [a_finding()]), a_result("t2", [a_finding(file="src/db.py")])]
    findings, accounting = synth.collect(
        results, loader({"t1": [a_finding()], "t2": [a_finding(file="src/db.py")]})
    )

    assert len(findings) == 2
    assert accounting.findings_grounded == 2


def test_the_same_finding_from_two_workers_is_one_finding_with_two_reporters() -> None:
    """Corroboration, not duplication. Two workers reaching the same line
    independently is a fact worth reporting; it is not two problems."""
    same = a_finding()
    reworded = a_finding(claim="the password is never checked")

    findings, accounting = synth.collect(
        [a_result("t1", [same]), a_result("t2", [reworded])],
        loader({"t1": [same], "t2": [reworded]}),
    )

    assert len(findings) == 1
    assert findings[0].corroboration == 2
    assert findings[0].reported_by == ("t1", "t2")
    # Both were grounded; one is distinct. Both numbers are reported.
    assert (accounting.findings_grounded, accounting.unique_findings) == (2, 1)


def test_the_claim_is_not_part_of_the_identity() -> None:
    """The wording is the least reliable part of either report."""
    findings, _ = synth.collect(
        [a_result("t1", [a_finding(claim="A")]), a_result("t2", [a_finding(claim="B")])],
        loader({"t1": [a_finding(claim="A")], "t2": [a_finding(claim="B")]}),
    )

    assert len(findings) == 1


def test_evidence_differing_only_in_whitespace_is_the_same_finding() -> None:
    findings, _ = synth.collect(
        [
            a_result("t1", [a_finding()]),
            a_result("t2", [a_finding(quoted_lines="2|  if not user:")]),
        ],
        loader({"t1": [a_finding()], "t2": [a_finding(quoted_lines="2|  if not user:")]}),
    )

    assert len(findings) == 1


def test_ids_are_stable_and_traceable_to_a_task() -> None:
    findings, _ = synth.collect([a_result("t1", [a_finding()])], loader({"t1": [a_finding()]}))

    assert findings[0].id == "t1#1"


def test_an_unreadable_artifact_loses_one_worker_not_the_run() -> None:
    """Read at synthesis time, after every worker has been paid for."""
    findings, accounting = synth.collect(
        [a_result("t1", [a_finding()]), a_result("t2", [a_finding(file="x.py")])],
        lambda ref: [a_finding().model_dump()] if ref.sha == "t1" else None,
    )

    assert len(findings) == 1
    assert accounting.tasks == 2


def test_a_malformed_finding_row_is_skipped_not_fatal() -> None:
    findings, _ = synth.collect(
        [a_result("t1", [a_finding()])],
        lambda ref: [{"garbage": True}, a_finding().model_dump()],
    )

    assert len(findings) == 1


# --- the computed half ----------------------------------------------------


def test_the_accounting_names_what_failed() -> None:
    results = [
        a_result("t1", [a_finding()]),
        a_result(
            "t2",
            [],
            status="failed",
            error=ErrorEnvelope(code=ErrorCode.RATE_LIMIT, message="429 everywhere"),
        ),
    ]
    _, accounting = synth.collect(results, loader({"t1": [a_finding()]}))
    text = synth.render_accounting(accounting, None)

    assert "1 ok, 0 degraded, 1 failed" in text
    assert "429 everywhere" in text


def test_discarded_findings_are_reported_not_just_absent() -> None:
    """A reader who cannot see that six claims were thrown out reads the
    silence as diligence."""
    results = [a_result("t1", [a_finding()], reported=7)]
    _, accounting = synth.collect(results, loader({"t1": [a_finding()]}))
    text = synth.render_accounting(accounting, None)

    assert accounting.findings_discarded == 6
    assert "6 discarded as ungrounded" in text


def test_a_failing_evaluation_is_quoted_verbatim() -> None:
    evaluation = EvaluationReport(
        task_count=3,
        ok_count=0,
        failed_count=0,
        empty_count=0,
        degraded_count=3,
        passed=False,
        reasons=["no task succeeded cleanly (3 degraded, 0 failed of 3)"],
    )
    _, accounting = synth.collect([], loader({}))

    text = synth.render_accounting(accounting, evaluation)

    assert "did not pass its own checks" in text
    assert "no task succeeded cleanly" in text


def test_a_clean_run_gets_no_caveat_paragraph() -> None:
    """The warning must not fire on a clean run or it becomes furniture."""
    results = [a_result("t1", [a_finding()])]
    _, accounting = synth.collect(results, loader({"t1": [a_finding()]}))
    evaluation = EvaluationReport(
        task_count=1, ok_count=1, failed_count=0, empty_count=0, degraded_count=0, passed=True
    )

    text = synth.render_accounting(accounting, evaluation)

    assert "Read the section below in that light" not in text
    assert "did not pass" not in text


def test_corroboration_is_visible_to_the_synthesizer() -> None:
    findings, _ = synth.collect(
        [a_result("t1", [a_finding()]), a_result("t2", [a_finding()])],
        loader({"t1": [a_finding()], "t2": [a_finding()]}),
    )

    assert "corroborated by 2 workers" in synth.render_findings(findings)
