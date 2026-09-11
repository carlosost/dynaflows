"""Collecting, deduplicating and accounting for findings. ADR-020.

Everything here is deterministic and testable without a graph, a gateway or an
event loop. The model's contribution to the final report is one section; this
module is the rest of it, and it is the half that cannot be lost -- a run whose
synthesis call fails still tells the user exactly what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dynaflows.contracts.state import EvaluationReport, WorkerResult
from dynaflows.graph.prompts import Finding

_WHITESPACE = re.compile(r"\s+")
_LINE_PREFIX = re.compile(r"^\s*\d+\|\s?", re.MULTILINE)


def _key(finding: Finding) -> tuple[str, str]:
    """What makes two findings the same finding.

    File plus normalised evidence, NOT the claim: two workers describing the
    same line in different words have found one problem, and the wording is
    the least reliable part of either report.
    """
    evidence = _WHITESPACE.sub(" ", _LINE_PREFIX.sub("", finding.evidence)).strip().lower()
    return finding.file.strip(), evidence


@dataclass(frozen=True, slots=True)
class Corroborated:
    """One finding, and every worker that independently reached it."""

    id: str
    finding: Finding
    reported_by: tuple[str, ...]

    @property
    def corroboration(self) -> int:
        return len(self.reported_by)


@dataclass(slots=True)
class Accounting:
    """What the run did, computed rather than described.

    Every number here comes from state. None of it is a model's account of its
    own limitations, which is the one source guaranteed to understate them.
    """

    tasks: int = 0
    ok: int = 0
    degraded: int = 0
    failed: int = 0
    findings_reported: int = 0
    findings_grounded: int = 0
    unique_findings: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def findings_discarded(self) -> int:
        return self.findings_reported - self.findings_grounded


def collect(results: list[WorkerResult], load: object) -> tuple[list[Corroborated], Accounting]:
    """Gather every verified finding, collapse duplicates, and count the rest.

    `load` reads a findings reference back from the run store. Injected rather
    than imported so this stays testable without a filesystem.
    """
    accounting = Accounting(tasks=len(results))
    merged: dict[tuple[str, str], Corroborated] = {}
    order: list[tuple[str, str]] = []

    for result in results:
        accounting.ok += result.status == "ok"
        accounting.degraded += result.status == "degraded"
        accounting.failed += result.status == "failed"
        accounting.findings_reported += result.findings_reported
        accounting.findings_grounded += result.findings_grounded
        if result.status == "failed":
            reason = result.error.message if result.error else "no reason recorded"
            accounting.failures.append(f"{result.task_id}: {reason}")

        if result.findings_ref is None:
            continue
        raw = load(result.findings_ref)  # type: ignore[operator]
        if not isinstance(raw, list):
            continue
        for index, item in enumerate(raw):
            try:
                finding = Finding.model_validate(item)
            except Exception:  # noqa: BLE001 -- a bad row is one finding, not the run
                continue
            key = _key(finding)
            existing = merged.get(key)
            if existing is None:
                merged[key] = Corroborated(
                    id=f"{result.task_id}#{index + 1}",
                    finding=finding,
                    reported_by=(result.task_id,),
                )
                order.append(key)
            elif result.task_id not in existing.reported_by:
                merged[key] = Corroborated(
                    id=existing.id,
                    finding=existing.finding,
                    reported_by=(*existing.reported_by, result.task_id),
                )

    findings = [merged[key] for key in order]
    accounting.unique_findings = len(findings)
    return findings, accounting


def render_accounting(accounting: Accounting, evaluation: EvaluationReport | None) -> str:
    """The computed half of the report. ADR-020.

    Written first and written from state, so it survives a synthesis call that
    fails, returns nothing, or is never made. A reader who sees only prose
    cannot tell a thorough run from one where three workers died.
    """
    lines = ["## What this run actually did", ""]
    lines.append(
        f"- {accounting.tasks} task(s): {accounting.ok} ok, "
        f"{accounting.degraded} degraded, {accounting.failed} failed"
    )
    lines.append(
        f"- {accounting.findings_reported} finding(s) claimed, "
        f"{accounting.findings_grounded} survived citation checking"
    )
    if accounting.findings_discarded:
        lines.append(
            f"- **{accounting.findings_discarded} discarded as ungrounded.** A worker cited a "
            "file it was not shown, a line past the end of one, or a quote that is not there. "
            "Each worker's report lists its own."
        )
    if accounting.unique_findings != accounting.findings_grounded:
        collapsed = accounting.findings_grounded - accounting.unique_findings
        lines.append(f"- {collapsed} duplicate(s) collapsed; {accounting.unique_findings} distinct")
    for failure in accounting.failures:
        lines.append(f"- **failed**: {failure}")
    if evaluation is not None and not evaluation.passed:
        lines.append("")
        lines.append("**This run did not pass its own checks:**")
        lines.extend(f"- {reason}" for reason in evaluation.reasons)
    if accounting.degraded or accounting.failed or accounting.findings_discarded:
        lines.append("")
        lines.append(
            "Read the section below in that light: it summarises only the findings that "
            "survived checking, and it cannot describe what a failed or degraded worker "
            "did not report."
        )
    return "\n".join(lines)


def render_findings(findings: list[Corroborated]) -> str:
    """The findings as the synthesizer is shown them, with stable ids.

    The id is what the model cites, and citing is what makes its output
    checkable (ADR-019, one level up).
    """
    blocks = []
    for item in findings:
        corroboration = (
            f" [corroborated by {item.corroboration} workers]" if item.corroboration > 1 else ""
        )
        blocks.append(
            f"[{item.id}] ({item.finding.severity}){corroboration} {item.finding.claim}\n"
            f"  {item.finding.file}:{item.finding.lines}\n"
            f"  evidence: {item.finding.evidence.strip()[:300]}\n"
            f"  proposed: {item.finding.remediation}"
        )
    return "\n\n".join(blocks)
