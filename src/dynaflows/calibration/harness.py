"""Loading the manifest and scoring a worker's findings against it. ADR-022.

Deterministic and model-free, like every other check in this project that
decides whether something is true.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from dynaflows.graph.grounding import parse_range
from dynaflows.graph.prompts import Finding

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True, slots=True)
class Defect:
    id: str
    start: int
    end: int
    kind: str
    why: str

    def covers(self, start: int, end: int) -> bool:
        """Any overlap counts.

        A model citing the `except` line and one citing the `return None`
        beneath it have found the same defect, and scoring them differently
        would measure citation style rather than detection.
        """
        return start <= self.end and end >= self.start


@dataclass(frozen=True, slots=True)
class Scorecard:
    findings: tuple[Finding, ...]
    found: tuple[str, ...]
    missed: tuple[Defect, ...]
    unplanted: tuple[Finding, ...]
    discarded: int
    reported: int

    @property
    def planted(self) -> int:
        return len(self.found) + len(self.missed)

    @property
    def recall(self) -> float:
        return len(self.found) / self.planted if self.planted else 0.0

    @property
    def discard_rate(self) -> float:
        """The §4.5 number ADR-019 said would decide whether the evidence match
        is too strict. It has had no measurement behind it until now."""
        return self.discarded / self.reported if self.reported else 0.0


def fixture_paths(name: str = "broken_client") -> tuple[Path, Path]:
    return FIXTURES / f"{name}.py", FIXTURES / f"{name}.manifest.toml"


def load_manifest(path: Path) -> list[Defect]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Defect(
            id=str(entry["id"]),
            start=int(entry["lines"][0]),
            end=int(entry["lines"][1]),
            kind=str(entry.get("kind", "")),
            why=str(entry.get("why", "")),
        )
        for entry in raw.get("defect", [])
    ]


def score(
    findings: list[Finding], defects: list[Defect], *, reported: int, discarded: int
) -> Scorecard:
    """Match verified findings to planted defects by line overlap.

    `reported` and `discarded` come from the grounding check, not from the list
    above: a finding rejected as ungrounded still tells us the worker tried,
    and counting only what survived would hide a worker that is inventing.
    """
    found: list[str] = []
    unplanted: list[Finding] = []

    for finding in findings:
        span = parse_range(finding.lines)
        if span is None:
            unplanted.append(finding)
            continue
        hit = next((d for d in defects if d.covers(*span)), None)
        if hit is None:
            unplanted.append(finding)
        elif hit.id not in found:
            found.append(hit.id)

    return Scorecard(
        findings=tuple(findings),
        found=tuple(found),
        missed=tuple(d for d in defects if d.id not in found),
        unplanted=tuple(unplanted),
        discarded=discarded,
        reported=reported,
    )
