"""Checking a worker's citations. ADR-019.

Deterministic, no model, no second call. The graph knows exactly what text each
worker was shown, so a citation is checkable by string search: a file either
was in the pack or was not, a line either exists or does not, a quote either
appears or does not.

Run `w1` is the reason. A worker with no source invented four filenames, cited
line numbers in them, and reported a HIGH severity finding. Every one of those
citations would have failed the first check here.

Kept apart from the node so every rule is testable without a graph, a gateway
or an event loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from dynaflows.contracts.playbook import Chunk
from dynaflows.graph.prompts import Finding

_LINE_PREFIX = re.compile(r"^\s*\d+\|\s?", re.MULTILINE)
_RANGE = re.compile(r"^\s*(\d+)\s*(?:[-–:]\s*(\d+))?\s*$")
_WHITESPACE = re.compile(r"\s+")


class Ungrounded(StrEnum):
    """Why a finding was discarded. Four reasons, not one (AP-20): a worker
    citing a file it never saw, a line past the end of one it did, and a quote
    that is not in the file are different mistakes, and only the counts
    together say whether the check is too strict."""

    UNKNOWN_FILE = "cited a file that was not in its context"
    BAD_RANGE = "line range was not parseable"
    OUT_OF_RANGE = "line range is past the end of the file"
    EVIDENCE_NOT_FOUND = "quoted evidence does not appear in the cited lines"


@dataclass(frozen=True, slots=True)
class Dropped:
    finding: Finding
    reason: Ungrounded

    def render(self) -> str:
        return f"{self.finding.file}:{self.finding.lines} -- {self.reason}"


@dataclass(frozen=True, slots=True)
class Grounding:
    kept: tuple[Finding, ...]
    dropped: tuple[Dropped, ...]

    @property
    def reported(self) -> int:
        return len(self.kept) + len(self.dropped)

    @property
    def all_ungrounded(self) -> bool:
        """Every claim it made failed. Not the same as making no claims, and
        the difference decides whether a worker is `ok` or `degraded`."""
        return bool(self.dropped) and not self.kept


def _normalise(text: str) -> str:
    """Strip line-number prefixes and collapse whitespace.

    A model quoting numbered source may or may not include the prefix, and may
    re-indent. Neither is a grounding failure, and treating them as one would
    make the check fire on correct work -- which is how a check gets removed
    (playbook 5.2, Pattern 5).
    """
    return _WHITESPACE.sub(" ", _LINE_PREFIX.sub("", text)).strip()


def parse_range(raw: str) -> tuple[int, int] | None:
    match = _RANGE.match(raw)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    if start < 1 or end < start:
        return None
    return start, end


def verify(findings: list[Finding], chunks: list[Chunk]) -> Grounding:
    """Keep the findings whose citations check out; say why the rest did not.

    Chunks are what the worker ACTUALLY saw, after packing and truncation --
    not what it was meant to see. A finding citing a section that was dropped
    for budget is ungrounded from this worker's point of view, and that is the
    honest reading: it could not have read what it claims to quote.
    """
    by_path = {chunk.source_path: chunk for chunk in chunks}
    kept: list[Finding] = []
    dropped: list[Dropped] = []

    for finding in findings:
        chunk = by_path.get(finding.file.strip())
        if chunk is None:
            dropped.append(Dropped(finding, Ungrounded.UNKNOWN_FILE))
            continue

        body = chunk.body.splitlines()
        span = parse_range(finding.lines)
        if span is None:
            dropped.append(Dropped(finding, Ungrounded.BAD_RANGE))
            continue
        start, end = span
        if start > len(body):
            dropped.append(Dropped(finding, Ungrounded.OUT_OF_RANGE))
            continue

        # Verified against the whole chunk, not only the cited span. An
        # off-by-a-few line reference with a real quote is a citation error,
        # not a fabrication, and discarding it would lose a true finding to a
        # formatting mistake.
        if _normalise(finding.evidence) not in _normalise(chunk.body):
            dropped.append(Dropped(finding, Ungrounded.EVIDENCE_NOT_FOUND))
            continue

        kept.append(finding)

    return Grounding(kept=tuple(kept), dropped=tuple(dropped))
