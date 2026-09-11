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
    NOT_A_DEFECT = "describes the code rather than reporting a problem with it"


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


# A line short enough or punctuation-only enough that its presence proves
# nothing. A model quoting a docstring's first line often closes the quote
# itself with `\"\"\"`, which is a syntactic completion rather than a claim
# about the file, and rejecting the whole citation over it discards a correct
# finding.
# Decoration at the ends of a quoted line: string delimiters, brackets,
# commas, ellipses. Stripped from the QUOTE before matching, never from the
# file -- the file is the authority and must not be loosened.
_EDGE_NOISE = re.compile(r"^[^0-9A-Za-z_]+|[^0-9A-Za-z_)\]}:]+$")

# A quote is split on newlines AND on elisions. Run `s5`: a model wrote
# `def check_langsmith(...) -> TelemetryStatus: ... return TelemetryStatus(...)`
# joining two real but non-adjacent fragments. Both exist in the file; the
# combined string does not. An elision is the model saying "these two real
# fragments, with something between", which is a true statement about the file.
_ELLIPSIS = re.compile(r"\n|\.{3,}|…")

# Below this, a matching line proves nothing: `)` or `else:` appear everywhere.
# PLACEHOLDER (playbook 4.5) -- the drop rate this produces is the measurement
# that should set it.
_MIN_CORE = 12

# Shorter than this and a "failure scenario" is a label, not a scenario.
_MIN_FAILURE = 15

# Phrases that mean "there is nothing wrong here". A finding whose remediation
# is one of these is a description, not a finding.
_NO_REMEDY = frozenset(
    {
        "",
        "none",
        "n/a",
        "na",
        "no remediation needed",
        "no remediation required",
        "no remediation",
        "no action needed",
        "no action required",
        "no change needed",
        "no changes needed",
        "not applicable",
        "none needed",
        "none required",
    }
)


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
        if not _evidence_present(finding.evidence, chunk.body):
            dropped.append(Dropped(finding, Ungrounded.EVIDENCE_NOT_FOUND))
            continue

        if _is_description(finding):
            # Grounded and worthless. Run `s4` produced three of these: real
            # file, real lines, verbatim quote, severity "low", remediation
            # "No remediation needed". Every structural check passed and the
            # synthesis summarised what the code does. Verification that
            # validates form and not substance is a rubber stamp with extra
            # steps.
            dropped.append(Dropped(finding, Ungrounded.NOT_A_DEFECT))
            continue

        kept.append(finding)

    return Grounding(kept=tuple(kept), dropped=tuple(dropped))


def _evidence_present(evidence: str, body: str) -> bool:
    """Whether the quote is really in the file, allowing cosmetic additions.

    Exact substring matching was the first implementation and it is brittle in
    one specific direction. A model quoting the first line of a MULTI-line
    docstring closes it with a delimiter that is not in the source:

        source   \"\"\"Every command configures tracing. ADR-011.
                 (blank)
                 More prose.
                 \"\"\"
        quote    \"\"\"Every command configures tracing. ADR-011.\"\"\"

    That is a real quote with a syntactic completion attached, and five of
    eight findings in run `s4` were discarded for it. Every one cited a line
    that existed.

    So each line is compared by its alphanumeric CORE, with decoration at the
    ends stripped from the quote only. A fabricated line still matches nothing;
    a real line survives an added delimiter, a trailing comma or an ellipsis.
    Lines whose core is too short to prove anything are skipped, and a quote
    made entirely of those falls back to the strict test.
    """
    haystack = _normalise(body)
    cores = [_core(part) for part in _ELLIPSIS.split(evidence)]
    substantial = [core for core in cores if len(core) >= _MIN_CORE]
    if not substantial:
        # Nothing substantial to check. A quote of `)` is in every Python file
        # and proves nothing about whether the model read this one, so the
        # fallback demands the same minimum before it can accept.
        strict = _normalise(evidence)
        return len(strict) >= _MIN_CORE and strict in haystack
    return all(core in haystack for core in substantial)


def _core(line: str) -> str:
    """The line without its line number or whatever decorates its ends.

    The prefix strip has to happen here as well as in `_normalise`: the worker
    is SHOWN numbered lines and told to copy verbatim, so `2|     if not user:`
    is the compliant answer and must not be a grounding failure.
    """
    return _EDGE_NOISE.sub("", _WHITESPACE.sub(" ", _LINE_PREFIX.sub("", line))).strip()


def _is_description(finding: Finding) -> bool:
    """A finding that names no failure, or proposes no fix, is a description.

    Run `s5` claimed fifteen findings. "Telemetry errors are detected and
    classified", "Settings are properly configured for telemetry", "Telemetry
    is configured before any provider call exists" -- correct statements about
    working code, correctly cited, and worth nothing. Citation checking proves
    a model READ the file; only this asks whether it found anything.

    Both fields are checked because either alone is easy to satisfy by
    accident: a description can carry a plausible-sounding remediation, and a
    real defect can have an obvious fix stated in three words.
    """
    remedy = _WHITESPACE.sub(" ", finding.remediation).strip().rstrip(".").lower()
    if remedy in _NO_REMEDY:
        return True
    failure = _WHITESPACE.sub(" ", finding.failure).strip().rstrip(".").lower()
    # PLACEHOLDER (playbook 4.5). "Returns None on a 404" is 21 characters and
    # is a real failure scenario; the threshold is set below that on purpose,
    # because a gate that fires on ordinary work gets disabled.
    return failure in _NO_REMEDY or len(failure) < _MIN_FAILURE
