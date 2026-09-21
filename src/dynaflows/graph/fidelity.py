"""Did the brief keep what the request said? §7, and the rule behind ADR-018.

The enhancer's system prompt has said "Preserve the user's intent exactly" and
"if the request is already precise, return it close to unchanged" since it was
written. On 2026-09-17 a 200-word request with four explicit constraints came
back as one sentence:

    "Modify repository.section_map() to enforce a token budget for the
     planner prompt."

Gone: which levers to use, which not to, the precedent to follow, and -- the
dangerous one -- "do not choose a budget value". The model that did it was a
different model from the one that had returned the same class of prompt
unchanged an hour earlier. **Brief quality tracks model identity, and asking
more firmly does not change that.** So this checks instead of asking.

It does NOT reject. A brief that drops something may still be right -- the
user may have been repetitive, or the enhancer may have folded two sentences
into one correctly. What it must never do is let a silent deletion through a
gate whose whole purpose is to show the human what will actually run. So the
concerns go on G1 next to the brief, and the human decides.

Deliberately dumb. No model call, no embedding, no semantic similarity: this
runs on every enhancement and its job is to be a tripwire, not a judge. Every
rule here answers a question with a checkable answer -- did this token survive,
did this negation survive -- because a fuzzy check that is usually right is a
check nobody believes when it fires.
"""

from __future__ import annotations

import re

__all__ = ["concerns_about"]

# A request long enough for compression to be a real risk. Below this, a short
# brief is just a short request and flagging it would fire on ordinary work --
# which is how a gate gets ignored (playbook 5.2, Pattern 5).
_MIN_LENGTH = 300

# Under half the original length is not an edit, it is a summary.
_SHRINK_RATIO = 0.5

# A sentence carrying one of these is the user ruling something OUT, and a
# dropped prohibition is worse than a dropped suggestion: the agent will not
# merely fail to do something, it will actively do the thing that was refused.
_PROHIBITION = re.compile(
    r"\b(do not|don't|never|must not|avoid|rather than|instead of)\b", re.IGNORECASE
)

# Things a request names that a brief cannot paraphrase away without losing
# the reference: dotted calls, dunder-ish private names, CONSTANTS, anchors,
# dates and figures. A word is not checked -- "budget" surviving proves
# nothing -- but `_FORMAL_RE` or `5,191` surviving proves the specific fact
# did.
#
# Two orderings here are load-bearing, and both were found by running the
# real 2026-09-17 case through this rather than by reading it:
#   - the ISO date alternative comes BEFORE the number one, or `2026-09-16`
#     is reported as three separate missing figures (2026, 09, 16) and the
#     list the human scans fills with fragments of one fact;
#   - the number alternative ends `(?:%|\b)` and not `%?\b`, because a `\b`
#     after `%` cannot hold against the following space, so the engine
#     backtracks and `54%` is reported as `54`.
# A concern list padded with noise is a concern list people stop reading.
_IDENTIFIER = re.compile(
    r"""(?x)
    \b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\(\)   # repository.section_map()
    | \b_[A-Za-z][A-Za-z0-9_]{2,}                          # _FORMAL_RE, _summary
    | \b[A-Z][A-Z0-9_]{4,}\b                               # SOURCE_MAP_BUDGET_TOKENS
    | \b(?:ADR|AP)-\d+\b                                   # ADR-009, AP-20
    | §\s?\d+(?:\.\d+)*                                    # §7, §4.5
    | \b\d{4}-\d{2}-\d{2}\b                                 # 2026-09-16
    | \b\d[\d,]*(?:\.\d+)?(?:%|\b)                          # 5,191  90  14%
    """
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _identifiers(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _IDENTIFIER.finditer(text):
        seen.setdefault(match.group(0).strip(), None)
    return list(seen)


def _prohibitions(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip() and _PROHIBITION.search(s)]


def concerns_about(raw_prompt: str, brief: str) -> list[str]:
    """What the brief appears to have dropped. Empty means nothing detected.

    "Nothing detected" is not "nothing lost" -- this reads tokens, not
    meaning. The list is evidence for a human at G1, never a verdict.
    """
    raw = (raw_prompt or "").strip()
    written = (brief or "").strip()
    if not raw or not written:
        return []

    concerns: list[str] = []

    if len(raw) >= _MIN_LENGTH and len(written) < len(raw) * _SHRINK_RATIO:
        shrink = 100 - round(100 * len(written) / len(raw))
        concerns.append(
            f"the brief is {shrink}% shorter than your request -- check that nothing "
            "you specified was summarised away"
        )

    dropped = [
        identifier
        for identifier in _identifiers(raw)
        if identifier.lower() not in written.lower()
    ]
    if dropped:
        shown = ", ".join(dropped[:8])
        more = f" (+{len(dropped) - 8} more)" if len(dropped) > 8 else ""
        concerns.append(f"named in your request but not in the brief: {shown}{more}")

    # Checked last and reported last because it is the one worth acting on.
    # A dropped prohibition does not make the agent fail to do something; it
    # makes the agent do the thing that was ruled out.
    for sentence in _prohibitions(raw):
        marker = next(
            (i for i in _identifiers(sentence) if i.lower() not in written.lower()), None
        )
        if marker is None and _PROHIBITION.search(written):
            continue
        concerns.append(f'a constraint may have been dropped: "{_clip(sentence)}"')

    return concerns


def _clip(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
