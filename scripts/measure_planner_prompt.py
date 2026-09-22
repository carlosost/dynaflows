"""What the planner prompt is made of, and how close each part is to its budget. §7.

Run this whenever a 402 says the prompt is too large, instead of guessing:

    uv run python scripts/measure_planner_prompt.py

The composition was measured by hand twice (2026-09-15: 9,211 tokens; the
`change` run of 2026-09-16: 9,342). Both times the answer was the same part,
and both times the measurement was ad hoc and had to be rebuilt from scratch.
A number this project keeps needing is a script, not a memory.

It costs nothing: every figure here is produced locally from the indexed
corpus and the source tree. No provider call.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynaflows.graph.budgets import SECTION_MAP_BUDGET_TOKENS  # noqa: E402
from dynaflows.graph.capabilities import render_capabilities  # noqa: E402
from dynaflows.playbook import get_playbook_repository  # noqa: E402
from dynaflows.playbook.repository import _FORMAL_RE  # noqa: E402
from dynaflows.store.source_map import SOURCE_MAP_BUDGET_TOKENS  # noqa: E402
from dynaflows.playbook.tokens import estimate_tokens  # noqa: E402
from dynaflows.settings import get_settings  # noqa: E402
from dynaflows.store.source_map import build_source_map  # noqa: E402


def main() -> int:
    settings = get_settings()
    repository = get_playbook_repository()

    section_map = repository.section_map(SECTION_MAP_BUDGET_TOKENS)
    # `.render()`, NOT `str()`. The first version of this used `str()` on a
    # dataclass whose repr contains the rendered text AND a frozenset of
    # every path AND the field names -- roughly the prompt text twice over.
    # It reported the source map at 5,488 tokens and 91% of its budget, and
    # both figures were of an object the planner never sends. Measure what
    # goes on the wire, not what `print()` happens to produce.
    sections = section_map.render()
    source_map = build_source_map(settings.project_root)
    capabilities = render_capabilities("read")

    rows = [
        (
            "repository.section_map()",
            estimate_tokens(sections),
            f"{SECTION_MAP_BUDGET_TOKENS:,}",
        ),
        ("build_source_map()", estimate_tokens(source_map.render()), f"{SOURCE_MAP_BUDGET_TOKENS:,}"),
        ("render_capabilities()", estimate_tokens(capabilities), "n/a"),
    ]
    total = sum(count for _, count, _ in rows)

    width = max(len(name) for name, _, _ in rows)
    print(f"\n  {'part'.ljust(width)}  {'tokens':>8}  budget")
    for name, count, budget in rows:
        share = f"{100 * count / total:.0f}%" if total else "-"
        print(f"  {name.ljust(width)}  {count:>8,}  {budget:<8} {share}")
    print(f"  {'TOTAL (without the shell or the brief)'.ljust(width)}  {total:>8,}")

    # The FIRST version of this counted a row as anchored if it contained "#"
    # or a section sign anywhere. That is a lookalike for the project's rule,
    # not the rule, and it disagreed with an earlier hand count by a factor of
    # two while reading like a measurement. `_FORMAL_RE` is what
    # `repository.section_line` itself uses to decide, so this now asks the
    # same question the code asks. **When the codebase already owns the
    # predicate, importing it beats re-deriving something that resembles it.**
    # `.text`, not `.render()` -- the TRUNCATED banner is not a row and would
    # inflate both the row count and the anchor count if split on newlines.
    lines = [line for line in section_map.text.splitlines() if line.strip()]
    anchored = sum(1 for line in lines if _FORMAL_RE.match(line.split(" | ", 1)[0].strip()))
    print(f"\n  section map: {len(lines)} row(s), {anchored} with a formal anchor (AP-/ADR-/§)")
    print(
        f"  section map: {section_map.listed} of {section_map.total} rows listed"
        + (" -- TRUNCATED" if section_map.truncated else "")
    )

    used = estimate_tokens(source_map.render())
    print(
        f"  source map: {used:,} of {SOURCE_MAP_BUDGET_TOKENS:,} "
        f"({100 * used / SOURCE_MAP_BUDGET_TOKENS:.0f}% of its budget)"
    )
    print(
        "\n  Both catalogues are budgeted and both truncate the same way:\n"
        "  deterministically, formal-anchor rows (or the SourceMap's own\n"
        "  ordering) surviving first, with the cut stated in the rendered text\n"
        "  rather than left for the reader to infer from a shorter list. The\n"
        "  section map's cut also reaches gate G2 (`SectionMap.truncated`);\n"
        "  the source map's does not yet -- it still truncates silently there.\n"
        "\n  A cap on either is not free. ADR-009's premise is that the planner\n"
        "  names anchors from the section map, so truncation blinds the primary\n"
        "  retrieval path and has to reach the gate rather than the log.\n"
        "\n  These are ESTIMATES (CHARS_PER_TOKEN_ESTIMATE = 3.6). Checked twice:\n"
        "    - 2026-09-16, the real 402 on the real planner prompt: OpenRouter\n"
        "      counted 9,342 where this script implied ~10,160. About 8% high.\n"
        "    - scripts/calibrate_token_estimate.py against a SMALL-tier model:\n"
        "      3.49-4.08 chars/token, mean 3.82, spanning the constant.\n"
        "  So 3.6 is roughly right and errs conservative. It was carried as an\n"
        "  unvalidated guess since Phase 0; it is no longer one.\n"
        "\n  An earlier note here claimed ~21%. That was measuring str() on the\n"
        "  SourceMap dataclass -- its repr carries the rendered text AND a\n"
        "  frozenset of every path -- so the gap was the bug, not the estimator."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
