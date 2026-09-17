"""Is CHARS_PER_TOKEN_ESTIMATE right? Ask the provider, not a proxy. §7.

    uv run python scripts/calibrate_token_estimate.py [model-id]

`CHARS_PER_TOKEN_ESTIMATE = 3.6` has been load-bearing since Phase 0 and has
never been checked. Every context budget in this project is enforced against
it: ADR-021's worker budget, `SOURCE_MAP_BUDGET_TOKENS`, the G2 cost estimate
the human approves. If it is wrong, every one of those is wrong by the same
factor, silently, in the same direction.

**Why a real call rather than a tokenizer library.** `tiktoken` would measure
OpenAI's tokenizer, and this project routes through OpenRouter to whichever
model a tier resolves to. A proxy that is itself unvalidated would replace one
unchecked number with another, which is how §4.5 defines a placeholder. The
provider reports `usage.prompt_tokens` for the request it actually received --
that is the number the 402 ceiling is measured in, so it is the number worth
knowing.

**Cost:** a handful of calls with `max_tokens=1`. The prompt is what is being
measured; the completion is thrown away.

**The samples are real text from this repository**, not lorem ipsum. Token
density varies with content -- prose, code, and a catalogue of pipe-separated
rows are three different distributions, and an average over invented text
would be an average of nothing in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynaflows.contracts.tiers import Tier  # noqa: E402
from dynaflows.gateway.invoker import raw_completion  # noqa: E402
from dynaflows.gateway.registry import get_model_registry  # noqa: E402
from dynaflows.playbook import get_playbook_repository  # noqa: E402
from dynaflows.playbook.tokens import CHARS_PER_TOKEN_ESTIMATE, estimate_tokens  # noqa: E402
from dynaflows.settings import get_settings  # noqa: E402
from dynaflows.store.source_map import build_source_map  # noqa: E402


def _samples(settings: object, repository: object) -> list[tuple[str, str]]:
    sections = repository.section_map()  # type: ignore[attr-defined]
    source_map = build_source_map(settings.project_root).render()  # type: ignore[attr-defined]
    code = (
        Path(__file__).resolve().parents[1] / "src" / "dynaflows" / "executor" / "workspace.py"
    ).read_text(encoding="utf-8")
    prose = (
        Path(__file__).resolve().parents[1] / "docs" / "PROJECT_MEMORY.md"
    ).read_text(encoding="utf-8")
    return [
        ("section map (catalogue rows)", sections[:8000]),
        ("source map (paths + summaries)", source_map[:8000]),
        ("python source", code[:8000]),
        ("markdown prose", prose[:8000]),
    ]


def main(argv: list[str]) -> int:
    settings = get_settings()
    repository = get_playbook_repository()

    model = argv[0] if argv else get_model_registry(settings.models_config).chain(Tier.SMALL).preferred
    print(f"  measuring against {model}\n")

    rows: list[tuple[str, int, int, float, float]] = []
    for label, text in _samples(settings, repository):
        if not text.strip():
            continue
        status, body = raw_completion(settings, model, prompt=text, max_tokens=1)
        if status != 200 or isinstance(body, str):
            print(f"  {label}: call failed ({status}) -- {str(body)[:200]}")
            continue
        actual = int((body.get("usage") or {}).get("prompt_tokens") or 0)
        if actual <= 0:
            print(f"  {label}: the provider reported no prompt_tokens; nothing to compare")
            continue
        # The request carries a few tokens of chat scaffolding beyond our text.
        # Not corrected for: it is small against 2,000+ and correcting it with
        # a guessed constant would put an unmeasured number back into a
        # measurement taken to remove one.
        estimated = estimate_tokens(text)
        rows.append((label, estimated, actual, estimated / actual, len(text) / actual))

    if not rows:
        print("\n  No usable samples. Nothing is claimed.")
        return 1

    width = max(len(label) for label, *_ in rows)
    print(f"  {'sample'.ljust(width)}  {'ours':>7}  {'actual':>7}  {'ratio':>6}  chars/token")
    for label, estimated, actual, ratio, density in rows:
        print(f"  {label.ljust(width)}  {estimated:>7,}  {actual:>7,}  {ratio:>6.2f}  {density:.2f}")

    densities = [density for *_, density in rows]
    mean = sum(densities) / len(densities)
    print(
        f"\n  CHARS_PER_TOKEN_ESTIMATE is {CHARS_PER_TOKEN_ESTIMATE}; measured "
        f"{min(densities):.2f}-{max(densities):.2f}, mean {mean:.2f}"
    )

    # Per sample, not averaged. The first version printed ONE verdict from the
    # mean, one line after printing a caveat that a single constant cannot be
    # right for all content types -- and it said "conservative" while the
    # source map, the catalogue nearest its budget, was reading LOW. An
    # average that hides the one row you need is worse than no average.
    low = [(label, ratio) for label, _, _, ratio, _ in rows if ratio < 1.0]
    if low:
        print(
            "\n  UNDER-counted, which is the dangerous direction -- a prompt believed\n"
            "  to fit can be rejected by the provider, on a limit it was measured\n"
            "  against:"
        )
        for label, ratio in low:
            print(f"    · {label}: ours is {100 * (1 - ratio):.0f}% below the real count")
    high = [(label, ratio) for label, _, _, ratio, _ in rows if ratio >= 1.0]
    if high:
        print(
            "\n  OVER-counted -- conservative, but context is being dropped that\n"
            "  would have fit:"
        )
        for label, ratio in high:
            print(f"    · {label}: ours is {100 * (ratio - 1):.0f}% above the real count")

    print(
        f"\n  {len(rows)} sample(s), one call each, against ONE model. Different\n"
        "  providers tokenise differently, so this calibrates the model named\n"
        "  above and nothing else -- pass a model id to measure another. The\n"
        "  402 that prompted this came from the FRONTIER planner model, which is\n"
        "  not necessarily what was measured here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
