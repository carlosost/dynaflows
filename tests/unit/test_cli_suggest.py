"""`models --suggest` exists to be pasted into a TOML file.

That makes "the output parses as TOML" a contract, not a nicety -- and it is a
contract that was broken on the first real run: Rich read `[tiers.small]` as a
style tag and swallowed it, and turned a model id ending `:free` into an emoji,
because `:free:` is Rich's emoji shorthand. Both are the same mistake, which is
letting data pass through a markup renderer.

These tests run the command at a deliberately narrow width, because Rich also
hard-wraps to the terminal, which would split a model id across two lines.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from dynaflows.cli import app
from dynaflows.gateway.probe import ModelInfo

pytestmark = pytest.mark.deterministic

_CATALOGUE = [
    ModelInfo("nex-agi/nex-n2.5-mini:free", 262_000, 0.0, 0.0),
    ModelInfo("liquid/lfm-2.5-2.6b:free", 131_000, 0.0, 0.0),
    ModelInfo("mistralai/mistral-nemo", 131_000, 0.02, 0.03),
    ModelInfo("openai/gpt-5-nano", 400_000, 0.02, 0.20),
    ModelInfo("meta-llama/llama-4-scout", 10_000_000, 0.10, 0.30),
    ModelInfo("x-ai/grok-4.20", 2_000_000, 1.25, 2.50),
]


@pytest.fixture
def suggest_output(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr("dynaflows.gateway.probe.catalogue", lambda settings: _CATALOGUE)
    monkeypatch.setenv("COLUMNS", "60")  # deliberately narrow
    result = CliRunner().invoke(app, ["models", "--suggest"])
    assert result.exit_code == 0, result.output
    yield result.output


def test_suggest_output_is_valid_toml(suggest_output: str) -> None:
    tomllib.loads(suggest_output)


def test_every_tier_header_survives_rendering(suggest_output: str) -> None:
    """The regression: Rich consumed `[tiers.small]` as markup, so the pasted
    block was four bare `chain =` keys and TOML rejected it."""
    parsed = tomllib.loads(suggest_output)
    assert set(parsed["tiers"]) == {"small", "mid", "mid_high", "frontier"}


def test_model_ids_are_not_mangled_by_emoji_shorthand(suggest_output: str) -> None:
    """':free' at the end of an id is Rich's `:emoji:` syntax when followed by
    a colon. The literal id must appear."""
    parsed = tomllib.loads(suggest_output)
    assert "nex-agi/nex-n2.5-mini:free" in parsed["tiers"]["small"]["chain"]
    assert "🆓" not in suggest_output


def test_free_endpoints_are_excluded_from_the_worker_tier(suggest_output: str) -> None:
    """Workers run N times; a rate-limited free endpoint buys 429s, not
    savings (OQ-03)."""
    parsed = tomllib.loads(suggest_output)
    assert parsed["tiers"]["mid"]["chain"]
    assert not any(m.endswith(":free") for m in parsed["tiers"]["mid"]["chain"])


def test_verifier_tier_is_family_disjoint_from_the_worker_tier(suggest_output: str) -> None:
    """ADR-006 by construction, rather than by hoping the reader notices."""
    parsed = tomllib.loads(suggest_output)
    families = lambda ids: {m.split("/", 1)[0] for m in ids}  # noqa: E731
    assert not (
        families(parsed["tiers"]["mid"]["chain"]) & families(parsed["tiers"]["mid_high"]["chain"])
    )


def test_the_planner_tier_is_left_empty_for_a_human(suggest_output: str) -> None:
    """Refusing to choose is the feature. Price ranks cost, not reasoning."""
    parsed = tomllib.loads(suggest_output)
    assert parsed["tiers"]["frontier"]["chain"] == []
    assert "NOT SUGGESTED" in suggest_output


def test_long_lines_are_not_wrapped_at_narrow_widths(suggest_output: str) -> None:
    """Rich wraps to the terminal by default; a split model id breaks the paste."""
    for line in suggest_output.splitlines():
        if line.strip().startswith('"'):
            assert line.rstrip().endswith((",", "]")) or "#" in line, line
