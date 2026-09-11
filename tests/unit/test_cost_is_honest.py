"""A wrong number is harder to notice than an error.

A live run reported "$0.0000 spent, 1 call(s)" for a planner call OpenRouter
itself priced at $0.00501886. Nothing failed: `PriceBook` is empty in every
code path in `src/`, so `cost()` returned 0.0, and 0.0 formats as $0.0000.
There was no error to see, no test to fail, and the only symptom was a number
that looked like good news.

These tests hold the three rules that came out of it:
  1. the provider's own cost is used when it sends one,
  2. an unknown cost is None, never 0.0,
  3. the report says so out loud.
"""

from __future__ import annotations

from typing import Any

from dynaflows.contracts.calls import CallResult, CostLedger, PriceBook, RawResponse
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.invoker import _usage


class _Message:
    def __init__(self, usage: dict[str, Any] | None, metadata: dict[str, Any] | None) -> None:
        self.usage_metadata = usage
        self.response_metadata = metadata


def _result(cost: float | None, *, cache_hit: bool = False) -> CallResult:
    return CallResult(
        payload=None,
        model_id="vendor/model",
        tier=Tier.FRONTIER,
        raw="{}",
        tokens_in=19_710,
        tokens_out=2_120,
        cost_usd=cost,
        fallback_depth=0,
        attempts=1,
        cache_hit=cache_hit,
    )


def test_the_providers_own_cost_is_read_off_the_message() -> None:
    """The exact shape OpenRouter returned on the failing run: LangChain passes
    the provider's `usage` block through to response_metadata whole."""
    message = _Message(
        {"input_tokens": 19_710, "output_tokens": 2_120},
        {"token_usage": {"prompt_tokens": 19_710, "completion_tokens": 2_120, "cost": 0.00501886}},
    )

    assert _usage(message)["cost_usd"] == 0.00501886


def test_a_provider_that_reports_no_cost_yields_none_not_zero() -> None:
    message = _Message({"input_tokens": 10, "output_tokens": 2}, {"token_usage": {}})

    assert _usage(message)["cost_usd"] is None


def test_a_free_model_reporting_zero_is_known_not_unknown() -> None:
    """`:free` models report cost 0. That is a fact, not an absence, and it
    must not be counted as unpriced."""
    message = _Message({"input_tokens": 120, "output_tokens": 400}, {"token_usage": {"cost": 0}})

    assert _usage(message)["cost_usd"] == 0.0


def test_an_unpriced_call_is_counted_rather_than_added_as_zero() -> None:
    ledger = CostLedger().record(_result(None))

    assert ledger.calls_unpriced == 1
    assert ledger.usd_spent == 0.0
    assert ledger.calls_made == 1
    # The tokens are known even when the money is not; they are what makes the
    # missing price visible instead of plausible.
    assert ledger.tokens_in == 19_710


def test_a_priced_call_does_not_count_as_unpriced() -> None:
    ledger = CostLedger().record(_result(0.005))

    assert ledger.calls_unpriced == 0
    assert ledger.usd_spent == 0.005


def test_an_unpriced_cache_hit_is_also_counted() -> None:
    """usd_avoided has the same failure mode as usd_spent (AP-20: two counters,
    two facts). A cache hit on an unpriced row avoids an unknown amount."""
    ledger = CostLedger().record(_result(None, cache_hit=True))

    assert ledger.calls_unpriced == 1
    assert ledger.usd_avoided == 0.0
    assert ledger.calls_cached == 1


def test_the_price_book_is_still_the_fallback_when_it_has_the_model() -> None:
    from dynaflows.gateway.client import GatewayClient
    from dynaflows.gateway.registry import ModelRegistry

    client = GatewayClient(
        registry=ModelRegistry(
            version="t",
            tiers={},
            require_structured_outputs=True,
            enforce_verifier_family_diversity=False,
            min_context_tokens=0,
        ),
        invoker=None,
        prices=PriceBook(prompt={"vendor/model": 1.0}, completion={"vendor/model": 2.0}),
    )
    response = RawResponse(text="{}", tokens_in=1_000_000, tokens_out=1_000_000)

    assert client._cost_of("vendor/model", response) == 3.0  # noqa: SLF001
    # A model the book does not know is unknown, not free.
    assert client._cost_of("other/model", response) is None  # noqa: SLF001


def test_the_provider_wins_over_the_price_book() -> None:
    """The provider's figure knows about markup, the cached-prompt discount and
    which upstream served the call. A local table knows none of that."""
    from dynaflows.gateway.client import GatewayClient
    from dynaflows.gateway.registry import ModelRegistry

    client = GatewayClient(
        registry=ModelRegistry(
            version="t",
            tiers={},
            require_structured_outputs=True,
            enforce_verifier_family_diversity=False,
            min_context_tokens=0,
        ),
        invoker=None,
        prices=PriceBook(prompt={"vendor/model": 999.0}),
    )
    response = RawResponse(text="{}", tokens_in=10, tokens_out=10, cost_usd=0.004)

    assert client._cost_of("vendor/model", response) == 0.004  # noqa: SLF001


def test_the_report_refuses_to_imply_a_total_it_does_not_have() -> None:
    from dynaflows.cli import _money

    priced = _money(CostLedger(usd_spent=0.005, calls_made=1))
    assert "LOWER BOUND" not in priced

    unpriced = _money(CostLedger(usd_spent=0.0, calls_made=1, calls_unpriced=1))
    assert "LOWER BOUND" in unpriced
    assert "1 unpriced" in unpriced
