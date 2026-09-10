"""The gateway ladder. ADR-010 and ADR-013.

Every failure mode is driven with an injected invoker, an injected sleeper and
an injected clock. No network, no real waiting -- 2.2 says the deterministic
tier must never be skippable, and a suite that sleeps through backoff is a
suite someone eventually skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from dynaflows.contracts.calls import Budget, CallRequest, PriceBook, RawResponse
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.breaker import CircuitBreaker
from dynaflows.gateway.cache import ResponseCache, connect_cache
from dynaflows.gateway.client import GatewayClient
from dynaflows.gateway.registry import load_registry

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Answer(BaseModel):
    value: str


CONFIG = """
version = "t"
[constraints]
require_structured_outputs = true
enforce_verifier_family_diversity = false
min_context_tokens = 0
[tiers.small]
chain = ["acme/small"]
[tiers.mid]
chain = ["acme/one", "bolt/two", "cobalt/three"]
[tiers.mid_high]
chain = ["bolt/high"]
[tiers.frontier]
chain = ["cobalt/top"]
"""


@pytest.fixture
def registry(tmp_path: Path):  # noqa: ANN201
    path = tmp_path / "models.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return load_registry(path)


class Recorder:
    """An invoker that plays a scripted sequence and records what it was asked."""

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, CallRequest]] = []

    async def __call__(self, model_id: str, request: CallRequest) -> RawResponse:
        self.calls.append((model_id, request))
        item = self.script.pop(0) if self.script else self.script_default()
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def script_default(self) -> RawResponse:
        return RawResponse(text='{"value": "ok"}', tokens_in=10, tokens_out=5)

    @property
    def models_tried(self) -> list[str]:
        return [m for m, _ in self.calls]


def client(registry, invoker, **kwargs):  # noqa: ANN001, ANN201
    slept: list[float] = []

    async def sleeper(delay: float) -> None:
        slept.append(delay)

    gateway = GatewayClient(
        registry=registry,
        invoker=invoker,
        sleeper=sleeper,
        jitter=lambda: 1.0,
        **kwargs,
    )
    gateway.slept = slept  # type: ignore[attr-defined]
    return gateway


def rate_limit() -> DynaflowsError:
    return DynaflowsError.of(ErrorCode.RATE_LIMIT, "429")


REQUEST = CallRequest(tier=Tier.MID, prompt="hello", schema=Answer)


# --- happy path ----------------------------------------------------------


async def test_a_successful_call_returns_the_parsed_payload(registry) -> None:  # noqa: ANN001
    gateway = client(registry, Recorder())
    result = await gateway.call(REQUEST)
    assert isinstance(result.payload, Answer)
    assert result.payload.value == "ok"
    assert result.model_id == "acme/one"
    assert result.fallback_depth == 0
    assert result.attempts == 1


async def test_tokens_and_calls_are_recorded_in_the_ledger(registry) -> None:  # noqa: ANN001
    gateway = client(registry, Recorder())
    await gateway.call(REQUEST)
    assert gateway.ledger.calls_made == 1
    assert gateway.ledger.tokens_in == 10
    assert gateway.ledger.tokens_out == 5


# --- layer 6: retry ------------------------------------------------------


async def test_a_rate_limit_is_retried_on_the_same_model(registry) -> None:  # noqa: ANN001
    recorder = Recorder(rate_limit(), rate_limit(), RawResponse(text='{"value":"ok"}'))
    gateway = client(registry, recorder)
    result = await gateway.call(REQUEST)
    assert result.model_id == "acme/one"
    assert result.attempts == 3
    assert recorder.models_tried == ["acme/one"] * 3


async def test_backoff_grows_and_is_jittered_not_fixed(registry) -> None:  # noqa: ANN001
    """Equal-spaced backoff across a fan-out re-synchronises the herd it was
    supposed to disperse."""
    gateway = client(
        registry, Recorder(rate_limit(), rate_limit(), RawResponse(text='{"value":"o"}'))
    )
    await gateway.call(REQUEST)
    assert gateway.slept == [2.0, 4.0]  # jitter stubbed to 1.0


async def test_a_schema_failure_is_not_retried_on_the_same_model(registry) -> None:  # noqa: ANN001
    """Re-asking the same model the same question rarely produces different
    JSON. Layer 8 handles it by including the validation error instead."""
    recorder = Recorder(RawResponse(text="not json"), RawResponse(text='{"value":"fixed"}'))
    gateway = client(registry, recorder)
    result = await gateway.call(REQUEST)
    assert recorder.models_tried == ["acme/one", "acme/small"]
    assert result.repaired is True


# --- layer 7: fallback ---------------------------------------------------


async def test_an_exhausted_model_falls_through_to_the_next_in_the_chain(registry) -> None:  # noqa: ANN001
    recorder = Recorder(*[rate_limit()] * 3, RawResponse(text='{"value":"ok"}'))
    gateway = client(registry, recorder)
    result = await gateway.call(REQUEST)
    assert result.model_id == "bolt/two"
    assert result.fallback_depth == 1
    assert gateway.ledger.fallbacks == 1


async def test_exhausting_every_model_raises_the_last_error(registry) -> None:  # noqa: ANN001
    gateway = client(registry, Recorder(*[rate_limit()] * 9))
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert excinfo.value.envelope.code is ErrorCode.RATE_LIMIT


async def test_a_fatal_error_aborts_immediately_instead_of_walking_the_chain(
    registry,  # noqa: ANN001
) -> None:
    """A 401 is not about this model. Falling through repeats it once per
    chain entry and reports the last one instead of the real one -- with a
    3-model chain in a 12-way fan-out, 36 doomed requests before anything
    says "401". This test found exactly that bug."""
    recorder = Recorder(DynaflowsError.of(ErrorCode.AUTH_FAILED, "401"))
    gateway = client(registry, recorder)
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert excinfo.value.envelope.code is ErrorCode.AUTH_FAILED
    assert len(recorder.calls) == 1


async def test_an_empty_chain_is_a_config_error_naming_the_fix(registry) -> None:  # noqa: ANN001
    object.__setattr__(registry.chain(Tier.MID), "chain", ())
    gateway = client(registry, Recorder())
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert "models --suggest" in excinfo.value.envelope.message


# --- layer 3: circuit breaker -------------------------------------------


async def test_a_broken_model_is_skipped_without_being_called(registry) -> None:  # noqa: ANN001
    now = [0.0]
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, now=lambda: now[0])
    breaker.record_failure("acme/one")
    recorder = Recorder(RawResponse(text='{"value":"ok"}'))
    gateway = client(registry, recorder, breaker=breaker)
    result = await gateway.call(REQUEST)
    assert recorder.models_tried == ["bolt/two"]
    assert result.model_id == "bolt/two"


async def test_the_breaker_reopens_the_model_after_the_cooldown(registry) -> None:  # noqa: ANN001
    now = [0.0]
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, now=lambda: now[0])
    breaker.record_failure("acme/one")
    assert breaker.is_open("acme/one")
    now[0] = 61.0
    assert not breaker.is_open("acme/one")


async def test_a_success_clears_the_failure_streak(registry) -> None:  # noqa: ANN001
    breaker = CircuitBreaker(threshold=2, now=lambda: 0.0)
    breaker.record_failure("acme/one")
    breaker.record_success("acme/one")
    breaker.record_failure("acme/one")
    assert not breaker.is_open("acme/one")


async def test_a_schema_failure_does_not_open_the_breaker(registry) -> None:  # noqa: ANN001
    """The model answered; it just answered badly. That is not the model being
    down, and treating it as such would take a healthy model out of rotation."""
    breaker = CircuitBreaker(threshold=1, now=lambda: 0.0)
    gateway = client(
        registry,
        Recorder(RawResponse(text="nope"), RawResponse(text='{"value":"fixed"}')),
        breaker=breaker,
    )
    await gateway.call(REQUEST)
    assert not breaker.is_open("acme/one")


# --- layer 1: budget -----------------------------------------------------


async def test_the_budget_halts_the_next_call_with_a_typed_error(registry) -> None:  # noqa: ANN001
    prices = PriceBook(prompt={"acme/one": 1000.0}, completion={"acme/one": 1000.0})
    gateway = client(registry, Recorder(), prices=prices, budget=Budget(usd_limit=0.001))
    await gateway.call(REQUEST)
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert excinfo.value.envelope.code is ErrorCode.BUDGET_EXCEEDED


async def test_a_token_ceiling_works_without_any_price_data(registry) -> None:  # noqa: ANN001
    """Without prices usd_spent stays 0.0, which is honest. A fabricated price
    would make the ledger confidently wrong."""
    gateway = client(registry, Recorder(), budget=Budget(token_limit=10))
    await gateway.call(REQUEST)
    assert gateway.ledger.usd_spent == 0.0
    with pytest.raises(DynaflowsError):
        await gateway.call(REQUEST)


# --- layer 2: the cache (ADR-013) ---------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(connect_cache(tmp_path / "calls.db"))


async def test_an_identical_call_is_served_from_cache_without_the_provider(
    registry, cache: ResponseCache
) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text='{"value":"ok"}', tokens_in=10, tokens_out=5))
    gateway = client(registry, recorder, cache=cache)
    await gateway.call(REQUEST)
    second = await gateway.call(REQUEST)
    assert len(recorder.calls) == 1
    assert second.cache_hit is True
    assert second.payload.value == "ok"


async def test_a_cache_hit_counts_as_money_avoided_not_money_spent(
    registry, cache: ResponseCache
) -> None:  # noqa: ANN001
    """AP-20: one number answers neither 'what did this run cost' nor 'what
    would it have cost cold'."""
    prices = PriceBook(prompt={"acme/one": 1000.0}, completion={"acme/one": 1000.0})
    gateway = client(registry, Recorder(), cache=cache, prices=prices)
    await gateway.call(REQUEST)
    spent = gateway.ledger.usd_spent
    await gateway.call(REQUEST)
    assert gateway.ledger.usd_spent == spent
    assert gateway.ledger.usd_avoided == pytest.approx(spent)
    assert gateway.ledger.calls_cached == 1


async def test_a_different_variant_is_a_different_call(registry, cache: ResponseCache) -> None:  # noqa: ANN001
    """ADR-013.3: this is how Phase 5 asks for sampling diversity without
    turning the cache off."""
    recorder = Recorder(RawResponse(text='{"value":"a"}'), RawResponse(text='{"value":"b"}'))
    gateway = client(registry, recorder, cache=cache)
    first = await gateway.call(REQUEST)
    second = await gateway.call(
        CallRequest(tier=Tier.MID, prompt="hello", schema=Answer, variant=1)
    )
    assert len(recorder.calls) == 2
    assert (first.payload.value, second.payload.value) == ("a", "b")


async def test_a_changed_prompt_misses_the_cache(registry, cache: ResponseCache) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text='{"value":"a"}'), RawResponse(text='{"value":"b"}'))
    gateway = client(registry, recorder, cache=cache)
    await gateway.call(REQUEST)
    await gateway.call(CallRequest(tier=Tier.MID, prompt="different", schema=Answer))
    assert len(recorder.calls) == 2


async def test_a_rate_limit_is_never_cached(registry, cache: ResponseCache) -> None:  # noqa: ANN001
    """ADR-013.5. Caching a transient error makes it permanent."""
    gateway = client(registry, Recorder(*[rate_limit()] * 9), cache=cache)
    with pytest.raises(DynaflowsError):
        await gateway.call(REQUEST)
    assert cache.count() == 0


async def test_a_schema_failure_is_never_cached(registry, cache: ResponseCache) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text="not json"), RawResponse(text='{"value":"fixed"}'))
    gateway = client(registry, recorder, cache=cache)
    await gateway.call(REQUEST)
    assert cache.count() == 1  # only the repaired, validated response
    assert "fixed" in (cache.get(list_keys(cache)[0]).response.text)


def list_keys(cache: ResponseCache) -> list[str]:
    return [r["key"] for r in cache._connection.execute("SELECT key FROM calls")]  # noqa: SLF001


async def test_a_corrupt_cache_row_is_treated_as_a_miss_not_a_failure(
    registry, cache: ResponseCache
) -> None:  # noqa: ANN001
    """A corrupt row must never be able to end a run."""
    key = REQUEST.cache_key("acme/one")
    cache.put(
        key,
        model_id="acme/one",
        variant=0,
        response=RawResponse(text="{ this is not json"),
        cost_usd=0.0,
    )
    gateway = client(registry, Recorder(RawResponse(text='{"value":"ok"}')), cache=cache)
    result = await gateway.call(REQUEST)
    assert result.cache_hit is False
    assert result.payload.value == "ok"


async def test_clearing_the_cache_forces_a_fresh_call(registry, cache: ResponseCache) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text='{"value":"a"}'), RawResponse(text='{"value":"b"}'))
    gateway = client(registry, recorder, cache=cache)
    await gateway.call(REQUEST)
    assert cache.clear() == 1
    await gateway.call(REQUEST)
    assert len(recorder.calls) == 2


# --- layer 8: repair -----------------------------------------------------


async def test_the_repair_call_runs_at_small_tier_not_the_original_tier(registry) -> None:  # noqa: ANN001
    """Repairing JSON is a formatting task, not a reasoning one."""
    recorder = Recorder(RawResponse(text="junk"), RawResponse(text='{"value":"fixed"}'))
    gateway = client(registry, recorder, budget=Budget())
    await gateway.call(REQUEST)
    assert recorder.calls[1][0] == "acme/small"
    assert recorder.calls[1][1].tier is Tier.SMALL


async def test_only_one_repair_is_attempted(registry) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text="junk"), RawResponse(text="still junk"))
    gateway = client(registry, recorder)
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert excinfo.value.envelope.code is ErrorCode.SCHEMA_INVALID
    assert len(recorder.calls) == 2


async def test_a_schema_failure_is_counted_separately_from_a_skip(registry) -> None:  # noqa: ANN001
    recorder = Recorder(RawResponse(text="junk"), RawResponse(text='{"value":"fixed"}'))
    gateway = client(registry, recorder)
    await gateway.call(REQUEST)
    assert gateway.ledger.schema_failures == 1
    assert gateway.ledger.calls_skipped == 0


async def test_a_request_without_a_schema_returns_text_unparsed(registry) -> None:  # noqa: ANN001
    gateway = client(registry, Recorder(RawResponse(text="plain prose")))
    result = await gateway.call(CallRequest(tier=Tier.MID, prompt="hi"))
    assert result.payload == "plain prose"


# --- layer 4/5: concurrency and timeout ---------------------------------


async def test_the_semaphore_bounds_requests_in_flight(registry) -> None:  # noqa: ANN001
    import asyncio

    peak = 0
    live = 0

    async def invoker(model_id: str, request: CallRequest) -> RawResponse:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return RawResponse(text='{"value":"ok"}')

    gateway = client(registry, invoker, max_concurrent=2)
    await asyncio.gather(*(gateway.call(REQUEST) for _ in range(8)))
    assert peak <= 2


async def test_a_hanging_provider_becomes_a_typed_timeout(registry) -> None:  # noqa: ANN001
    import asyncio

    async def invoker(model_id: str, request: CallRequest) -> RawResponse:
        await asyncio.sleep(10)
        return RawResponse(text="never")

    gateway = client(registry, invoker, timeout_seconds=0.01, max_attempts=1)
    with pytest.raises(DynaflowsError) as excinfo:
        await gateway.call(REQUEST)
    assert excinfo.value.envelope.code is ErrorCode.TIMEOUT
