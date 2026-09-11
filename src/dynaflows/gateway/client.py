"""The gateway ladder. ADR-010, and ADR-013's cache.

Order is not arbitrary. Each layer only makes sense once the one above it has
run, and the whole point of a single chokepoint is that this order exists in
exactly one place:

  1 budget      -- stop before spending more
  2 cache       -- do not pay twice for the same call (ADR-013)
  3 breaker     -- skip a model already known to be down
  4 semaphore   -- bound requests per API key, process-global
  5 timeout     -- no unbounded provider call exists in this codebase
  6 retry       -- backoff with full jitter, honouring Retry-After
  7 fallback    -- advance to the next capability-homogeneous model
  8 repair      -- ONE cheap re-ask on a schema failure, then give up

Everything injectable is injected -- invoker, sleeper, clock, cache, registry.
A ladder tested through real HTTP and real sleeps is a ladder whose tests are
too slow to run, and 2.2 says the deterministic tier must never be skippable.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any

from pydantic import ValidationError

from dynaflows.contracts.calls import (
    Budget,
    CallRequest,
    CallResult,
    CostLedger,
    PriceBook,
    RawResponse,
)
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.breaker import CircuitBreaker
from dynaflows.gateway.cache import CacheBackend, NullCache
from dynaflows.gateway.registry import ModelRegistry

Invoker = Callable[[str, CallRequest], Awaitable[RawResponse]]
Sleeper = Callable[[float], Awaitable[None]]

# Which failures are worth another attempt. A schema failure is NOT here:
# re-asking the same model the same question rarely produces different JSON,
# and layer 8 handles it properly by including the validation error.
_RETRYABLE = frozenset({ErrorCode.RATE_LIMIT, ErrorCode.TIMEOUT, ErrorCode.MODEL_UNAVAILABLE})

# Failures that are NOT about this model, so falling through the chain repeats
# them once per entry and reports the last one instead of the real one. A bad
# API key is the clearest case: with a 3-model chain in a 12-way fan-out it
# would fire 36 doomed requests before anything says "401".
_FATAL = frozenset({ErrorCode.AUTH_FAILED, ErrorCode.CONFIG_INVALID, ErrorCode.BUDGET_EXCEEDED})


@dataclass
class GatewayClient:
    registry: ModelRegistry
    invoker: Invoker
    cache: CacheBackend = field(default_factory=NullCache)
    prices: PriceBook = field(default_factory=PriceBook)
    budget: Budget = field(default_factory=Budget)
    ledger: CostLedger = field(default_factory=CostLedger)
    max_concurrent: int = 6
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    # Providers price a request as prompt + the FULL output allowance, so an
    # unset max_tokens reserves the model's ceiling -- 65,536 on the model that
    # found this. The result was a 402 for a request that would have used a
    # fraction of it. A cap is applied HERE so a call site that forgets one
    # cannot reserve the maximum. PLACEHOLDER (playbook 4.5).
    default_max_tokens: int = 4096
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    sleeper: Sleeper = asyncio.sleep
    jitter: Callable[[], float] = random.random
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False)

    def _sem(self) -> asyncio.Semaphore:
        # Created lazily and shared: this is the limit the provider actually
        # enforces (per key, across the process). LangGraph's max_concurrency
        # bounds BRANCHES, which is a different question (ADR-014).
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    # -- layer 6 ----------------------------------------------------------
    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        # Full jitter: exponential ceiling, uniform below it. Equal-spaced
        # backoff across a 12-way fan-out re-synchronises the herd it was
        # meant to disperse.
        return self.jitter() * min(2.0**attempt, 30.0)

    async def _attempt_model(self, model_id: str, request: CallRequest) -> tuple[RawResponse, int]:
        """Layers 4-6 for one model. Raises the last error if all attempts fail."""
        last: DynaflowsError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with self._sem():
                    return (
                        await asyncio.wait_for(
                            self.invoker(model_id, request), timeout=self.timeout_seconds
                        ),
                        attempt,
                    )
            except TimeoutError:
                last = DynaflowsError.of(
                    ErrorCode.TIMEOUT,
                    f"{model_id}: no response within {self.timeout_seconds}s",
                    attempt,
                )
            except DynaflowsError as exc:
                last = exc
                if exc.envelope.code not in _RETRYABLE:
                    raise
            if attempt < self.max_attempts:
                retry_after = getattr(last, "retry_after", None)
                await self.sleeper(self._backoff(attempt, retry_after))
        assert last is not None
        raise last

    # -- layer 8 ----------------------------------------------------------
    def _parse(self, request: CallRequest, text: str) -> Any:
        if request.schema is None:
            return text
        try:
            return request.schema.model_validate_json(text)
        except ValidationError as exc:
            raise DynaflowsError.of(ErrorCode.SCHEMA_INVALID, str(exc)) from exc

    def _repair_request(self, request: CallRequest, error: str) -> CallRequest:
        """One re-ask, at SMALL tier. Repairing JSON is a formatting task, not
        a reasoning one, so paying frontier prices for it is waste."""
        schema = request.schema.model_json_schema() if request.schema else {}
        return CallRequest(
            tier=Tier.SMALL,
            prompt=(
                "The following text was supposed to satisfy a JSON schema and did not.\n"
                "Return ONLY corrected JSON. No prose, no code fence.\n\n"
                f"SCHEMA:\n{json.dumps(schema)}\n\nTEXT:\n{request.prompt}\n\nERROR:\n{error}"
            ),
            schema=request.schema,
            variant=request.variant,
            label=f"{request.label}.repair",
        )

    # -- the ladder -------------------------------------------------------
    async def call(self, request: CallRequest, *, _allow_repair: bool = True) -> CallResult:
        if request.max_tokens is None:
            # Applied before the cache key is computed, so the key and the
            # request that produced it always agree.
            request = dc_replace(request, max_tokens=self.default_max_tokens)

        if reason := self.budget.exceeded_by(self.ledger):
            # Halts to a resumable checkpoint, not a crash (ADR-010 layer 1).
            raise DynaflowsError.of(ErrorCode.BUDGET_EXCEEDED, reason)

        chain = self.registry.chain(request.tier).chain
        if not chain:
            raise DynaflowsError.of(
                ErrorCode.CONFIG_INVALID,
                f"tier '{request.tier}' has an empty chain; run `dynaflows models --suggest`",
            )

        last: DynaflowsError | None = None
        skipped: list[str] = []
        for depth, model_id in enumerate(chain):
            key = request.cache_key(model_id)
            if (hit := self.cache.get(key)) is not None:
                try:
                    cached_payload = self._parse(request, hit.response.text)
                except DynaflowsError:
                    # A cached row that no longer validates is treated as a
                    # MISS, not a failure. The key includes the schema, so this
                    # can only mean the row is corrupt -- and a corrupt cache
                    # row must never be able to end a run.
                    pass
                else:
                    result = self._result(
                        request,
                        model_id,
                        hit.response,
                        depth,
                        0,
                        hit.cost_usd,
                        cached_payload,
                        cache_hit=True,
                    )
                    self.ledger = self.ledger.record(result)
                    return result

            if self.breaker.is_open(model_id):
                skipped.append(model_id)
                continue

            try:
                response, attempts = await self._attempt_model(model_id, request)
            except DynaflowsError as exc:
                if exc.envelope.code in _FATAL:
                    raise
                last = exc
                if exc.envelope.code in _RETRYABLE:
                    self.breaker.record_failure(model_id)
                continue

            cost = self._cost_of(model_id, response)
            try:
                payload = self._parse(request, response.text)
            except DynaflowsError as exc:
                # A schema failure is the model's output, not the model being
                # down: it does not open the breaker and does not fall through
                # to the next model, because layer 8 is the right response.
                # The tokens were still spent, so they are still recorded.
                self.ledger = self.ledger.record(
                    self._result(request, model_id, response, depth, attempts, cost, response.text)
                )
                self.ledger = _bump(self.ledger, "schema_failures")
                if not _allow_repair:
                    raise
                repaired = await self.call(
                    self._repair_request(request, exc.envelope.message), _allow_repair=False
                )
                return _with(repaired, repaired_flag=True, fallback_depth=depth)

            self.breaker.record_success(model_id)
            # ADR-013.5/.6: written here -- validated success only, inside the
            # gateway, BEFORE the caller returns and long before a checkpoint.
            self.cache.put(
                key, model_id=model_id, variant=request.variant, response=response, cost_usd=cost
            )
            result = self._result(request, model_id, response, depth, attempts, cost, payload)
            self.ledger = self.ledger.record(result)
            return result

        if skipped:
            self.ledger = _bump(self.ledger, "calls_skipped")
        raise last or DynaflowsError.of(
            ErrorCode.MODEL_UNAVAILABLE,
            f"every model in tier '{request.tier}' is circuit-broken: {', '.join(skipped)}",
        )

    def _cost_of(self, model_id: str, response: RawResponse) -> float | None:
        """What this call cost, or None when nobody can say.

        Order matters. The provider's own figure wins because it is the only
        one that knows about markup, cached-prompt discounts and which upstream
        served the request. The price book is a fallback for providers that
        report nothing. If neither has an answer the result is None, NOT 0.0 --
        a run that reported "$0.0000 spent" after a real planner call had no
        error, no failing test and no warning, which is the whole reason this
        returns an optional.
        """
        if response.cost_usd is not None:
            return response.cost_usd
        if model_id in self.prices.prompt or model_id in self.prices.completion:
            return self.prices.cost(model_id, response.tokens_in, response.tokens_out)
        return None

    def _result(
        self,
        request: CallRequest,
        model_id: str,
        response: RawResponse,
        depth: int,
        attempts: int,
        cost: float | None,
        payload: Any,
        *,
        cache_hit: bool = False,
    ) -> CallResult:
        return CallResult(
            payload=payload,
            model_id=model_id,
            tier=request.tier,
            raw=response.text,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=cost,
            fallback_depth=depth,
            attempts=attempts,
            cache_hit=cache_hit,
            served_by=response.served_by,
        )


def _bump(ledger: CostLedger, field_name: str) -> CostLedger:
    from dataclasses import replace

    return replace(ledger, **{field_name: getattr(ledger, field_name) + 1})


def _with(result: CallResult, *, repaired_flag: bool, fallback_depth: int) -> CallResult:
    from dataclasses import replace

    return replace(result, repaired=repaired_flag, fallback_depth=fallback_depth)


def get_gateway(
    *,
    settings: Any = None,
    cache_enabled: bool = True,
    budget: Budget | None = None,
) -> GatewayClient:
    """Sole construction path for the gateway (playbook §3.1).

    Tests patch THIS, never langchain_openai -- that is the factory boundary,
    and mocking the library instead is AP-02.
    """
    from dynaflows.gateway.cache import get_response_cache
    from dynaflows.gateway.invoker import build_langchain_invoker
    from dynaflows.gateway.registry import get_model_registry
    from dynaflows.settings import get_settings

    settings = settings or get_settings()
    return GatewayClient(
        registry=get_model_registry(settings.models_config),
        invoker=build_langchain_invoker(settings),
        cache=get_response_cache(settings.calls_db, enabled=cache_enabled),
        budget=budget or Budget(),
        max_concurrent=settings.max_concurrent,
        timeout_seconds=settings.timeout_seconds,
    )
