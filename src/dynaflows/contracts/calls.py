"""What a provider call looks like crossing the gateway boundary. ADR-010/013."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel

from dynaflows.contracts.tiers import Tier


@dataclass(frozen=True, slots=True)
class CallRequest:
    """One logical request. The tier is named, never the model.

    Which model serves this is the registry's business (ADR-006) and the
    fallback ladder's (ADR-010); a caller that named a model would pin a
    decision that belongs in configuration.
    """

    tier: Tier
    prompt: str
    system: str | None = None
    schema: type[BaseModel] | None = None
    # ADR-013: how sampling diversity is requested. Two calls that differ only
    # here are two different cache entries, which is what lets Phase 5 ask for
    # independent attempts at an identical prompt without disabling the cache.
    variant: int = 0
    temperature: float | None = None
    max_tokens: int | None = None
    label: str = "call"
    metadata: tuple[tuple[str, str], ...] = ()

    def cache_key(self, model_id: str) -> str:
        """Content address for this exact call against this exact model.

        Deliberately EXCLUDES run_id, task_id, thread_id and any timestamp.
        Include one of them and the cache never hits (ADR-013.2). Includes the
        model id, so a response from one model is never served for another.
        """
        material = {
            "model": model_id,
            "system": self.system,
            "prompt": self.prompt,
            "schema": self.schema.model_json_schema() if self.schema else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "variant": self.variant,
        }
        canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RawResponse:
    """What an invoker returns. Deliberately untyped text plus usage.

    ADR-013.4 stores this verbatim: a parsed form has already lost something a
    later caller might need, which is the AP-01 family.
    """

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    served_by: str | None = None


@dataclass(frozen=True, slots=True)
class CallResult:
    payload: Any
    model_id: str
    tier: Tier
    raw: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    fallback_depth: int
    attempts: int
    cache_hit: bool
    repaired: bool = False
    served_by: str | None = None

    def trace_metadata(self) -> dict[str, str | int | float | bool]:
        """ADR-011: what LangSmith needs to answer 'why did this cost that'."""
        return {
            "model_id": self.model_id,
            "tier": self.tier.value,
            "fallback_depth": self.fallback_depth,
            "attempts": self.attempts,
            "cache_hit": self.cache_hit,
            "repaired": self.repaired,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class CostLedger:
    """Two money counters, never merged (ADR-013.7, AP-20).

    `usd_spent` is money that left the account. `usd_avoided` is money a cache
    hit did not spend. A single "cost" number answers neither "what did this run
    cost" nor "what would it have cost cold" -- and gate G2's estimate needs
    the second one.
    """

    usd_spent: float = 0.0
    usd_avoided: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls_made: int = 0
    calls_cached: int = 0
    # AP-20 again: a call that failed and a call that was never attempted are
    # different facts and do not share a counter.
    schema_failures: int = 0
    calls_skipped: int = 0
    fallbacks: int = 0

    def record(self, result: CallResult) -> CostLedger:
        if result.cache_hit:
            return replace(
                self,
                usd_avoided=self.usd_avoided + result.cost_usd,
                calls_cached=self.calls_cached + 1,
            )
        return replace(
            self,
            usd_spent=self.usd_spent + result.cost_usd,
            tokens_in=self.tokens_in + result.tokens_in,
            tokens_out=self.tokens_out + result.tokens_out,
            calls_made=self.calls_made + 1,
            fallbacks=self.fallbacks + (1 if result.fallback_depth else 0),
        )


@dataclass(frozen=True, slots=True)
class Budget:
    """Per-run ceilings. ADR-010 layer 1.

    KNOWN LIMITATION, recorded rather than hidden: the check is post-hoc. Cost
    is only known after a response arrives, so the ceiling stops the call
    AFTER the one that crossed it, not the one that crosses it. Overshoot is
    bounded by a single call. Pre-flight estimation needs the price catalogue
    on every call, which is a network round trip the ladder should not take.
    """

    usd_limit: float | None = None
    token_limit: int | None = None

    def exceeded_by(self, ledger: CostLedger) -> str | None:
        if self.usd_limit is not None and ledger.usd_spent >= self.usd_limit:
            return f"spent ${ledger.usd_spent:.4f} of ${self.usd_limit:.4f}"
        if self.token_limit is not None:
            used = ledger.tokens_in + ledger.tokens_out
            if used >= self.token_limit:
                return f"used {used} of {self.token_limit} tokens"
        return None


@dataclass(frozen=True, slots=True)
class PriceBook:
    """USD per million tokens, by model id. Empty is a legitimate state.

    Without prices the token ceiling is the effective guard and `usd_spent`
    stays 0.0 -- which is honest. A fabricated price would make the ledger
    confidently wrong, which is worse than visibly incomplete.
    """

    prompt: dict[str, float] = field(default_factory=dict)
    completion: dict[str, float] = field(default_factory=dict)

    def cost(self, model_id: str, tokens_in: int, tokens_out: int) -> float:
        return (
            self.prompt.get(model_id, 0.0) * tokens_in
            + self.completion.get(model_id, 0.0) * tokens_out
        ) / 1_000_000
