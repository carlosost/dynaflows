"""Live checks against OpenRouter. ADR-010: nothing outside this package
imports an HTTP client or a provider SDK.

Two capabilities, both used by `dynaflows doctor` and `dynaflows models`:

  catalogue()          -- which endpoints currently support json_schema
  probe_structured()   -- one traced, schema-enforced call, end to end

This is Phase 0's entire provider surface. The resiliency ladder is 1.2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.settings import Settings

if TYPE_CHECKING:
    from dynaflows.contracts.calls import CallRequest

_STRUCTURED_FILTER = {"supported_parameters": "structured_outputs"}


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    context_length: int
    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float

    @property
    def family(self) -> str:
        return self.id.split("/", 1)[0] if "/" in self.id else self.id


def _price(raw: dict[str, object], key: str) -> float:
    try:
        return float(str(raw.get(key) or 0.0)) * 1_000_000
    except TypeError, ValueError:
        return 0.0


def catalogue(settings: Settings, *, timeout: float | None = None) -> list[ModelInfo]:
    """Every endpoint that currently supports structured outputs.

    ADR-006 depends on this: a fallback chain must be capability-homogeneous,
    and the only authority on which models qualify is the live list.
    """
    if not settings.openrouter_api_key:
        raise DynaflowsError.of(ErrorCode.AUTH_FAILED, "OPENROUTER_API_KEY is not set")
    url = f"{settings.openrouter_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    try:
        response = httpx.get(
            url,
            params=_STRUCTURED_FILTER,
            headers=headers,
            timeout=timeout or settings.timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise DynaflowsError.of(ErrorCode.TIMEOUT, f"{url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise DynaflowsError.of(ErrorCode.MODEL_UNAVAILABLE, f"{url}: {exc}") from exc

    if response.status_code in (401, 403):
        raise DynaflowsError.of(
            ErrorCode.AUTH_FAILED, f"OpenRouter rejected the key ({response.status_code})"
        )
    if response.status_code != 200:
        raise DynaflowsError.of(
            ErrorCode.MODEL_UNAVAILABLE, f"{url} returned HTTP {response.status_code}"
        )

    models: list[ModelInfo] = []
    for entry in response.json().get("data", []):
        pricing = entry.get("pricing") or {}
        models.append(
            ModelInfo(
                id=str(entry.get("id", "")),
                context_length=int(entry.get("context_length") or 0),
                prompt_usd_per_mtok=_price(pricing, "prompt"),
                completion_usd_per_mtok=_price(pricing, "completion"),
            )
        )
    return [m for m in models if m.id]


class _Handshake(BaseModel):
    """The schema the hello-world call must satisfy.

    Deliberately not free text. Phase 0's job is to prove that
    `response_format: json_schema` is honoured end to end, so the probe fails
    loudly if a provider returns prose instead of a typed object.
    """

    ok: bool = Field(description="always true")
    model_said: str = Field(description="the single word: handshake")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model_id: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SdkAttempt:
    """What the real call path did, as a value rather than a traceback."""

    ok: bool
    detail: str
    tokens_in: int = 0
    tokens_out: int = 0


def call_through_gateway(
    settings: Settings,
    model_id: str,
    request: CallRequest,
) -> SdkAttempt:
    """Run ONE CallRequest through the production path, on a pinned model.

    This is the half of a reproduction that must not be re-implemented: if the
    diagnostic built its own ChatOpenAI, it would prove something about a call
    the product never makes. Retries and cache off, so one request is one
    provider call and the failure is not smeared across a chain.
    """
    import asyncio  # noqa: PLC0415

    from dynaflows.gateway.cache import NullCache  # noqa: PLC0415
    from dynaflows.gateway.client import GatewayClient  # noqa: PLC0415
    from dynaflows.gateway.invoker import build_langchain_invoker  # noqa: PLC0415
    from dynaflows.gateway.registry import TierChain, load_registry  # noqa: PLC0415

    registry = load_registry(settings.models_config)
    pinned = replace(
        registry,
        tiers={**registry.tiers, request.tier: TierChain(request.tier, "repro", (model_id,))},
    )
    gateway = GatewayClient(
        registry=pinned,
        invoker=build_langchain_invoker(settings),
        cache=NullCache(),
        max_attempts=1,
        timeout_seconds=settings.timeout_seconds,
    )
    try:
        result = asyncio.run(gateway.call(request))
    except DynaflowsError as exc:
        return SdkAttempt(False, str(exc))
    except Exception as exc:  # noqa: BLE001 -- a diagnostic reports, never propagates
        return SdkAttempt(False, f"{type(exc).__name__}: {exc}")
    return SdkAttempt(
        True,
        f"{type(result.payload).__name__} returned",
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
    )


def probe_structured(settings: Settings, model_id: str) -> ProbeResult:
    """One traced, schema-enforced call. Never raises; reports instead.

    Runs THROUGH the gateway rather than beside it. An earlier version built
    its own ChatOpenAI here, which would have made two implementations of "call
    the provider" -- the AP-11 shape, where a fix applied to one has no effect
    on the other. Retries and cache are disabled so the doctor measures the
    provider, not the ladder.
    """
    import asyncio

    from dynaflows.contracts.calls import CallRequest
    from dynaflows.contracts.tiers import Tier
    from dynaflows.gateway.cache import NullCache
    from dynaflows.gateway.client import GatewayClient
    from dynaflows.gateway.invoker import build_langchain_invoker
    from dynaflows.gateway.registry import TierChain, load_registry

    registry = load_registry(settings.models_config)
    # Probe THIS model, whatever the tier's chain says.
    pinned = replace(
        registry, tiers={**registry.tiers, Tier.SMALL: TierChain(Tier.SMALL, "probe", (model_id,))}
    )
    gateway = GatewayClient(
        registry=pinned,
        invoker=build_langchain_invoker(settings),
        cache=NullCache(),
        max_attempts=1,
        timeout_seconds=settings.timeout_seconds,
    )
    request = CallRequest(
        tier=Tier.SMALL,
        prompt="Reply with ok=true and model_said set to the single word: handshake",
        schema=_Handshake,
        label="doctor.probe_structured",
        metadata=(("tier_probe", "true"),),
    )
    try:
        result = asyncio.run(gateway.call(request))
    except DynaflowsError as exc:
        return ProbeResult(model_id, False, str(exc))
    except Exception as exc:  # noqa: BLE001 -- a doctor check reports, never propagates
        return ProbeResult(model_id, False, f"{type(exc).__name__}: {exc}")

    payload = result.payload
    if not isinstance(payload, _Handshake):
        return ProbeResult(model_id, False, f"schema not honoured; got {type(payload).__name__}")
    return ProbeResult(model_id, True, f"structured output honoured (said: {payload.model_said!r})")
