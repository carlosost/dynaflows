"""Live checks against OpenRouter. ADR-010: nothing outside this package
imports an HTTP client or a provider SDK.

Two capabilities, both used by `dynaflows doctor` and `dynaflows models`:

  catalogue()          -- which endpoints currently support json_schema
  probe_structured()   -- one traced, schema-enforced call, end to end

This is Phase 0's entire provider surface. The resiliency ladder is 1.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field, SecretStr

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.settings import Settings

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


def probe_structured(settings: Settings, model_id: str) -> ProbeResult:
    """One traced, schema-enforced call. Never raises; reports instead.

    `require_parameters` routes only to providers that actually implement
    json_schema for this model -- ADR-006. Without it OpenRouter may fall back
    to a provider that ignores the schema, which is the failure this whole
    probe exists to detect.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_id,
            api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
            base_url=settings.openrouter_base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
            model_kwargs={"extra_body": {"provider": {"require_parameters": True}}},
        ).with_structured_output(_Handshake)

        result = llm.invoke(
            "Reply with ok=true and model_said set to the single word: handshake",
            config={
                "run_name": "doctor.probe_structured",
                "tags": ["doctor", "phase-0"],
                "metadata": {"model_id": model_id, "tier_probe": True},
            },
        )
    except Exception as exc:  # noqa: BLE001 -- a doctor check reports, never propagates
        return ProbeResult(model_id, False, f"{type(exc).__name__}: {exc}")

    if not isinstance(result, _Handshake):
        return ProbeResult(model_id, False, f"schema not honoured; got {type(result).__name__}")
    return ProbeResult(model_id, True, f"structured output honoured (said: {result.model_said!r})")
