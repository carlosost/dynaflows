"""The single place a provider SDK is constructed and called. ADR-010.

Everything above this file works in typed `DynaflowsError` terms; everything
below is provider-shaped. Keeping the translation in one function is what lets
the ladder be tested with a fake invoker and no HTTP at all.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from dynaflows.contracts.calls import CallRequest, RawResponse
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.settings import Settings

_STATUS_TO_CODE = {
    401: ErrorCode.AUTH_FAILED,
    403: ErrorCode.AUTH_FAILED,
    404: ErrorCode.MODEL_UNAVAILABLE,
    402: ErrorCode.MODEL_UNAVAILABLE,
    408: ErrorCode.TIMEOUT,
    429: ErrorCode.RATE_LIMIT,
    500: ErrorCode.MODEL_UNAVAILABLE,
    502: ErrorCode.MODEL_UNAVAILABLE,
    503: ErrorCode.MODEL_UNAVAILABLE,
    504: ErrorCode.TIMEOUT,
}


def classify(exc: BaseException) -> ErrorCode:
    """Provider exception -> our error contract.

    Read by class NAME and status attribute rather than by importing the
    provider's exception hierarchy: OpenRouter fronts several SDKs and the
    hierarchy is not ours to depend on. UNKNOWN is the honest default, and it
    falls through to the next model rather than aborting the run -- an
    unrecognised error might be model-specific.

    UNVERIFIED against a live provider: this mapping was written from
    documented status codes, not from observed exceptions (AP-19 habit 3).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int) and status in _STATUS_TO_CODE:
        return _STATUS_TO_CODE[status]
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return ErrorCode.RATE_LIMIT
    if "timeout" in name:
        return ErrorCode.TIMEOUT
    if "authentication" in name or "permission" in name:
        return ErrorCode.AUTH_FAILED
    if "notfound" in name or "unavailable" in name or "connection" in name:
        return ErrorCode.MODEL_UNAVAILABLE
    return ErrorCode.UNKNOWN


def retry_after_of(exc: BaseException) -> float | None:
    """Honour the provider's own advice when it gives any (ADR-010 layer 6)."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after") or headers.get("Retry-After"))
    except TypeError, ValueError:
        return None


def build_langchain_invoker(settings: Settings) -> Any:
    """The production invoker. The ONLY construction of a chat client."""
    from langchain_openai import ChatOpenAI

    async def invoke(model_id: str, request: CallRequest) -> RawResponse:
        messages: list[tuple[str, str]] = []
        if request.system:
            messages.append(("system", request.system))
        messages.append(("human", request.prompt))

        kwargs: dict[str, Any] = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        chat = ChatOpenAI(
            model=model_id,
            api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
            base_url=settings.openrouter_base_url,
            timeout=settings.timeout_seconds,
            # Retry lives in the ladder, which honours Retry-After and records
            # attempts in the trace. A second retry loop inside the SDK would
            # multiply both silently.
            max_retries=0,
            # ADR-006: route only to providers that actually implement
            # json_schema for this model, instead of one that ignores it.
            extra_body={"provider": {"require_parameters": True}},
            **kwargs,
        )
        # with_structured_output returns a Runnable, not a ChatOpenAI. Binding
        # it to a separate name keeps the type honest instead of widening the
        # first one to Any to make the assignment fit.
        runnable: Any = (
            chat.with_structured_output(request.schema, method="json_schema")
            if request.schema is not None
            else chat
        )

        try:
            result = await runnable.ainvoke(
                messages,
                config={
                    "run_name": request.label,
                    "tags": ["gateway", request.tier.value],
                    "metadata": {"model_id": model_id, **dict(request.metadata)},
                },
            )
        except Exception as exc:  # noqa: BLE001 -- translated, never propagated raw
            error = DynaflowsError.of(classify(exc), f"{model_id}: {exc}")
            error.retry_after = retry_after_of(exc)  # type: ignore[attr-defined]
            raise error from exc

        if request.schema is not None:
            return RawResponse(text=result.model_dump_json(), served_by=model_id)
        text = getattr(result, "content", None)
        usage = getattr(result, "usage_metadata", None) or {}
        return RawResponse(
            text=text if isinstance(text, str) else str(result),
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
            served_by=model_id,
        )

    return invoke
