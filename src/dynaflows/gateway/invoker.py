"""The single place a provider SDK is constructed and called. ADR-010.

Everything above this file works in typed `DynaflowsError` terms; everything
below is provider-shaped. Keeping the translation in one function is what lets
the ladder be tested with a fake invoker and no HTTP at all.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr

from dynaflows.contracts.calls import CallRequest, RawResponse
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.settings import Settings

_STATUS_TO_CODE = {
    401: ErrorCode.AUTH_FAILED,
    403: ErrorCode.AUTH_FAILED,
    404: ErrorCode.MODEL_UNAVAILABLE,
    402: ErrorCode.INSUFFICIENT_CREDIT,
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
    # OpenRouter can return a body with `choices: null` when an upstream
    # provider fails mid-request; the SDK then iterates None and raises a bare
    # TypeError. That is a provider failure -- worth another model -- not an
    # unknown one, and "'NoneType' object is not iterable" is a terrible thing
    # to hand a user.
    text = str(exc)
    if isinstance(exc, TypeError) and "NoneType" in text and "iterable" in text:
        return ErrorCode.MODEL_UNAVAILABLE

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


def remedy_of(exc: BaseException) -> str | None:
    """The provider's own suggested fix, when it sends one.

    A 402 from OpenRouter carries `metadata.remedy_hint` naming exactly what to
    change. Discarding it and printing a stack trace instead makes the user
    rediscover what the response already told them.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata.get("remedy_hint"):
        return str(metadata["remedy_hint"])
    return None


def retry_after_of(exc: BaseException) -> float | None:
    """Honour the provider's own advice when it gives any (ADR-010 layer 6)."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after") or headers.get("Retry-After"))
    except TypeError, ValueError:
        return None


def _usage(message: Any) -> dict[str, Any]:
    """Token counts and the provider's own cost from an AIMessage.

    Zeros are honest when the provider sent none. They are NOT honest when we
    threw the message away -- which is what happened before include_raw.

    The cost is read rather than computed. OpenRouter states what a call cost
    in its `usage` block, and LangChain passes that block through to
    `response_metadata["token_usage"]` whole. That number already accounts for
    provider markup, the cached-prompt discount and whichever upstream actually
    served the request -- none of which a local price table can know. A missing
    cost stays None; see RawResponse.cost_usd.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    metadata = getattr(message, "response_metadata", None) or {}
    reported = metadata.get("token_usage") or {}
    raw_cost = reported.get("cost") if isinstance(reported, dict) else None
    cost: float | None
    try:
        cost = None if raw_cost is None else float(raw_cost)
    except TypeError, ValueError:
        cost = None
    return {
        "tokens_in": int(usage.get("input_tokens", 0) or 0),
        "tokens_out": int(usage.get("output_tokens", 0) or 0),
        "cost_usd": cost,
    }


def raw_completion(
    settings: Settings,
    model_id: str,
    *,
    prompt: str = "Reply with ok=true and model_said set to the single word: handshake",
    system: str | None = None,
    max_tokens: int = 4096,
    schema: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    """One chat/completions POST, bypassing every SDK. Returns (status, body).

    A diagnostic, and it exists because of a real failure: a structured call
    died with "'NoneType' object is not iterable" from somewhere inside the
    provider SDK, which named no cause a user could act on. When the stack
    cannot explain a response, the only honest next step is to look at the
    response (AP-19 habit 3).

    `response_format` overrides `schema` and is passed through verbatim. That
    is what lets a reproduction send the block LangChain would have sent,
    rather than a hand-written approximation of it -- the whole value of the
    comparison is that only ONE thing differs from the failing call.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
        body["provider"] = {"require_parameters": True}
    elif schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "probe", "strict": True, "schema": schema},
        }
        body["provider"] = {"require_parameters": True}

    response = httpx.post(
        f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json=body,
        timeout=settings.timeout_seconds,
    )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def langchain_response_format(schema: Any) -> dict[str, Any]:
    """The response_format block that actually reaches OpenRouter.

    Not a reconstruction. There are two converters in this path and only the
    second one matters: LangChain's `_convert_to_openai_response_format`
    returns a pydantic CLASS unchanged when given one, and hands it to the
    openai SDK, which converts it with `type_to_response_format_param` and puts
    THAT on the wire. A diagnostic built on LangChain's converter would send a
    pydantic class as JSON and prove nothing.

    Both are private imports. That is the cost of reproducing a call exactly;
    re-deriving the conversion here would be the AP-11 shape -- a second
    implementation that drifts from the real one silently, because a
    reproduction that disagrees with production still looks like a
    reproduction.
    """
    from langchain_core.utils.pydantic import is_basemodel_subclass  # noqa: PLC0415
    from openai.lib._parsing._completions import (  # noqa: PLC0415
        type_to_response_format_param,
    )

    converted: Any
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        converted = type_to_response_format_param(schema)
    else:
        from langchain_openai.chat_models.base import (  # noqa: PLC0415
            _convert_to_openai_response_format,
        )

        converted = _convert_to_openai_response_format(schema, strict=None)

    if not isinstance(converted, dict):
        raise DynaflowsError.of(
            ErrorCode.SCHEMA_INVALID,
            f"conversion returned {type(converted).__name__}, not a response_format block",
        )
    return dict(converted)


def build_chat_client(settings: Settings, model_id: str, request: CallRequest) -> Any:
    """Construct the provider chat client. The ONLY place it happens (ADR-010).

    Module level rather than inline in the invoker's closure for one reason:
    a test has to be able to ask what this will put on the wire. The live
    failure that motivated it was a parameter LangChain renamed between this
    constructor and the HTTP request, which no fake-based test can observe.
    """
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    kwargs: dict[str, Any] = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature

    # ADR-006: route only to providers that actually implement json_schema for
    # this model, instead of one that ignores it.
    extra_body: dict[str, Any] = {"provider": {"require_parameters": True}}
    if request.max_tokens is not None:
        # AP-05, observed live. Passing max_tokens to the ChatOpenAI
        # CONSTRUCTOR does not send max_tokens: LangChain renames it to
        # `max_completion_tokens` on the way out (base.py `_default_params`
        # and `_get_request_payload`, unconditionally). OpenRouter's
        # require_parameters filter does not recognise that name for these
        # endpoints, drops every candidate, and reports it as a 404 "Filter by
        # Parameters" for one model and a 200 with `choices: null` for another
        # -- which surfaced as "'NoneType' object is not iterable" from inside
        # the openai SDK, naming neither the parameter nor the rename.
        #
        # extra_body is merged into the request body verbatim, so the name
        # chosen here is the name that reaches the provider.
        extra_body["max_tokens"] = request.max_tokens

    return ChatOpenAI(
        model=model_id,
        api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
        base_url=settings.openrouter_base_url,
        timeout=settings.timeout_seconds,
        # Retry lives in the ladder, which honours Retry-After and records
        # attempts in the trace. A second retry loop inside the SDK would
        # multiply both silently.
        max_retries=0,
        extra_body=extra_body,
        **kwargs,
    )


def build_langchain_invoker(settings: Settings) -> Any:
    """The production invoker. Constructs nothing itself; see build_chat_client."""

    async def invoke(model_id: str, request: CallRequest) -> RawResponse:
        messages: list[tuple[str, str]] = []
        if request.system:
            messages.append(("system", request.system))
        messages.append(("human", request.prompt))

        chat = build_chat_client(settings, model_id, request)

        # with_structured_output returns a Runnable, not a ChatOpenAI. Binding
        # it to a separate name keeps the type honest instead of widening the
        # first one to Any to make the assignment fit.
        #
        # include_raw=True is not optional here. Without it the runnable
        # returns ONLY the parsed model, and the AIMessage carrying
        # usage_metadata is discarded -- so every structured call recorded
        # 0 tokens and $0.00, and the ledger, the budget ceiling and the G2
        # estimate were all quietly counting nothing.
        runnable: Any = (
            chat.with_structured_output(request.schema, method="json_schema", include_raw=True)
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
            # The exception TYPE is half the diagnosis. "'NoneType' object is
            # not iterable" names no subject; "TypeError: 'NoneType' object is
            # not iterable" at least says the failure was structural rather
            # than a provider status code.
            error = DynaflowsError.of(classify(exc), f"{model_id}: {type(exc).__name__}: {exc}")
            error.retry_after = retry_after_of(exc)  # type: ignore[attr-defined]
            error.remedy = remedy_of(exc)  # type: ignore[attr-defined]
            raise error from exc

        if request.schema is not None:
            # include_raw gives {"raw": AIMessage, "parsed": Model, "parsing_error": ...}
            parsed = result.get("parsed")
            if parsed is None:
                raise DynaflowsError.of(
                    ErrorCode.SCHEMA_INVALID,
                    f"{model_id}: {result.get('parsing_error') or 'no parsed output'}",
                )
            return RawResponse(
                text=parsed.model_dump_json(),
                served_by=model_id,
                **_usage(result.get("raw")),
            )
        text = getattr(result, "content", None)
        return RawResponse(
            text=text if isinstance(text, str) else str(result),
            served_by=model_id,
            **_usage(result),
        )

    return invoke
