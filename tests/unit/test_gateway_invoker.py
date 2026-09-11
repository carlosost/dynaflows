"""Provider exception -> our error contract. ADR-010.

The classifier decides whether the ladder retries, falls through, or aborts, so
getting it wrong is not cosmetic: a 401 classified as retryable burns the whole
chain, and a 429 classified as fatal ends a run that would have succeeded.

UNVERIFIED against a live provider -- these fakes carry documented status codes
and class names, not observed exceptions (AP-19 habit 3).
"""

from __future__ import annotations

import pytest

from dynaflows.contracts.errors import ErrorCode
from dynaflows.gateway.invoker import classify, retry_after_of

pytestmark = pytest.mark.deterministic


class Fake(Exception):
    def __init__(self, name: str, status: int | None = None, headers: dict | None = None) -> None:
        super().__init__(name)
        type(self).__name__ = name
        if status is not None:
            self.status_code = status
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ErrorCode.AUTH_FAILED),
        (403, ErrorCode.AUTH_FAILED),
        (402, ErrorCode.INSUFFICIENT_CREDIT),
        (404, ErrorCode.MODEL_UNAVAILABLE),
        (408, ErrorCode.TIMEOUT),
        (429, ErrorCode.RATE_LIMIT),
        (503, ErrorCode.MODEL_UNAVAILABLE),
        (504, ErrorCode.TIMEOUT),
    ],
)
def test_status_codes_map_to_the_error_contract(status: int, expected: ErrorCode) -> None:
    assert classify(Fake("SomeError", status)) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RateLimitError", ErrorCode.RATE_LIMIT),
        ("APITimeoutError", ErrorCode.TIMEOUT),
        ("AuthenticationError", ErrorCode.AUTH_FAILED),
        ("PermissionDeniedError", ErrorCode.AUTH_FAILED),
        ("NotFoundError", ErrorCode.MODEL_UNAVAILABLE),
        ("APIConnectionError", ErrorCode.MODEL_UNAVAILABLE),
    ],
)
def test_class_names_are_a_fallback_when_there_is_no_status(name: str, expected: ErrorCode) -> None:
    """Read by name rather than by importing the provider's exception tree:
    OpenRouter fronts several SDKs and that hierarchy is not ours to depend on."""
    assert classify(Fake(name)) is expected


def test_an_unrecognised_error_is_unknown_not_guessed() -> None:
    """UNKNOWN falls through to the next model rather than aborting the run --
    an unrecognised failure might well be model-specific."""
    assert classify(Fake("SomethingNovel")) is ErrorCode.UNKNOWN


def test_a_status_code_wins_over_a_misleading_class_name() -> None:
    assert classify(Fake("TimeoutIshError", 429)) is ErrorCode.RATE_LIMIT


def test_retry_after_is_read_from_the_response_headers() -> None:
    assert retry_after_of(Fake("RateLimitError", 429, {"retry-after": "12"})) == 12.0


def test_a_missing_or_unparseable_retry_after_is_none_not_zero() -> None:
    """Zero would mean 'retry immediately', which is the opposite of what a
    missing header means."""
    assert retry_after_of(Fake("RateLimitError", 429)) is None
    assert retry_after_of(Fake("RateLimitError", 429, {"retry-after": "soon"})) is None


# --------------------------------------------------------------------------
# Usage metadata. Every structured call recorded 0 tokens and $0.00 because
# with_structured_output returns only the parsed model and discards the
# AIMessage that carries usage_metadata. The ledger, the budget ceiling and
# the G2 estimate were all counting nothing.
# --------------------------------------------------------------------------


class Message:
    def __init__(self, usage: dict | None = None, content: str = "hi") -> None:
        self.usage_metadata = usage
        self.content = content


def test_usage_is_read_from_the_message() -> None:
    from dynaflows.gateway.invoker import _usage

    assert _usage(Message({"input_tokens": 120, "output_tokens": 45})) == {
        "tokens_in": 120,
        "tokens_out": 45,
        "cost_usd": None,
    }


def test_a_provider_that_sends_no_usage_yields_zeros_not_a_crash() -> None:
    from dynaflows.gateway.invoker import _usage

    zeros = {"tokens_in": 0, "tokens_out": 0, "cost_usd": None}
    assert _usage(Message(None)) == zeros
    assert _usage(None) == zeros


def test_null_token_fields_are_treated_as_zero() -> None:
    """Some providers send the keys with null values rather than omitting them."""
    from dynaflows.gateway.invoker import _usage

    assert _usage(Message({"input_tokens": None, "output_tokens": None})) == {
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": None,
    }


# --------------------------------------------------------------------------
# 402, met live. The classifier's FIRST real failure, and it got it wrong:
# 402 was MODEL_UNAVAILABLE, which is retryable, so a request that could never
# succeed was sent three times.
# --------------------------------------------------------------------------


def test_insufficient_credit_is_not_retryable() -> None:
    """A model that is down may come back, so retrying is reasonable. A
    request you cannot afford costs the same on every attempt."""
    from dynaflows.gateway.client import _RETRYABLE

    assert ErrorCode.INSUFFICIENT_CREDIT not in _RETRYABLE


def test_insufficient_credit_still_falls_through_to_a_cheaper_model() -> None:
    """Not fatal either: the next entry in the chain may well be affordable,
    which is the whole reason a chain has more than one model."""
    from dynaflows.gateway.client import _FATAL

    assert ErrorCode.INSUFFICIENT_CREDIT not in _FATAL


def test_the_providers_own_remedy_is_kept() -> None:
    """The 402 body names exactly what to change. Discarding it and printing a
    stack trace makes the user rediscover what the response already said."""
    from dynaflows.gateway.invoker import remedy_of

    exc = Fake("APIStatusError", 402)
    exc.body = {
        "error": {
            "message": "requires more credits",
            "metadata": {"remedy_hint": "Add credits, or lower max_tokens"},
        }
    }
    assert remedy_of(exc) == "Add credits, or lower max_tokens"


def test_a_response_without_a_remedy_is_not_invented() -> None:
    from dynaflows.gateway.invoker import remedy_of

    assert remedy_of(Fake("APIStatusError", 500)) is None
    plain = Fake("APIStatusError", 402)
    plain.body = {"error": {"message": "no metadata here"}}
    assert remedy_of(plain) is None


def test_a_malformed_provider_response_is_a_provider_failure_not_unknown() -> None:
    """OpenRouter can return `choices: null` when an upstream provider fails
    mid-request; the SDK iterates None and raises a bare TypeError. Live, that
    surfaced as `UNKNOWN ...: 'NoneType' object is not iterable` -- a message
    naming no cause, on a code that does not fall through to another model.
    """
    assert classify(TypeError("'NoneType' object is not iterable")) is (ErrorCode.MODEL_UNAVAILABLE)


def test_an_unrelated_type_error_is_still_unknown() -> None:
    """The rule is narrow on purpose: widening it would swallow real bugs in
    our own code as 'the provider is down'."""
    assert classify(TypeError("unsupported operand type(s)")) is ErrorCode.UNKNOWN
