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
        (402, ErrorCode.MODEL_UNAVAILABLE),
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
