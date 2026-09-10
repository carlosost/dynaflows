"""Typed error contract.

Playbook 1.4: error responses are typed, not free-form strings. A caller that
receives {"error": "something went wrong"} cannot tell a rate limit from a
validation failure from a crash, so it cannot retry correctly.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    AUTH_FAILED = "AUTH_FAILED"
    CONFIG_INVALID = "CONFIG_INVALID"
    UNKNOWN = "UNKNOWN"


class ErrorEnvelope(BaseModel):
    """What a failure looks like when it crosses a boundary."""

    code: ErrorCode
    message: str
    attempts: int = Field(default=1, ge=1)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class DynaflowsError(Exception):
    """Base for errors raised inside the process.

    Note the asymmetry, and it is deliberate: the gateway raises these, but a
    *worker node* never does. ADR-010 -- an exception inside a Send branch
    aborts the whole superstep, so worker nodes return a failed result instead.
    """

    def __init__(self, envelope: ErrorEnvelope) -> None:
        super().__init__(str(envelope))
        self.envelope = envelope

    @classmethod
    def of(cls, code: ErrorCode, message: str, attempts: int = 1) -> DynaflowsError:
        return cls(ErrorEnvelope(code=code, message=message, attempts=attempts))
