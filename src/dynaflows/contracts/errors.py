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
    # 402. Distinct from MODEL_UNAVAILABLE on purpose: a model that is down
    # may come back, so retrying is reasonable.
    #
    # This used to say "a request you cannot afford costs the same on every
    # attempt, so retrying is three wasted calls", and that reasoning was
    # wrong. OpenRouter reserves `max_tokens` worth of balance BEFORE the
    # call, so the price of the attempt is a function of a number we choose:
    #
    #   "You requested up to 4096 tokens, but can only afford 2432"
    #
    # The identical request at 2432 succeeds. The provider even states the
    # affordable ceiling, and the gateway reads it and retries once at that
    # figure -- the same rule as taking the provider's own cost number
    # instead of computing one.
    #
    # The name is kept and is imperfect: this is a reservation ceiling, not
    # an empty account, and a balance too small to be worth using is the only
    # case where it means what it says.
    INSUFFICIENT_CREDIT = "INSUFFICIENT_CREDIT"
    # A model spent the whole output allowance and had nothing left to close
    # its answer with. Met live twice: a model burned 2,297 reasoning tokens
    # against a 2,048 cap, and later both workers of run `q5` stopped at
    # exactly `completion_tokens=4096` with unparseable JSON.
    #
    # "Retrying the same model with the same cap cannot help" is true and
    # stops one step short of the remedy. The provider states the binding
    # constraint -- the completion count IS the cap -- so the answer is to
    # retry that model with a DIFFERENT cap.
    #
    # Exactly the mirror of INSUFFICIENT_CREDIT above, which the gateway
    # learned to read an hour earlier:
    #
    #   402         "you can only afford 2432"     -> retry smaller
    #   truncation  "you produced exactly 4096"    -> retry larger
    #
    # Same class of error, same provider signal, opposite direction, and the
    # second one sat unread in the same `except` block while the first was
    # being fixed. **A remedy found for one error is worth testing against
    # its neighbours.**
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
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
