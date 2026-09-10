"""Circuit breaker. ADR-010 layer 6.

Per model id: K consecutive failures marks it unhealthy for T seconds, and
chains skip it. Without this, a model that is down absorbs the full retry
budget of every call in a 12-way fan-out before the run makes any progress.

The clock is injected. A breaker tested with real sleeps is a breaker whose
tests are slow enough that someone eventually skips them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown_seconds: float = 60.0
    now: Callable[[], float] = field(default=lambda: 0.0)
    _failures: dict[str, int] = field(default_factory=dict, init=False)
    _open_until: dict[str, float] = field(default_factory=dict, init=False)

    def is_open(self, model_id: str) -> bool:
        until = self._open_until.get(model_id)
        if until is None:
            return False
        if self.now() >= until:
            # Cooldown elapsed: forget the streak so one more failure does not
            # immediately re-open on a stale count.
            self._open_until.pop(model_id, None)
            self._failures.pop(model_id, None)
            return False
        return True

    def record_failure(self, model_id: str) -> None:
        count = self._failures.get(model_id, 0) + 1
        self._failures[model_id] = count
        if count >= self.threshold:
            self._open_until[model_id] = self.now() + self.cooldown_seconds

    def record_success(self, model_id: str) -> None:
        self._failures.pop(model_id, None)
        self._open_until.pop(model_id, None)
