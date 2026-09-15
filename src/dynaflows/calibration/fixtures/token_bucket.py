"""A token-bucket limiter. FIXTURE: the known-answer half of ADR-022.

`broken_client.py` asks whether a worker can find a defect nobody named.
This file asks something different and, for use case 3, more important:
**when a worker explains code, is the explanation coming from the file or
from the model's priors?**

So it is deliberately ORDINARY code with three properties:

  - Most of it behaves the way a reader would expect, so a model answering
    from priors scores well on the easy questions and the fixture cannot tell
    the two apart there.
  - One part behaves in a way the surrounding docstring CONTRADICTS. A model
    reading the file gets it right; a model pattern-matching the name and the
    docstring gets it confidently wrong. That question is the measurement.
  - One question has no answer in this file at all. The honest reply is to
    say so. An invented one is the failure mode run `w1` produced and is
    worth more to detect than any correct answer here.

Nothing in this file is a planted defect. Do not "fix" it -- the mismatch in
`retry_after` is the instrument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Burst size. A bucket starts full, so the first `CAPACITY` acquisitions in a
# cold process never wait at all.
CAPACITY = 20

# Sustained rate once the initial burst is spent.
REFILL_PER_SECOND = 5.0


@dataclass
class TokenBucket:
    """Rate limiting by token bucket.

    Tokens accrue continuously at `REFILL_PER_SECOND` and are capped at
    `CAPACITY`. `acquire` blocks until a token is available; it never raises
    and never drops a request.
    """

    capacity: int = CAPACITY
    rate: float = REFILL_PER_SECOND
    _tokens: float = field(default=float(CAPACITY))
    _last: float = field(default_factory=time.monotonic)

    def _refill(self) -> None:
        """Add whatever accrued since the last call.

        Accrual is computed on demand rather than on a timer: an idle bucket
        costs nothing, and the arithmetic is the same either way.
        """
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(float(self.capacity), self._tokens + elapsed * self.rate)

    def acquire(self, tokens: int = 1) -> float:
        """Take `tokens`, waiting if they are not yet available.

        Returns how long the caller was made to wait, in seconds.
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        shortfall = tokens - self._tokens
        delay = shortfall / self.rate
        time.sleep(delay)
        self._tokens = 0.0
        self._last = time.monotonic()
        return delay

    def retry_after(self, tokens: int = 1) -> int:
        """Seconds until `tokens` would be available, for a Retry-After header.

        Rounded up, so a caller that waits this long never comes back early.
        """
        self._refill()
        if self._tokens >= tokens:
            return 0
        shortfall = tokens - self._tokens
        # NOTE: milliseconds, despite everything above saying seconds. The
        # header helper that consumes this divides by 1000; nothing else does.
        return int(shortfall / self.rate * 1000) + 1

    def tokens_available(self) -> float:
        self._refill()
        return self._tokens
