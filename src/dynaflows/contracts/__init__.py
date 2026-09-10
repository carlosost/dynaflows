"""Typed contracts. Written before the code that uses them (playbook 1.4).

Phase 0 ships only the contracts that have a caller today: `Tier` (used by the
model registry and doctor) and `ErrorEnvelope` (used by the gateway).

`WorkflowState`, `Plan`, `PlanTask` and `WorkerResult` are specified in
docs/PROJECT_MEMORY.md 2.1-2.3 and are NOT implemented here. Their callers do
not exist until Phase 1 steps 1.3, 1.5 and 1.6. Building them now would create
exactly the shape AP-11 warns about: an abstraction whose only exercise is its
own test.
"""

from dynaflows.contracts.errors import DynaflowsError, ErrorCode, ErrorEnvelope
from dynaflows.contracts.tiers import Tier

__all__ = ["DynaflowsError", "ErrorCode", "ErrorEnvelope", "Tier"]
