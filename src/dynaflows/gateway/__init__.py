"""Gateway -- the sole chokepoint for provider traffic (ADR-010).

No module outside this package imports a provider SDK or an HTTP client.
Enforced by scripts/lint_architecture.py, by the ruff banned-api rule in
pyproject.toml, and by tests/architecture/test_gateway_boundary.py.

Phase 0 scope is deliberately narrow: the model registry, and one traced
schema-enforced call used by `dynaflows doctor` to prove the
OpenRouter -> structured-output -> LangSmith path end to end.

The full resiliency ladder from ADR-010 -- semaphore, retry, tier fallback,
circuit breaker, JSON repair, budget ceiling -- is Phase 1 step 1.2. It is not
written here, because nothing calls it yet (AP-11).
"""

from dynaflows.gateway.registry import (
    ModelRegistry,
    TierChain,
    get_model_registry,
)

__all__ = ["ModelRegistry", "TierChain", "get_model_registry"]
