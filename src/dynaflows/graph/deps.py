"""How a node gets the things it cannot construct itself.

LangGraph hands every node its `RunnableConfig`, and `configurable` is the
documented place for per-run dependencies. Using it instead of a module-level
singleton means a test injects a fake gateway by passing a config -- no
patching, no import-order games.
"""

from typing import Any

from dynaflows.contracts.errors import DynaflowsError, ErrorCode

GATEWAY_KEY = "gateway"
AUTO_APPROVE_KEY = "auto_approve"


def configurable(config: Any) -> dict[str, Any]:
    return dict((config or {}).get("configurable") or {})


def gateway_from(config: Any) -> Any:
    gateway = configurable(config).get(GATEWAY_KEY)
    if gateway is None:
        raise DynaflowsError.of(
            ErrorCode.CONFIG_INVALID,
            "no gateway in config['configurable']; the graph was invoked without one",
        )
    return gateway


def auto_approved(config: Any, gate: str) -> bool:
    """Gates the caller chose to skip. ADR-005: G1 may be skipped with
    --yes-prompt; G2 is threshold-driven and is never blanket-skipped."""
    return gate in set(configurable(config).get(AUTO_APPROVE_KEY) or ())
