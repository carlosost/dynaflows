"""Every checkpointable type is registered. ADR-008.

LangGraph will BLOCK deserialisation of unregistered types in a future
version. That is AP-05's shape -- works now, stops working on an upgrade --
and the casualty is resume: every checkpoint ever written becomes unreadable.

The strict round-trip test in test_graph_skeleton.py proves today's state
survives. This test is the other half: it proves the LIST cannot silently fall
behind the contracts.
"""

from __future__ import annotations

import enum
import inspect

import pytest
from pydantic import BaseModel

from dynaflows.contracts import calls, state
from dynaflows.graph.checkpoint import ALLOWED_STATE_TYPES

pytestmark = pytest.mark.deterministic


def serializable_types_in(module: object) -> set[type]:
    """Pydantic models and enums DEFINED in this module (not imported into it)."""
    return {
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__  # type: ignore[attr-defined]
        and (issubclass(obj, BaseModel) or issubclass(obj, enum.Enum))
    }


@pytest.mark.parametrize("module", [state, calls], ids=["contracts.state", "contracts.calls"])
def test_every_state_type_is_registered_for_checkpointing(module: object) -> None:
    missing = serializable_types_in(module) - set(ALLOWED_STATE_TYPES)
    assert not missing, (
        f"{sorted(t.__name__ for t in missing)} can reach a checkpoint but are not in "
        "ALLOWED_STATE_TYPES; a future LangGraph will refuse to deserialise them and "
        "every existing checkpoint becomes unreadable"
    )


def test_the_allowlist_holds_classes_not_strings() -> None:
    """A string list can name a class that no longer exists and nothing
    notices. An import fails on the spot."""
    assert all(inspect.isclass(t) for t in ALLOWED_STATE_TYPES)
