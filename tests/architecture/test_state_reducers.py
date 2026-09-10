"""Every concurrent key carries a reducer. PMA §2.1.

LangGraph raises InvalidUpdateError when two parallel branches write one key
with no reducer -- at runtime, only under fan-out, and therefore only in the
situation hardest to reproduce on a laptop. These tests move that failure to
import time.
"""

from __future__ import annotations

import typing

import pytest

from dynaflows.contracts.state import FAN_OUT_KEYS, WorkflowState

pytestmark = pytest.mark.deterministic


def reduced_keys() -> set[str]:
    hints = typing.get_type_hints(WorkflowState, include_extras=True)
    return {
        name
        for name, hint in hints.items()
        if typing.get_origin(hint) is typing.Annotated and typing.get_args(hint)[1:]
    }


def test_every_declared_fan_out_key_has_a_reducer() -> None:
    missing = FAN_OUT_KEYS - reduced_keys()
    assert not missing, (
        f"{sorted(missing)} are written by concurrent branches but carry no reducer; "
        "LangGraph will raise InvalidUpdateError under fan-out"
    )


def test_no_key_carries_a_reducer_without_being_declared_concurrent() -> None:
    """The reverse direction, and it is not pedantry: a reducer on a key only
    one node writes means either the key is concurrent and FAN_OUT_KEYS is
    stale, or the reducer is silently combining values that should replace."""
    extra = reduced_keys() - FAN_OUT_KEYS
    assert not extra, f"{sorted(extra)} carry reducers but are not in FAN_OUT_KEYS"


def test_the_known_gap_is_recorded_not_pretended_away() -> None:
    """This pair of tests binds FAN_OUT_KEYS to the annotations. It does NOT
    detect a new fan-out node writing a key nobody added to FAN_OUT_KEYS --
    only LangGraph's runtime error catches that, which is what
    test_graph_skeleton's real fan-out test exists for."""
    assert {"results", "cost"} == FAN_OUT_KEYS
