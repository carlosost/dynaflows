"""A reducer that forgets a field loses the fact, not the run.

`calls_unpriced` was added to `CostLedger` and not to `merge_cost`. Nothing
failed. The field simply never survived a merge, so the "the total is a LOWER
BOUND" warning built to surface unpriced calls could not fire in any run with
more than one node -- which is every run.

That is the fourth time in this project a new fact has gone missing because
the place that aggregates it was a hand-written list: zero tokens, zero cost,
`degraded` with no counter, and now this. The pattern is the target, not the
individual field, so this test asserts the PROPERTY -- every field is summed --
rather than naming the fields.
"""

from __future__ import annotations

import dataclasses

import pytest

from dynaflows.contracts.calls import CostLedger
from dynaflows.contracts.state import merge_cost

pytestmark = pytest.mark.deterministic

# Distinct primes so a field summed from the wrong source is visible in the
# total rather than hidden by a coincidence.
_LEFT = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_RIGHT = (41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89)


def _ledger(values: tuple[int, ...]) -> CostLedger:
    return CostLedger(
        **{
            field.name: float(value) if field.type in ("float", float) else value
            for field, value in zip(dataclasses.fields(CostLedger), values, strict=False)
        }
    )


def test_every_ledger_field_survives_a_merge() -> None:
    left, right = _ledger(_LEFT), _ledger(_RIGHT)

    merged = merge_cost(left, right)

    for field in dataclasses.fields(CostLedger):
        expected = getattr(left, field.name) + getattr(right, field.name)
        assert getattr(merged, field.name) == expected, (
            f"merge_cost drops '{field.name}'. A field added to CostLedger and not to the "
            f"reducer is silently discarded at every superstep."
        )


def test_the_reducer_is_not_a_hand_written_list() -> None:
    """The regression guard for the guard.

    A future edit could revert `merge_cost` to explicit additions and still
    pass the test above, until the next field is added -- at which point the
    test fails long after the change that caused it. Asserting the shape keeps
    the failure next to the cause.
    """
    import ast
    import inspect
    import textwrap

    from dynaflows.contracts import state

    # The BODY, not the docstring. The docstring names the field that was
    # dropped, because that is what the docstring is for -- and the first
    # version of this test failed on its own explanation.
    tree = ast.parse(textwrap.dedent(inspect.getsource(state.merge_cost)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if ast.get_docstring(function) else function.body
    source = "\n".join(ast.unparse(node) for node in body)
    named = [f.name for f in dataclasses.fields(CostLedger) if f.name in source]

    assert not named, (
        f"merge_cost names {named} explicitly. Enumerate dataclasses.fields() instead: "
        "a list maintained in two places was maintained in one."
    )


def test_merging_an_empty_ledger_changes_nothing() -> None:
    """LangGraph calls a reducer with whatever is already there, and a node
    that spent nothing still returns a delta."""
    ledger = _ledger(_LEFT)

    assert merge_cost(ledger, CostLedger()) == ledger
    assert merge_cost(CostLedger(), ledger) == ledger
