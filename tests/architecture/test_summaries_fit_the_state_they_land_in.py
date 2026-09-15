"""Every schema field that becomes `WorkerResult.summary` must fit it.

A live `answer` worker wrote 1,300 characters into `AnswerReport.answer`,
which is capped at 2,000 and assigned to `WorkerResult.summary`, capped at
1,200. Pydantic refused the assignment, the `ValidationError` escaped a
LangGraph `Send` branch, and the whole superstep died -- "one task failed"
became "the run is gone", which the worker's own docstring names as the
worst outcome available in this design.

Two contracts that must agree, in two files, with nothing checking that they
did. Adding a third capability would have been a third chance to get it
wrong, so the rule is asserted rather than remembered.
"""

from __future__ import annotations

import pytest

from dynaflows.contracts.state import WorkerResult
from dynaflows.graph.nodes import CAPABILITY_HANDLERS
from dynaflows.graph.prompts import AnswerReport, WorkerReport

pytestmark = pytest.mark.deterministic


def _cap(model: type, field: str) -> int | None:
    for entry in model.model_fields[field].metadata:
        limit = getattr(entry, "max_length", None)
        if limit is not None:
            return int(limit)
    return None


def test_the_state_field_is_capped_at_all() -> None:
    """If this cap were removed the tests below would pass vacuously, which
    is a worse outcome than the bug they describe."""
    assert _cap(WorkerResult, "summary") == 1200


@pytest.mark.parametrize(("model", "field"), [(WorkerReport, "summary"), (AnswerReport, "answer")])
def test_a_report_never_promises_more_than_state_accepts(model: type, field: str) -> None:
    """The model is told a limit; state enforces one. A model told it may
    write 2,000 characters will eventually write more than 1,200, and the
    place that discovers it is a Send branch with no exception handler."""
    limit = _cap(model, field)

    assert limit is not None, f"{model.__name__}.{field} has no cap"
    assert limit <= _cap(WorkerResult, "summary")


def test_every_capability_is_covered_by_this_rule() -> None:
    """A third capability is a third chance to make the same mistake. If one
    is registered whose report is not checked above, this fails."""
    checked = {WorkerReport, AnswerReport}
    registered = {handler.schema for handler in CAPABILITY_HANDLERS.values()}

    assert registered == checked, (
        f"a capability's report schema is not covered by the cap test above: {registered ^ checked}"
    )
