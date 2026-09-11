"""ADR-021: the budget that starved run `s1`.

Seven files, 19,979 tokens of source, a 6,000-token budget, and a worker that
received one file and honestly reported it could not see the other six. Every
layer behaved correctly and the run produced nothing, which is what makes this
worth a file of tests rather than a changed constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dynaflows.contracts.playbook import Chunk
from dynaflows.graph.budgets import (
    DEFAULT_CONTEXT_FLOOR,
    WORKER_OVERHEAD_TOKENS,
    worker_context_budget,
)
from dynaflows.playbook.pack import pack, pack_sections

pytestmark = pytest.mark.deterministic


@dataclass(frozen=True)
class _Registry:
    min_context_tokens: int


def chunks(prefix: str, count: int, tokens: int) -> list[Chunk]:
    # Sized against what pack() actually measures -- the RENDERED text, not the
    # chunk's own `tokens` field. The first version of this helper set the
    # field and built a body five times larger, so every "1,000-token" chunk
    # was 5,500 and the test asserted a budget nothing could meet.
    from dynaflows.playbook.tokens import CHARS_PER_TOKEN_ESTIMATE

    body = "word " * int(tokens * CHARS_PER_TOKEN_ESTIMATE / 5)
    return [
        Chunk(
            id=f"{prefix}{i}",
            source_path=f"{prefix}{i}.py",
            heading_path=f"{prefix}{i}.py",
            anchors=(f"{prefix}{i}",),
            body=body,
            tokens=tokens,
        )
        for i in range(count)
    ]


# --- the budget is derived ------------------------------------------------


def test_the_budget_follows_the_configured_floor() -> None:
    """Raising min_context_tokens raises the budget, in the one place an
    operator already looks."""
    small = worker_context_budget(_Registry(32_000))
    large = worker_context_budget(_Registry(128_000))

    assert large > small
    assert small < 32_000 - WORKER_OVERHEAD_TOKENS


def test_an_unset_floor_still_produces_a_working_budget() -> None:
    """`min_context_tokens = 0` is legitimate -- doctor reports it WARN, never
    OK -- so this path must work rather than yield zero."""
    assert worker_context_budget(_Registry(0)) == worker_context_budget(
        _Registry(DEFAULT_CONTEXT_FLOOR)
    )


def test_a_missing_registry_does_not_crash_a_worker() -> None:
    """A worker must never raise. A budget lookup is not worth an exception
    inside a Send branch."""
    assert worker_context_budget(None) > 0


def test_a_misconfigured_floor_still_leaves_room_to_work() -> None:
    assert worker_context_budget(_Registry(100)) >= 2_000


# --- two shares of one budget ---------------------------------------------


def test_the_subject_is_not_starved_by_the_reference() -> None:
    """Run `s1` in miniature. Ordered packing gives the reference everything it
    asks for; the split gives the subject most of the budget."""
    reference = chunks("ref", 4, 1_000)
    subject = chunks("src", 7, 1_000)

    ordered = pack([*reference, *subject], 5_000)
    split = pack_sections(reference, subject, 5_000)

    subject_ids = {c.id for c in subject}
    # Ordered packing spends the budget on the reference and reaches the
    # subject with nothing left -- which is worse than what `s1` did, and `s1`
    # produced no findings at all.
    assert len(subject_ids & set(ordered.included_ids)) <= 1
    assert len(subject_ids & set(split.included_ids)) >= 3


def test_the_reference_cannot_take_more_than_its_share() -> None:
    reference = chunks("ref", 20, 1_000)
    subject = chunks("src", 4, 1_000)

    split = pack_sections(reference, subject, 10_000, reference_share=0.25)

    included_reference = [i for i in split.included_ids if i.startswith("ref")]
    assert len(included_reference) <= 3


def test_an_unused_reference_share_goes_to_the_subject() -> None:
    """A cap, not a reservation. Holding back budget nothing will use would
    drop a file for no reason."""
    reference = chunks("ref", 1, 200)
    subject = chunks("src", 9, 1_000)

    split = pack_sections(reference, subject, 10_000)

    assert len([i for i in split.included_ids if i.startswith("src")]) >= 8


def test_a_dropped_chunk_is_reported_from_either_share() -> None:
    """The worker is told what it did not get, whichever half lost it."""
    split = pack_sections(chunks("ref", 9, 1_000), chunks("src", 9, 1_000), 6_000)

    assert any(i.startswith("ref") for i in split.dropped_ids)
    assert any(i.startswith("src") for i in split.dropped_ids)


def test_one_kind_missing_is_not_a_special_case() -> None:
    only_subject = pack_sections([], chunks("src", 3, 500), 10_000)
    only_reference = pack_sections(chunks("ref", 3, 500), [], 10_000)

    assert len(only_subject.included_ids) == 3
    assert len(only_reference.included_ids) == 3
