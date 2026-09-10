"""Context assembly. ADR-009, layer 3.

The contract under test is determinism: the same inputs must produce the same
bytes, or two runs cannot be diffed and the whole lookup-first design loses its
point.
"""

from __future__ import annotations

import pytest

from dynaflows.contracts.playbook import Chunk
from dynaflows.playbook.pack import pack
from dynaflows.playbook.tokens import estimate_tokens

pytestmark = pytest.mark.deterministic


def make(cid: str, heading: str, paragraphs: int = 1, words: int = 20) -> Chunk:
    body = "\n\n".join(" ".join(f"w{i}{j}" for j in range(words)) for i in range(paragraphs))
    return Chunk(
        id=cid,
        source_path="p.md",
        heading_path=heading,
        anchors=(cid,),
        body=body,
        tokens=estimate_tokens(body),
    )


def test_packing_is_byte_identical_across_runs() -> None:
    chunks = [make("a", "A"), make("b", "B"), make("c", "C")]
    assert pack(chunks, 500).text == pack(list(chunks), 500).text


def test_order_given_is_order_kept() -> None:
    """pack() does not re-rank. The caller passes anchor hits before search
    hits, so an exact request survives a tight budget and a fuzzy one does not."""
    packed = pack([make("b", "B"), make("a", "A")], 500)
    assert packed.included_ids == ("b", "a")


def test_duplicates_are_collapsed_keeping_the_first_position() -> None:
    packed = pack([make("a", "A"), make("b", "B"), make("a", "A")], 500)
    assert packed.included_ids == ("a", "b")


def test_the_heading_path_is_always_present() -> None:
    """A truncated chunk still has to say what it is."""
    assert "### A > B > C" in pack([make("a", "A > B > C")], 500).text


def test_a_chunk_too_large_is_truncated_at_a_paragraph_boundary() -> None:
    big = make("a", "A", paragraphs=6, words=40)
    packed = pack([big], 120)
    assert packed.truncated_id == "a"
    assert "section truncated" in packed.text
    assert packed.included_ids == ("a",)


def test_everything_after_a_truncation_is_dropped_not_squeezed() -> None:
    """The result is a prefix of the requested order, never a surprising
    subset of it."""
    packed = pack([make("a", "A", paragraphs=6, words=40), make("b", "B"), make("c", "C")], 120)
    assert packed.included_ids == ("a",)
    assert packed.dropped_ids == ("b", "c")


def test_a_chunk_whose_first_paragraph_does_not_fit_is_dropped_whole() -> None:
    """A heading with no body reads as though the node was shown the section."""
    packed = pack([make("a", "A", paragraphs=4, words=60)], 12)
    assert packed.included_ids == ()
    assert packed.dropped_ids == ("a",)
    assert packed.truncated_id is None


def test_a_generous_budget_includes_everything_and_reports_complete() -> None:
    packed = pack([make("a", "A"), make("b", "B")], 10_000)
    assert packed.included_ids == ("a", "b")
    assert packed.dropped_ids == ()
    assert packed.complete is True


def test_a_truncated_pack_is_not_complete() -> None:
    assert pack([make("a", "A", paragraphs=6, words=40)], 120).complete is False


def test_the_reported_token_count_matches_the_text() -> None:
    packed = pack([make("a", "A"), make("b", "B")], 10_000)
    assert packed.tokens == estimate_tokens(packed.text)


def test_the_separator_cost_is_charged_so_the_budget_is_not_overrun() -> None:
    chunks = [make("a", "A"), make("b", "B"), make("c", "C"), make("d", "D")]
    budget = 60
    packed = pack(chunks, budget)
    assert packed.tokens <= budget


def test_an_empty_input_is_an_empty_pack_not_a_crash() -> None:
    packed = pack([], 500)
    assert packed.text == ""
    assert packed.complete is True
