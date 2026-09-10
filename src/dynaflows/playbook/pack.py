"""Deterministic context assembly. ADR-009, layer 3.

The contract that matters: **the same inputs produce the same bytes.** Two runs
that differ can then be diffed, which is the whole reason retrieval here is
lookup-first rather than embedding-based.
"""

from __future__ import annotations

from collections.abc import Iterable

from dynaflows.contracts.playbook import Chunk, ContextPack
from dynaflows.playbook.tokens import estimate_tokens

_SEPARATOR = "\n\n---\n\n"


def _truncate_to_budget(chunk: Chunk, budget: int) -> str | None:
    """Longest prefix of the rendered chunk that fits, cut at a paragraph break.

    Returns None when not even the heading plus one paragraph fits, because a
    heading with no body is worse than an honest omission -- it looks like the
    node was shown the section.
    """
    header, _, body = chunk.render().partition("\n\n")
    kept: list[str] = []
    for paragraph in body.split("\n\n"):
        candidate = "\n\n".join([header, *kept, paragraph])
        if estimate_tokens(candidate) > budget:
            break
        kept.append(paragraph)
    if not kept:
        return None
    return "\n\n".join([header, *kept, "[… section truncated to fit the context budget]"])


def pack(chunks: Iterable[Chunk], budget_tokens: int) -> ContextPack:
    """Fill the budget in the order given; record everything left out.

    Ordering is the caller's responsibility and is meaningful: anchor hits are
    passed before search hits, so an exact request survives a tight budget and
    a fuzzy one is what gets dropped.

    One chunk at most is truncated -- the first that does not fit whole. Every
    chunk after it is dropped rather than squeezed, so the result is a prefix of
    the requested order and never a surprising subset of it.
    """
    ordered: list[Chunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id not in seen:
            seen.add(chunk.id)
            ordered.append(chunk)

    rendered: list[str] = []
    included: list[str] = []
    dropped: list[str] = []
    truncated_id: str | None = None
    used = 0

    for position, chunk in enumerate(ordered):
        if truncated_id is not None:
            dropped.append(chunk.id)
            continue
        text = chunk.render()
        cost = estimate_tokens(text) + (estimate_tokens(_SEPARATOR) if rendered else 0)
        if used + cost <= budget_tokens:
            rendered.append(text)
            included.append(chunk.id)
            used += cost
            continue

        remaining = budget_tokens - used - (estimate_tokens(_SEPARATOR) if rendered else 0)
        partial = _truncate_to_budget(chunk, remaining) if remaining > 0 else None
        if partial is not None:
            rendered.append(partial)
            included.append(chunk.id)
            truncated_id = chunk.id
            used += estimate_tokens(partial)
        else:
            dropped.append(chunk.id)
            # Nothing after this fits either: the list is size-ordered by
            # nothing, but the budget is already exhausted for whole chunks.
            dropped.extend(c.id for c in ordered[position + 1 :])
            break

    text = _SEPARATOR.join(rendered)
    return ContextPack(
        text=text,
        tokens=estimate_tokens(text),
        included_ids=tuple(included),
        dropped_ids=tuple(dict.fromkeys(dropped)),
        truncated_id=truncated_id,
    )
