"""Retrieval contracts. ADR-009.

Declared here rather than inside `playbook/` so the worker node can depend on
the shapes without importing the indexer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Chunk:
    """One leaf section of a markdown document.

    `heading_path` is the whole ancestry, joined with " > ". It is prepended to
    the body at pack time, which is the cheapest possible context restoration
    and the only part that survives truncation intact.
    """

    id: str  # sha256(source_path + heading_path)[:16] -- stable across rebuilds
    source_path: str
    heading_path: str
    anchors: tuple[str, ...]
    body: str
    tokens: int

    @property
    def title(self) -> str:
        return self.heading_path.rsplit(" > ", 1)[-1]

    def render(self) -> str:
        return f"### {self.heading_path}\n\n{self.body}".rstrip()


@dataclass(frozen=True, slots=True)
class ContextPack:
    """What a node actually saw, and what it did not.

    `dropped_ids` and `truncated_id` are as important as the text: they go into
    the LangSmith trace (ADR-011), which is what makes "why did this worker
    miss that rule" answerable from the trace alone rather than by guessing.
    """

    text: str
    tokens: int
    included_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    truncated_id: str | None = None

    @property
    def complete(self) -> bool:
        return not self.dropped_ids and self.truncated_id is None


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Three states, counted apart (AP-20).

    A source that changed, a source nobody has indexed yet, and an indexed
    source that no longer exists are different facts needing different actions.
    Collapsing them into one "out of date" number answers none of them.
    """

    changed: tuple[str, ...] = field(default_factory=tuple)
    added: tuple[str, ...] = field(default_factory=tuple)
    removed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not (self.changed or self.added or self.removed)

    def summary(self) -> str:
        if self.clean:
            return "index matches every source"
        parts = []
        if self.changed:
            parts.append(f"{len(self.changed)} changed: {', '.join(self.changed)}")
        if self.added:
            parts.append(f"{len(self.added)} unindexed: {', '.join(self.added)}")
        if self.removed:
            parts.append(f"{len(self.removed)} indexed but missing: {', '.join(self.removed)}")
        return "; ".join(parts)
