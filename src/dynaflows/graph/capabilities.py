"""The worker capability catalogue. ADR-001.

The planner does not invent node types; it names a capability from this list.
That is what makes the "dynamic" part of this workflow *plan data* rather than
generated topology, and it is the reason a plan can be validated at all.

**Phase 1 registers exactly ONE capability**, and the smallness is the point.
AP-11: an abstraction is built when its caller exists, not before. `analyse`
is what step 1.6's worker will actually implement — read the named inputs plus
the retrieved playbook context, and report findings. Registering `review`,
`compare`, `search` and friends would create surface with no implementation
behind it, and the planner would happily emit tasks nothing can run.

A catalogue of one is still doing real work: `capability` becomes a validated
identifier instead of free text, and a twelve-way fan-out of `analyse` tasks
with different objectives, inputs and anchors is exactly Fan-out-and-Synthesize.
`verify` arrives in Phase 2 with the node that performs it.
"""

from __future__ import annotations

from dataclasses import dataclass

from dynaflows.contracts.tiers import Tier


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    tier: Tier
    summary: str
    produces: str


CATALOGUE: tuple[Capability, ...] = (
    Capability(
        id="analyse",
        tier=Tier.MID,
        summary=(
            "Read the named inputs and the retrieved playbook sections, then report "
            "findings against the objective. Read-only: never edits, runs or deletes "
            "anything (ADR-016)."
        ),
        produces="A findings report: what was examined, what was found, and the evidence.",
    ),
)

# Bumped whenever the catalogue changes. It is hashed into plan_hash, so a plan
# is never compared across incompatible catalogues (ADR-001).
CATALOGUE_VERSION = "1"

CAPABILITY_IDS = frozenset(c.id for c in CATALOGUE)


def render_catalogue() -> str:
    """What the planner is shown. Compact: it is paid for on every plan call."""
    return "\n".join(f"- {c.id}: {c.summary} Produces: {c.produces}" for c in CATALOGUE)
