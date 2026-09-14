"""The worker capability catalogue. ADR-001.

The planner does not invent node types; it names a capability from this list.
That is what makes the "dynamic" part of this workflow *plan data* rather than
generated topology, and it is the reason a plan can be validated at all.

Each entry here has a node that implements it, and nothing else is listed.
AP-11: an abstraction is built when its caller exists, not before. Registering
`review`, `compare`, `search` and friends would create surface with no
implementation behind it, and the planner would happily emit tasks nothing can
run. `verify` arrives in Phase 2 with the node that performs it.

`capability` is therefore a validated identifier rather than free text, and a
twelve-way fan-out with different objectives, inputs and anchors is exactly
Fan-out-and-Synthesize.

The two entries divide on what the user ASKED, and it is a real division
rather than a label: an audit discards a grounded claim that only describes
the code, and an answer to "how does this work" is exactly such a claim.
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


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="analyse",
        tier=Tier.MID,
        summary=(
            "Read the named inputs and the retrieved playbook sections, then report "
            "findings against the objective. Read-only: never edits, runs or deletes "
            "anything (ADR-016). Use when the question is what is WRONG."
        ),
        produces="A findings report: what was examined, what was found, and the evidence.",
    ),
    Capability(
        id="answer",
        tier=Tier.MID,
        summary=(
            "Read the named inputs and the retrieved playbook sections, then answer "
            "the objective as a question about how the code works. Read-only. Use "
            "when the user asked what something DOES, not what is wrong with it."
        ),
        produces="An answer in prose, plus the cited lines that support each claim.",
    ),
)

# Bumped whenever the catalogue changes. It is hashed into plan_hash, so a plan
# is never compared across incompatible catalogues (ADR-001).
#
# 2 (2026-09-13): `answer` registered. It is a separate capability rather than
# a mode of `analyse` because the two disagree about the one thing that
# matters: `analyse` DISCARDS a grounded claim that merely describes the code
# (Ungrounded.NOT_A_DEFECT), and for "how does retrieval work" a description
# of the code is the answer. A finding also requires a failure, a severity and
# a remediation, all three meaningless for a question -- and a model made to
# fill them either invents them or says less, which this project has already
# measured.
CAPABILITIES_VERSION = "2"

CAPABILITY_IDS = frozenset(c.id for c in CAPABILITIES)


def render_capabilities() -> str:
    """What the planner is shown. Compact: it is paid for on every plan call."""
    return "\n".join(f"- {c.id}: {c.summary} Produces: {c.produces}" for c in CAPABILITIES)
