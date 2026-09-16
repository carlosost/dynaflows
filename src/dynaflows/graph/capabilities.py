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
from typing import Literal

from dynaflows.contracts.tiers import Tier


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    tier: Tier
    summary: str
    produces: str
    # WHICH pipeline can run this (ADR-023). Not decoration: the planner is
    # shown a catalogue and picks from it, so a catalogue that lists every
    # capability regardless of pipeline lets the READ planner emit an
    # `implement` task that no read worker implements -- a plan that validates
    # and then dispatches a branch which can only fail, N supersteps later and
    # far from the call that caused it. The same reasoning as the
    # `_capability_must_be_registered` validator, one level up.
    pipeline: Literal["read", "write"] = "read"


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
    Capability(
        id="implement",
        # The tier is not used: the executor hands the work to a coding agent
        # with its own model, chosen by `claude`, not by `models.toml`. Set to
        # FRONTIER so that a future reader who wires it to the gateway by
        # mistake gets the expensive-and-correct default rather than a cheap
        # model silently attempting a code change.
        tier=Tier.FRONTIER,
        summary=(
            "Make the change described by the objective, in an isolated git "
            "worktree, using a coding agent with its own tools and permissions "
            "(ADR-025). WRITES files. Use when the user asked for the codebase "
            "to be DIFFERENT, not for it to be explained or audited."
        ),
        produces="A diff, a test verdict comparing before and after, and the agent's account of what it did.",
        pipeline="write",
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
# 3 (2026-09-16): `implement` registered, and every capability gained a
# `pipeline`. The field arrived with its first real need: registering a
# capability that writes made it possible for the READ planner to emit one,
# and the only thing that had been preventing that was the catalogue having
# no write capability in it yet.
CAPABILITIES_VERSION = "3"

CAPABILITY_IDS = frozenset(c.id for c in CAPABILITIES)


def capabilities_for(pipeline: str) -> tuple[Capability, ...]:
    return tuple(c for c in CAPABILITIES if c.pipeline == pipeline)


def render_capabilities(pipeline: str = "read") -> str:
    """What the planner is shown. Compact: it is paid for on every plan call.

    Filtered by pipeline, so the READ planner is never offered a capability
    that writes and the WRITE planner is never offered one that only reads.
    A planner cannot choose something it was not shown, which is cheaper and
    more reliable than a prompt asking it not to.
    """
    return "\n".join(
        f"- {c.id}: {c.summary} Produces: {c.produces}" for c in capabilities_for(pipeline)
    )
