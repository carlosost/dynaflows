"""What a gate does with the answer a human gave it.

One module because there are now three gates across two pipelines (G1 and G2
in `nodes.py`, G3 in `write_nodes.py`) and they must agree about what counts
as approval. Two copies of this function would be two chances to disagree
about whether an unparseable answer means yes -- and the copy that said yes
would be the one that authorised a spend nobody approved.
"""

from __future__ import annotations

from typing import Any

from dynaflows.contracts.state import GateDecision, GateOutcome

__all__ = ["gate_outcome"]


def gate_outcome(answer: Any) -> GateOutcome:
    """Normalise whatever `Command(resume=...)` carried.

    A bare string is accepted so a human answering "approve" at a terminal is
    not a crash, but anything unrecognised is a REJECT: defaulting an
    unparseable answer to approval would let a typo authorise a fan-out -- or,
    at G3, a write into the user's working tree.
    """
    if isinstance(answer, GateOutcome):
        return answer
    if isinstance(answer, dict):
        return GateOutcome.model_validate(answer)
    if isinstance(answer, str) and answer.strip().lower() in set(GateDecision):
        return GateOutcome(decision=GateDecision(answer.strip().lower()))
    return GateOutcome(decision=GateDecision.REJECT, note=f"unparseable gate answer: {answer!r}")
