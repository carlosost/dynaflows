"""Prompt text and the schemas the models must satisfy.

Kept beside each other on purpose: a prompt and the shape it demands are one
decision, and splitting them is how they drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ENHANCER_SYSTEM = """\
You rewrite a developer's request into a precise, self-contained brief for an \
analysis workflow. You do not answer the request.

Rules:
- Preserve the user's intent exactly. Never widen or narrow the scope.
- Make implicit constraints explicit.
- Name what you had to assume, rather than silently choosing.
- If the request is already precise, return it close to unchanged and say so.
- No preamble, no meta-commentary about the rewrite itself.\
"""


class EnhancedPrompt(BaseModel):
    """What the enhancer returns. The human approves this, so it has to be
    readable at a glance, not a wall of text."""

    enhanced: str = Field(description="The rewritten brief. Self-contained.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything you had to assume. One short line each. Empty if none.",
    )
