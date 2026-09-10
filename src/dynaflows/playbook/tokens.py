"""Token estimation.

There is no correct answer available here. The tier's model is configurable
and OpenRouter fronts many providers, so the tokenizer that will actually
count these characters is not known at pack time.

The estimate therefore deliberately OVER-counts: packing less than the budget
is safe, packing more silently truncates a prompt at the provider. The constant
is a placeholder (playbook §4.5) with a concrete calibration path -- LangSmith
reports real token counts per call, so step 1.8 compares estimate to actual
across a fixed run and replaces this number with a measured one.
"""

from __future__ import annotations

import math

# English prose runs ~4 chars/token. 3.6 leans conservative on purpose.
CHARS_PER_TOKEN_ESTIMATE = 3.6


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)
