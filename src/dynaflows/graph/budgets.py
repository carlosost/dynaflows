"""How much context a node may spend. Derived, not chosen.

`WORKER_CONTEXT_BUDGET = 6_000` was a placeholder invented in step 1.6 and it
was wrong by a factor of four. Live run `s1` gave a worker seven files totalling
19,979 tokens and a budget that fitted one of them; the worker honestly
reported that it could not see the other six, and the run produced nothing.

The replacement is derived from a number the operator has already declared:
`min_context_tokens` in models.toml is the floor every configured model is
checked against by `doctor`. If every model is guaranteed that much window,
that much window minus what the call itself needs is what a node may fill.
"""

from __future__ import annotations

# What a worker call spends on things that are not retrieved context: the
# system prompt (~800), the objective and refusal list (~400), and the output
# allowance (2,048). Rounded up, because underestimating this truncates the
# prompt at the provider rather than here.
WORKER_OVERHEAD_TOKENS = 4_000

# Used when the operator has not set a floor. `min_context_tokens = 0` is a
# legitimate configuration -- `doctor` reports it as WARN, never OK -- so this
# path must work, and it is deliberately conservative: 32,000 is the floor the
# project documents, and half of it is a budget that fits on any current model.
DEFAULT_CONTEXT_FLOOR = 32_000

# Never fill the whole window. A provider counting tokens differently than our
# estimator is the normal case, not the exception (CHARS_PER_TOKEN_ESTIMATE is
# itself a placeholder), and the failure mode of being wrong is a truncated
# prompt at the provider, which is silent.
SAFETY_FRACTION = 0.75


def worker_context_budget(registry: object) -> int:
    """The token budget for one worker's retrieved context.

    Reads the registry rather than taking a number, so raising the floor in
    models.toml raises the budget -- one place, and the place an operator
    already looks.
    """
    floor = getattr(registry, "min_context_tokens", 0) or DEFAULT_CONTEXT_FLOOR
    usable = int((floor - WORKER_OVERHEAD_TOKENS) * SAFETY_FRACTION)
    # A floor small enough to make this negative is a misconfiguration doctor
    # reports; here it must still produce a working, if tiny, budget.
    return max(usable, 2_000)
