"""config/models.toml -> typed tier chains. ADR-006.

This module reads configuration only. It makes no network calls; validating a
chain against the live provider catalogue is `probe.py`, because that needs a
key and a network and this needs neither -- which is what makes this testable
in the deterministic tier.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.tiers import Tier

_TIER_KEYS: dict[Tier, str] = {
    Tier.SMALL: "small",
    Tier.MID: "mid",
    Tier.MID_HIGH: "mid_high",
    Tier.FRONTIER: "frontier",
}


def model_family(model_id: str) -> str:
    """The provider/family prefix of an OpenRouter id.

    'anthropic/claude-x' -> 'anthropic'. Used for ADR-006's diversity rule:
    a verifier drawn from the worker's family shares its failure modes.
    """
    head = model_id.split("/", 1)[0] if "/" in model_id else model_id
    return head.strip().lower()


@dataclass(frozen=True, slots=True)
class TierChain:
    tier: Tier
    purpose: str
    chain: tuple[str, ...]

    @property
    def is_populated(self) -> bool:
        return bool(self.chain)

    @property
    def preferred(self) -> str:
        if not self.chain:
            raise DynaflowsError.of(
                ErrorCode.CONFIG_INVALID,
                f"tier '{self.tier}' has an empty chain; run `dynaflows models --suggest`",
            )
        return self.chain[0]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(model_family(m) for m in self.chain))


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    version: str
    require_structured_outputs: bool
    enforce_verifier_family_diversity: bool
    min_context_tokens: int
    tiers: dict[Tier, TierChain]

    def chain(self, tier: Tier) -> TierChain:
        return self.tiers[tier]

    @property
    def unpopulated(self) -> tuple[Tier, ...]:
        return tuple(t for t, c in self.tiers.items() if not c.is_populated)

    def diversity_violation(self) -> str | None:
        """ADR-006: the verifier tier must not share a family with workers.

        Returns a human-readable reason, or None when the rule holds or cannot
        yet be evaluated because a chain is empty.
        """
        if not self.enforce_verifier_family_diversity:
            return None
        worker = self.tiers[Tier.MID]
        verifier = self.tiers[Tier.MID_HIGH]
        if not (worker.is_populated and verifier.is_populated):
            return None
        shared = set(worker.families) & set(verifier.families)
        if shared:
            return (
                f"verifier chain shares family {sorted(shared)!r} with the worker chain; "
                f"ADR-006 requires a different family. A verifier that falls back into the "
                f"worker's family inherits its blind spots exactly when it matters most."
            )
        return None


def context_violations(registry: ModelRegistry, context: Mapping[str, int]) -> list[str]:
    """Chain entries whose context window is below the configured floor.

    ADR-006 requires capability-homogeneous chains, and context length is a
    capability: an 8k fallback behind a 200k primary fails on the exact prompts
    the primary was chosen for.

    The floor is ABSOLUTE, not a ratio of the primary. A ratio was tried first
    and was wrong: it makes a deliberately large primary punish every sane
    fallback, and a gate that fires on ordinary work gets disabled -- after
    which nothing is protected (playbook 5.2, Pattern 5).

    `min_context_tokens = 0` disables the check. That is the honest default
    until the number is measured: 4.5 says implement and test the mechanism
    first, and treat the threshold as provisional until real traffic sets it.

    Pure by design -- it takes the model->context map rather than fetching it,
    so the rule is provable in the deterministic tier. Network lives in the
    caller.
    """
    floor = registry.min_context_tokens
    if floor <= 0:
        return []
    violations: list[str] = []
    for tier, chain in registry.tiers.items():
        for model_id in chain.chain:
            size = context.get(model_id)
            if size is None:
                continue  # absence is the tier-capability check's finding, not ours
            if size < floor:
                violations.append(
                    f"{tier.value}: '{model_id}' has {size // 1000}k context, "
                    f"below the {floor // 1000}k floor"
                )
    return violations


def load_registry(path: Path) -> ModelRegistry:
    if not path.is_file():
        raise DynaflowsError.of(ErrorCode.CONFIG_INVALID, f"model config not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DynaflowsError.of(ErrorCode.CONFIG_INVALID, f"{path}: {exc}") from exc

    constraints = raw.get("constraints", {})
    tiers_raw = raw.get("tiers", {})

    tiers: dict[Tier, TierChain] = {}
    for tier, key in _TIER_KEYS.items():
        section = tiers_raw.get(key)
        if section is None:
            raise DynaflowsError.of(
                ErrorCode.CONFIG_INVALID, f"{path}: missing section [tiers.{key}]"
            )
        chain = section.get("chain", [])
        if not isinstance(chain, list) or any(not isinstance(m, str) for m in chain):
            raise DynaflowsError.of(
                ErrorCode.CONFIG_INVALID, f"{path}: [tiers.{key}].chain must be a list of strings"
            )
        if len(set(chain)) != len(chain):
            raise DynaflowsError.of(
                ErrorCode.CONFIG_INVALID, f"{path}: [tiers.{key}].chain contains duplicates"
            )
        tiers[tier] = TierChain(
            tier=tier, purpose=str(section.get("purpose", "")), chain=tuple(chain)
        )

    return ModelRegistry(
        version=str(raw.get("version", "0")),
        require_structured_outputs=bool(constraints.get("require_structured_outputs", True)),
        enforce_verifier_family_diversity=bool(
            constraints.get("enforce_verifier_family_diversity", True)
        ),
        min_context_tokens=int(constraints.get("min_context_tokens", 0)),
        tiers=tiers,
    )


def get_model_registry(path: Path | None = None) -> ModelRegistry:
    """Sole construction path (playbook 3.1). Tests patch this, not tomllib."""
    if path is None:
        from dynaflows.settings import get_settings

        path = get_settings().models_config
    return load_registry(path)
