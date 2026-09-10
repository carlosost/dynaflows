"""Contract tests for the models.toml loader (ADR-006).

The rule under test that actually matters is verifier family diversity: a
verifier drawn from the worker's own family shares its failure modes and
rubber-stamps, so the check has to fire on the *preferred* verifier model even
when the rest of the chain is diverse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.registry import context_violations, load_registry, model_family

pytestmark = pytest.mark.deterministic

_TEMPLATE = """
version = "{version}"

[constraints]
require_structured_outputs = true
enforce_verifier_family_diversity = {diversity}
min_context_tokens = {floor}

[tiers.small]
chain = [{small}]
[tiers.mid]
chain = [{mid}]
[tiers.mid_high]
chain = [{mid_high}]
[tiers.frontier]
chain = [{frontier}]
"""


def _write(
    path: Path,
    *,
    small: str = "",
    mid: str = "",
    mid_high: str = "",
    frontier: str = "",
    diversity: str = "true",
    floor: str = "0",
    version: str = "1",
) -> Path:
    path.write_text(
        _TEMPLATE.format(
            version=version,
            diversity=diversity,
            floor=floor,
            small=small,
            mid=mid,
            mid_high=mid_high,
            frontier=frontier,
        ),
        encoding="utf-8",
    )
    return path


def test_model_family_splits_on_the_provider_prefix() -> None:
    assert model_family("acme/model-x") == "acme"
    assert model_family("ACME/Model-X") == "acme"
    assert model_family("bare-model") == "bare-model"


def test_empty_chains_are_reported_as_unpopulated(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml))
    assert set(registry.unpopulated) == set(Tier)


def test_preferred_on_an_empty_chain_raises_config_invalid(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml))
    with pytest.raises(DynaflowsError) as excinfo:
        _ = registry.chain(Tier.SMALL).preferred
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_populated_registry_exposes_preferred_and_families(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml, mid='"acme/fast", "bolt/cheap"', version="7"))
    chain = registry.chain(Tier.MID)
    assert chain.preferred == "acme/fast"
    assert chain.families == ("acme", "bolt")
    assert registry.version == "7"


def test_diversity_violation_when_verifier_shares_the_worker_family(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml, mid='"acme/fast"', mid_high='"acme/deep"'))
    violation = registry.diversity_violation()
    assert violation is not None
    assert "acme" in violation


def test_no_violation_when_families_differ(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml, mid='"acme/fast"', mid_high='"bolt/deep"'))
    assert registry.diversity_violation() is None


def test_diversity_check_is_skipped_while_a_chain_is_empty(models_toml: Path) -> None:
    """An unpopulated registry is not yet wrong; it is not yet answerable."""
    registry = load_registry(_write(models_toml, mid='"acme/fast"'))
    assert registry.diversity_violation() is None


def test_diversity_can_be_turned_off_explicitly(models_toml: Path) -> None:
    registry = load_registry(
        _write(models_toml, mid='"acme/fast"', mid_high='"acme/deep"', diversity="false")
    )
    assert registry.diversity_violation() is None


def test_duplicate_models_in_a_chain_are_rejected(models_toml: Path) -> None:
    with pytest.raises(DynaflowsError) as excinfo:
        load_registry(_write(models_toml, mid='"acme/fast", "acme/fast"'))
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_missing_tier_section_is_rejected(models_toml: Path) -> None:
    models_toml.write_text('version = "1"\n[tiers.small]\nchain = []\n', encoding="utf-8")
    with pytest.raises(DynaflowsError) as excinfo:
        load_registry(models_toml)
    assert "tiers.mid" in excinfo.value.envelope.message


def test_malformed_toml_is_reported_as_config_invalid(models_toml: Path) -> None:
    models_toml.write_text("this is not toml [[[", encoding="utf-8")
    with pytest.raises(DynaflowsError) as excinfo:
        load_registry(models_toml)
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_missing_file_is_reported_as_config_invalid(tmp_path: Path) -> None:
    with pytest.raises(DynaflowsError) as excinfo:
        load_registry(tmp_path / "absent.toml")
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_the_shipped_config_is_internally_consistent() -> None:
    """The real config/models.toml must load and must not misrepresent itself.

    This deliberately does NOT assert that the tiers are empty. An earlier
    version did, which made the deterministic tier fail the moment someone
    populated the registry -- i.e. the moment the project was used correctly.
    A test that encodes a transient state as a permanent invariant is worse
    than no test: it punishes the intended next step.

    What is actually invariant is *consistency*: the version string and the
    chains have to tell the same story, and a populated registry has to
    satisfy ADR-006.
    """
    from dynaflows.settings import get_settings

    registry = load_registry(get_settings(environ={}).models_config)

    # ADR-006's constraints are not optional.
    assert registry.require_structured_outputs is True
    assert registry.enforce_verifier_family_diversity is True

    claims_unpopulated = "unpopulated" in registry.version
    actually_unpopulated = set(registry.unpopulated) == set(Tier)
    assert claims_unpopulated == actually_unpopulated, (
        f"version={registry.version!r} disagrees with the chains: "
        f"unpopulated tiers = {[t.value for t in registry.unpopulated]}"
    )

    if not registry.unpopulated:
        assert registry.diversity_violation() is None, registry.diversity_violation()


# --------------------------------------------------------------------------
# ADR-006, second dimension: context length is a capability.
# Added after a real config paired a 400k primary with an 8k fallback.
# --------------------------------------------------------------------------


def test_the_check_is_disabled_when_the_floor_is_zero(models_toml: Path) -> None:
    """0 is the shipped default and must mean 'not measured', never 'passing'."""
    registry = load_registry(_write(models_toml, frontier='"acme/big", "acme/tiny"'))
    assert registry.min_context_tokens == 0
    assert context_violations(registry, {"acme/big": 400_000, "acme/tiny": 8_000}) == []


def test_a_model_below_the_floor_is_a_violation(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml, frontier='"acme/big", "acme/tiny"', floor="32000"))
    violations = context_violations(registry, {"acme/big": 400_000, "acme/tiny": 8_000})
    assert len(violations) == 1
    assert "acme/tiny" in violations[0]
    assert "frontier" in violations[0]


def test_a_deliberately_large_primary_does_not_punish_sane_fallbacks(
    models_toml: Path,
) -> None:
    """The regression that motivated switching from a ratio to an absolute
    floor: a 10M-token primary made every ordinary fallback look like a
    violation, and a gate that fires on ordinary work gets disabled."""
    registry = load_registry(
        _write(models_toml, mid_high='"acme/huge", "bolt/large"', floor="128000")
    )
    assert context_violations(registry, {"acme/huge": 10_000_000, "bolt/large": 2_000_000}) == []


def test_every_chain_member_is_checked_including_the_primary(models_toml: Path) -> None:
    registry = load_registry(_write(models_toml, mid='"acme/tiny", "bolt/tiny"', floor="32000"))
    assert len(context_violations(registry, {"acme/tiny": 8_000, "bolt/tiny": 4_000})) == 2


def test_a_model_absent_from_the_catalogue_is_not_our_finding(models_toml: Path) -> None:
    """Absence means the tier-capability check should report it. Guessing a
    context length here would hide that with a different error."""
    registry = load_registry(_write(models_toml, mid='"acme/unknown"', floor="32000"))
    assert context_violations(registry, {}) == []


def test_diversity_now_catches_a_shared_family_anywhere_in_the_chain(
    models_toml: Path,
) -> None:
    """The old rule only compared the verifier's *preferred* model, so a
    verifier that fell back into the worker's family passed -- silently, and
    exactly when the fallback mattered."""
    registry = load_registry(
        _write(models_toml, mid='"acme/fast"', mid_high='"bolt/deep", "acme/deep"')
    )
    violation = registry.diversity_violation()
    assert violation is not None
    assert "acme" in violation
