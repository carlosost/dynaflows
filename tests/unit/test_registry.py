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
from dynaflows.gateway.registry import load_registry, model_family

pytestmark = pytest.mark.deterministic

_TEMPLATE = """
version = "{version}"

[constraints]
require_structured_outputs = true
enforce_verifier_family_diversity = {diversity}

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
    version: str = "1",
) -> Path:
    path.write_text(
        _TEMPLATE.format(
            version=version,
            diversity=diversity,
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


def test_the_shipped_config_parses_and_is_honestly_unpopulated() -> None:
    """The real config/models.toml must load, and must admit it is empty.

    AP-19: if this file ever claims populated tiers that do not exist, the
    doctor's WARN becomes a lie. Assert the shipped artifact, not a fixture.
    """
    from dynaflows.settings import get_settings

    registry = load_registry(get_settings(environ={}).models_config)
    assert set(registry.unpopulated) == set(Tier)
    assert registry.require_structured_outputs is True
    assert registry.enforce_verifier_family_diversity is True
