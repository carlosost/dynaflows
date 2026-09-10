"""Doctor checks that need no network.

The sqlite check is the interesting one: it is the only automated place where
ADR-009's premise -- that SQLite ships FTS5 with bm25(), so retrieval needs no
new dependency -- gets re-verified on whatever machine is running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.doctor import Status, check_models_config, check_state_dir, run_checks
from dynaflows.settings import get_settings

pytestmark = pytest.mark.deterministic


def test_sqlite_check_confirms_fts5_and_bm25_on_this_machine(project: Path) -> None:
    settings = get_settings({"DYNAFLOWS_HOME": str(project / "state")}, root=project)
    check = check_state_dir(settings)
    assert check.status is Status.OK, check.detail
    assert "FTS5" in check.detail


def test_sqlite_check_leaves_no_probe_file_behind(project: Path) -> None:
    home = project / "state"
    settings = get_settings({"DYNAFLOWS_HOME": str(home)}, root=project)
    check_state_dir(settings)
    assert list(home.iterdir()) == []


def test_models_config_warns_rather_than_fails_when_unpopulated(project: Path) -> None:
    """OQ-01 open is a known state, not a broken one. AP-20: do not merge
    'not configured yet' with 'misconfigured' -- they need different actions."""
    (project / "config" / "models.toml").write_text(
        'version = "0"\n[tiers.small]\nchain=[]\n[tiers.mid]\nchain=[]\n'
        "[tiers.mid_high]\nchain=[]\n[tiers.frontier]\nchain=[]\n",
        encoding="utf-8",
    )
    settings = get_settings({}, root=project)
    check = check_models_config(settings)
    assert check.status is Status.WARN
    assert "--suggest" in check.detail


def test_models_config_fails_on_a_diversity_violation(project: Path) -> None:
    (project / "config" / "models.toml").write_text(
        'version = "1"\n'
        '[tiers.small]\nchain=["acme/s"]\n'
        '[tiers.mid]\nchain=["acme/m"]\n'
        '[tiers.mid_high]\nchain=["acme/h"]\n'
        '[tiers.frontier]\nchain=["acme/f"]\n',
        encoding="utf-8",
    )
    settings = get_settings({}, root=project)
    check = check_models_config(settings)
    assert check.status is Status.FAIL
    assert "ADR-006" in check.detail


def test_offline_mode_runs_no_network_check(project: Path) -> None:
    (project / "config" / "models.toml").write_text(
        'version = "0"\n[tiers.small]\nchain=[]\n[tiers.mid]\nchain=[]\n'
        "[tiers.mid_high]\nchain=[]\n[tiers.frontier]\nchain=[]\n",
        encoding="utf-8",
    )
    settings = get_settings({}, root=project)
    names = [c.name for c in run_checks(settings, offline=True)]
    assert names == ["environment", "sqlite", "models.toml"]


def test_environment_check_fails_loudly_with_no_keys(project: Path) -> None:
    checks = {c.name: c for c in run_checks(get_settings({}, root=project), offline=True)}
    assert checks["environment"].status is Status.FAIL
    assert "OPENROUTER_API_KEY" in checks["environment"].detail
