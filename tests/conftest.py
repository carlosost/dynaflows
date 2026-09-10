"""Shared fixtures.

Playbook 2.2: the deterministic tier touches no network, no provider and no
filesystem outside tmp. Nothing here reads os.environ or the real .env.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal project root: just the marker file settings.py looks for."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    return tmp_path


@pytest.fixture
def models_toml(project: Path) -> Path:
    return project / "config" / "models.toml"
