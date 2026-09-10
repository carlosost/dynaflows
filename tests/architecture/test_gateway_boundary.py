"""ADR-010's proof. Named by the ADR, per AP-19 habit 1.

Two halves, and both matter:
  1. The real source tree has no violations.
  2. The linter actually detects one. A rule that has never failed is a rule
     nobody has tested -- it is the "green but never runs" shape of AP-11
     applied to enforcement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from lint_architecture import GATEWAY_ONLY_MODULES, check_file, find_project_root  # noqa: E402

pytestmark = pytest.mark.deterministic

PROJECT_ROOT = find_project_root(Path(__file__).parent)


def test_no_provider_sdk_is_imported_outside_the_gateway() -> None:
    offenders = [
        violation
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        for violation in check_file(path, PROJECT_ROOT)
    ]
    assert offenders == [], "\n".join(offenders)


def test_the_gateway_itself_is_allowed_to_import_a_provider_sdk() -> None:
    probe = PROJECT_ROOT / "src" / "dynaflows" / "gateway" / "probe.py"
    assert "import httpx" in probe.read_text(encoding="utf-8")
    assert check_file(probe, PROJECT_ROOT) == []


@pytest.mark.parametrize("module", sorted(GATEWAY_ONLY_MODULES))
def test_the_linter_catches_a_violation_of_each_banned_module(module: str, tmp_path: Path) -> None:
    offender = tmp_path / "src" / "dynaflows" / "nodes" / "worker.py"
    offender.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    offender.write_text(f"import {module}\n", encoding="utf-8")

    violations = check_file(offender, tmp_path)
    assert len(violations) == 1
    assert "ADR-010" in violations[0]
    assert module in violations[0]


def test_from_imports_are_caught_too(tmp_path: Path) -> None:
    offender = tmp_path / "src" / "dynaflows" / "nodes" / "planner.py"
    offender.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    offender.write_text("from langchain_openai import ChatOpenAI\n", encoding="utf-8")
    assert len(check_file(offender, tmp_path)) == 1


def test_unrelated_imports_are_left_alone(tmp_path: Path) -> None:
    clean = tmp_path / "src" / "dynaflows" / "nodes" / "ok.py"
    clean.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    clean.write_text("import json\nfrom pathlib import Path\n", encoding="utf-8")
    assert check_file(clean, tmp_path) == []
