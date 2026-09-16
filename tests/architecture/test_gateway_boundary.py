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

from lint_architecture import (  # noqa: E402
    EXECUTOR_ONLY_MODULES,
    GATEWAY_ONLY_MODULES,
    SPAWN_MODULES,
    check_file,
    find_project_root,
)

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


def test_the_executor_is_allowed_to_spawn_a_process() -> None:
    """ADR-025. The one package whose job is shelling out to a coding agent.

    Asserted the same way the gateway's exemption is: by pointing at the real
    file that uses the banned import, so the exemption cannot rot into a rule
    that exempts a package no longer doing the thing.
    """
    workspace = PROJECT_ROOT / "src" / "dynaflows" / "executor" / "workspace.py"
    assert "import subprocess" in workspace.read_text(encoding="utf-8")
    assert check_file(workspace, PROJECT_ROOT) == []


@pytest.mark.parametrize("module", sorted(EXECUTOR_ONLY_MODULES))
def test_a_worker_may_not_spawn_a_process(module: str, tmp_path: Path) -> None:
    """The hole this rule was written to close.

    The filesystem rule forbids a worker calling `Path.mkdir` and said nothing
    about the same worker calling `subprocess.run`, which is strictly more
    capable. An AST walk cannot catch it by call name -- `run` is a method
    name this repo uses for ordinary things, and banning it would fire on
    correct code, which is how a gate gets disabled (playbook 5.2, Pattern 5).
    So the import is what is banned.
    """
    offender = tmp_path / "src" / "dynaflows" / "graph" / "nodes.py"
    offender.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    offender.write_text(f"import {module}\n", encoding="utf-8")

    violations = check_file(offender, tmp_path)
    assert len(violations) == 1
    assert "ADR-016" in violations[0]
    assert module in violations[0]


def test_the_executor_may_touch_the_filesystem() -> None:
    """It cuts a worktree; mkdir is not incidental to that, it is the job.

    ADR-016 is unchanged by the exemption: the worktree lives under
    .dynaflows/work/, the same boundary every other exempt package respects.
    """
    workspace = PROJECT_ROOT / "src" / "dynaflows" / "executor" / "workspace.py"
    assert "mkdir(" in workspace.read_text(encoding="utf-8")
    assert check_file(workspace, PROJECT_ROOT) == []


def test_the_editor_module_may_spawn_and_is_the_only_one_outside_the_executor() -> None:
    """The exemption is one file, and it is the file that needs it.

    Asserted against the real source so the exemption cannot outlive its
    reason: if the editor launch ever moves back into `cli.py`, this fails
    rather than leaving a name in an allowlist that no longer earns it.
    """
    assert SPAWN_MODULES == {"editing.py"}
    editing = PROJECT_ROOT / "src" / "dynaflows" / "editing.py"
    assert "import subprocess" in editing.read_text(encoding="utf-8")
    assert check_file(editing, PROJECT_ROOT) == []


def test_the_cli_itself_may_not_spawn(tmp_path: Path) -> None:
    """Why `editing.py` exists at all.

    `cli.py` needed to launch one editor. Exempting it would have granted a
    thousand-line module -- about to grow a `change` command -- the right to
    spawn anything. The rule is only worth having if its exceptions stay the
    size of the need, so the need moved and `cli.py` stayed bound.
    """
    offender = tmp_path / "src" / "dynaflows" / "cli.py"
    offender.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    offender.write_text("import subprocess\n", encoding="utf-8")

    violations = check_file(offender, tmp_path)
    assert len(violations) == 1
    assert "ADR-016" in violations[0]
