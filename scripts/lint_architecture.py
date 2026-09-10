#!/usr/bin/env python3
"""Architectural lint. Enforces ADRs that tests cannot.

Playbook AP-19: every enforcement mechanism in the playbook reads *code*. An
ADR is the one artifact that can assert anything and never be contradicted.
This script is how ADR-010 stops being prose.

Rule 1 (ADR-010) -- provider SDKs and HTTP clients are imported only inside
src/dynaflows/gateway/.

Deliberately NOT implemented yet: the interrupt-node purity rule (ADR-007).
Its subject -- node functions calling interrupt() -- does not exist until
Phase 1 step 1.4. A rule with no possible subject is untested by construction
(AP-11); it lands in the same session as the first interrupt node.

Usage:  lint_architecture.py [PATH ...]      (default: the whole src tree)
Exit:   0 clean, 1 violations, 2 bad invocation.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

GATEWAY_ONLY_MODULES = frozenset({"httpx", "openai", "langchain_openai", "requests", "aiohttp"})
GATEWAY_PACKAGE = Path("src") / "dynaflows" / "gateway"


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _imports(tree: ast.AST) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield _root_module(alias.name), node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield _root_module(node.module), node.lineno


def check_file(path: Path, project_root: Path) -> list[str]:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = path
    if GATEWAY_PACKAGE in relative.parents:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error, cannot lint: {exc.msg}"]

    return [
        f"{relative}:{lineno}: ADR-010 violation -- '{module}' may only be imported "
        f"inside {GATEWAY_PACKAGE}/"
        for module, lineno in _imports(tree)
        if module in GATEWAY_ONLY_MODULES
    ]


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SystemExit("pyproject.toml not found; run inside the project")


def main(argv: list[str]) -> int:
    project_root = find_project_root(Path(__file__).parent)
    targets = [Path(a) for a in argv] if argv else sorted((project_root / "src").rglob("*.py"))

    violations: list[str] = []
    checked = 0
    for target in targets:
        if target.suffix != ".py" or not target.is_file():
            continue
        checked += 1
        violations.extend(check_file(target, project_root))

    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        print(f"\n{len(violations)} violation(s) in {checked} file(s).", file=sys.stderr)
        return 1
    print(f"lint_architecture: {checked} file(s) clean (ADR-010).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
