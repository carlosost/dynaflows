#!/usr/bin/env python3
"""Architectural lint. Enforces ADRs that tests cannot.

Playbook AP-19: every enforcement mechanism in the playbook reads *code*. An
ADR is the one artifact that can assert anything and never be contradicted.
This script is how ADR-010 stops being prose.

Rule 1 (ADR-010) -- provider SDKs and HTTP clients are imported only inside
src/dynaflows/gateway/.

Rule 2 (ADR-007) -- a function that calls interrupt() performs no I/O.

Rule 3 (ADR-016, ADR-017) -- the filesystem is touched only inside
src/dynaflows/store/, plus the two packages that own a database.

ADR-016 says a worker never writes outside .dynaflows/, and ADR-017 says the
files a plan names are read in one place rather than in N parallel branches.
Both are claims about code, and an ADR that asserts testable behaviour names
the test that proves it (AP-19 habit 1). This is that test, and it is the
cheap half: "no module outside store/ calls write_text" is a property a parser
can check, whereas "every worker stays inside its bounds" would not be.

LangGraph resumes by re-running the interrupted node FROM ITS FIRST LINE. Work
done before the interrupt is therefore repeated on every resume: a paid call
is paid again, and worse, the human can be shown different text than the text
they were approving. The rule is mechanical -- no `await`, no I/O helper -- so
it is checked rather than remembered.

Added in step 1.4, the session that introduced the first interrupt node. Before
that it had no possible subject, and a rule with no subject is untested by
construction (AP-11).

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

# ADR-016/ADR-017. `store` owns run artifacts and input resolution; `playbook`
# and `gateway` own their SQLite files and the corpus they index; `settings`
# resolves the project root and reads .env. Every one of these is a deliberate,
# reviewed exception, which is why they are listed by name rather than by a
# pattern that would quietly admit the next module too.
FILESYSTEM_PACKAGES = (
    Path("src") / "dynaflows" / "store",
    Path("src") / "dynaflows" / "playbook",
    Path("src") / "dynaflows" / "gateway",
)
FILESYSTEM_MODULES = frozenset({"settings.py", "cli.py", "doctor.py", "checkpoint.py"})
# Write calls first: these are what ADR-016 forbids. Reads are listed too,
# because ADR-017's whole point is that reading happens in ONE place.
#
# NOT listed, deliberately: `replace` and `copy`. `Path.replace` renames a file,
# but `dataclasses.replace` and `dict.copy` are ordinary code and appear all
# over this repo. An AST walk sees only the attribute name, so including them
# made the linter fire four times on correct code the first time it ran -- and
# a gate that fires on ordinary work is a gate people disable (playbook 5.2,
# Pattern 5). `rename` and `unlink` cover the same ground unambiguously. This
# is a real hole in the rule and it is named here rather than left to be
# discovered.
FILESYSTEM_CALLS = frozenset(
    {
        "write_text", "write_bytes", "mkdir", "touch", "unlink", "rmdir", "rename",
        "chmod", "symlink_to", "rmtree", "remove", "makedirs",
        "copytree", "read_text", "read_bytes", "iterdir", "rglob", "glob", "walk",
    }
)

# Names that mean "this function touches the world". Not exhaustive by
# construction -- the `await` ban below is the load-bearing half, since every
# I/O path in this codebase is async.
IO_CALL_NAMES = frozenset(
    {
        "open",
        "connect",
        "connect_cache",
        "get_gateway",
        "gateway_from",
        "get_settings",
        "get_model_registry",
        "get_playbook_repository",
        "get_response_cache",
        "read_text",
        "write_text",
        "catalogue",
        "invoke",
        "ainvoke",
    }
)


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _imports(tree: ast.AST) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield _root_module(alias.name), node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield _root_module(node.module), node.lineno


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _interrupt_node_violations(tree: ast.AST, relative: Path) -> list[str]:
    """ADR-007: a gate node reads state, asks, and writes the answer. Nothing else."""
    problems: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = [c for c in ast.walk(func) if isinstance(c, ast.Call)]
        if not any(_call_name(c) == "interrupt" for c in calls):
            continue
        for await_node in (n for n in ast.walk(func) if isinstance(n, ast.Await)):
            problems.append(
                f"{relative}:{await_node.lineno}: ADR-007 violation -- "
                f"'{func.name}' calls interrupt() and awaits; a resumed graph re-runs "
                "the node from line one, so this work is repeated on every resume"
            )
        for call in calls:
            name = _call_name(call)
            if name in IO_CALL_NAMES:
                problems.append(
                    f"{relative}:{call.lineno}: ADR-007 violation -- "
                    f"'{func.name}' calls interrupt() and '{name}()'; an interrupt node "
                    "must stay pure so re-running it on resume costs nothing"
                )
    return problems


def _filesystem_violations(tree: ast.AST, relative: Path) -> list[str]:
    """ADR-016 and ADR-017, enforced instead of asserted."""
    if any(package in relative.parents for package in FILESYSTEM_PACKAGES):
        return []
    if relative.name in FILESYSTEM_MODULES:
        return []
    problems = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        name = _call_name(call)
        if name in FILESYSTEM_CALLS:
            problems.append(
                f"{relative}:{call.lineno}: ADR-016/ADR-017 violation -- "
                f"'{name}()' touches the filesystem outside "
                f"{FILESYSTEM_PACKAGES[0]}/; workers read and reason, they do not "
                "open files, and inputs are resolved once at dispatch"
            )
    return problems


def check_file(path: Path, project_root: Path) -> list[str]:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error, cannot lint: {exc.msg}"]

    # ADR-010 exempts the gateway package -- it is the one legal home for a
    # provider SDK. ADR-007 exempts nobody: an interrupt node is impure
    # wherever it lives.
    violations: list[str] = []
    if GATEWAY_PACKAGE not in relative.parents:
        violations += [
            f"{relative}:{lineno}: ADR-010 violation -- '{module}' may only be imported "
            f"inside {GATEWAY_PACKAGE}/"
            for module, lineno in _imports(tree)
            if module in GATEWAY_ONLY_MODULES
        ]
    return (
        violations
        + _interrupt_node_violations(tree, relative)
        + _filesystem_violations(tree, relative)
    )


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
    print(f"lint_architecture: {checked} file(s) clean (ADR-007, ADR-010, ADR-016/017).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
