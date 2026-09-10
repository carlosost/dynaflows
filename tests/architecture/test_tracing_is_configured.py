"""Every command that runs the graph configures tracing first. ADR-011.

This is the rule, not the instance. `run` and `resume` were both missing the
call, and the failure mode is the one ADR-011 exists to prevent: LangChain and
LangGraph read os.environ directly, `get_settings()` is deliberately pure and
exports nothing, so a run emits NO traces while `doctor` keeps reporting
LangSmith as OK. Nothing is broken, nothing is logged, and the project's
central claim quietly stops being true.

An AST check rather than a convention, so the next command cannot forget.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.deterministic

CLI = Path(__file__).resolve().parents[2] / "src" / "dynaflows" / "cli.py"


def called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def graph_commands() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """CLI commands that build a graph, counting their nested helpers.

    Scoped to `@app.command()` functions at module level. An earlier version
    walked every function and flagged the `_go` closures INSIDE run/resume --
    which do build a graph, but whose enclosing command already configures
    tracing. A check that fires on correct code gets disabled (§5.2 Pattern 5).
    """
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    commands: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any("app.command" in ast.unparse(d) for d in node.decorator_list):
            continue
        if "build_graph" in called_names(node):
            commands[node.name] = node
    return commands


def test_the_check_has_a_subject() -> None:
    """A rule with nothing to check is untested by construction (AP-11)."""
    assert set(graph_commands()) >= {"run", "resume"}


@pytest.mark.parametrize("name", sorted(graph_commands()))
def test_a_command_that_builds_a_graph_also_configures_tracing(name: str) -> None:
    assert "configure_tracing" in called_names(graph_commands()[name]), (
        f"`{name}` builds a graph without calling configure_tracing(); its runs "
        "would emit no traces while doctor still reports LangSmith as OK"
    )
