"""ADR-007's proof. The ADR has named this file since it was written.

LangGraph resumes by re-running the interrupted node FROM ITS FIRST LINE. Work
done before the interrupt is repeated on every resume: a paid call is paid
again, and worse, the human can be shown different text than the text they
were approving.

Both halves matter, as with the gateway boundary: the real tree is clean, AND
the linter actually catches a violation. A rule that has never failed is a
rule nobody has tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from lint_architecture import check_file, find_project_root  # noqa: E402

pytestmark = pytest.mark.deterministic

PROJECT_ROOT = find_project_root(Path(__file__).parent)


def write(tmp_path: Path, body: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    target = tmp_path / "src" / "dynaflows" / "graph" / "nodes.py"
    target.parent.mkdir(parents=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_no_interrupt_node_in_the_real_tree_does_io() -> None:
    offenders = [
        v
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        for v in check_file(path, PROJECT_ROOT)
        if "ADR-007" in v
    ]
    assert offenders == [], "\n".join(offenders)


def test_the_real_gate_node_is_covered_by_the_rule() -> None:
    """The rule has a subject. Without one it would pass vacuously (AP-11)."""
    source = (PROJECT_ROOT / "src" / "dynaflows" / "graph" / "nodes.py").read_text()
    assert "interrupt(" in source


def test_an_await_inside_an_interrupt_node_is_caught(tmp_path: Path) -> None:
    """The load-bearing half: every I/O path in this codebase is async."""
    target = write(
        tmp_path,
        "async def approve(state, config=None):\n"
        "    data = await fetch()\n"
        "    return interrupt(data)\n",
    )
    violations = [v for v in check_file(target, tmp_path) if "ADR-007" in v]
    assert len(violations) == 1
    assert "awaits" in violations[0]


def test_a_synchronous_io_call_inside_an_interrupt_node_is_caught(tmp_path: Path) -> None:
    target = write(
        tmp_path,
        "def approve(state, config=None):\n"
        "    settings = get_settings()\n"
        "    return interrupt(settings)\n",
    )
    violations = [v for v in check_file(target, tmp_path) if "ADR-007" in v]
    assert len(violations) == 1
    assert "get_settings" in violations[0]


def test_a_method_style_io_call_is_caught_too(tmp_path: Path) -> None:
    """`self.cache.connect()` is as much I/O as `connect()`."""
    target = write(
        tmp_path,
        "def approve(state, config=None):\n"
        "    handle = something.connect()\n"
        "    return interrupt(handle)\n",
    )
    assert [v for v in check_file(target, tmp_path) if "ADR-007" in v]


def test_a_pure_gate_node_passes(tmp_path: Path) -> None:
    target = write(
        tmp_path,
        "async def approve(state, config=None):\n"
        "    answer = interrupt({'show': state['x']})\n"
        "    return {'gate': answer}\n",
    )
    assert [v for v in check_file(target, tmp_path) if "ADR-007" in v] == []


def test_a_node_that_awaits_but_never_interrupts_is_left_alone(tmp_path: Path) -> None:
    """Every other node in the graph awaits. A rule that fired on them would
    be disabled within a day (§5.2 Pattern 5)."""
    target = write(
        tmp_path,
        "async def enhance(state, config=None):\n"
        "    result = await gateway.call(request)\n"
        "    return {'enhanced': result}\n",
    )
    assert [v for v in check_file(target, tmp_path) if "ADR-007" in v] == []


def test_the_rule_applies_inside_the_gateway_package_too(tmp_path: Path) -> None:
    """ADR-010 exempts the gateway; ADR-007 exempts nobody -- an interrupt
    node is impure wherever it lives."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    target = tmp_path / "src" / "dynaflows" / "gateway" / "odd.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "async def gate(state):\n    x = await thing()\n    return interrupt(x)\n",
        encoding="utf-8",
    )
    assert [v for v in check_file(target, tmp_path) if "ADR-007" in v]
