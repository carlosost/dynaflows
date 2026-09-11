"""ADR-016 and ADR-017, proven rather than asserted.

ADR-016 says a worker never writes outside `.dynaflows/`. ADR-017 says the
files a plan names are read in one place. Both are claims about code, and the
ADRs name this file as the thing that makes them true (AP-19 habit 1).

The important half is not that the linter passes on the current tree -- a
linter that checks nothing also passes. It is that it FAILS on a violation,
which is what the first two tests here drive.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.deterministic

ROOT = Path(__file__).resolve().parents[2]
LINTER = ROOT / "scripts" / "lint_architecture.py"


def _lint(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LINTER), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_shipped_tree_has_no_filesystem_call_outside_the_store(tmp_path: Path) -> None:
    result = _lint(ROOT / "src")
    assert result.returncode == 0, result.stderr


def test_a_worker_that_writes_a_file_is_rejected(tmp_path: Path) -> None:
    """The violation ADR-016 exists to prevent, written out."""
    offender = tmp_path / "nodes.py"
    offender.write_text(
        "from pathlib import Path\n"
        "async def worker(state, config=None):\n"
        "    Path('/tmp/report.md').write_text('findings')\n"
        "    return {}\n",
        encoding="utf-8",
    )

    result = _lint(offender)

    assert result.returncode == 1
    assert "ADR-016/ADR-017 violation" in result.stderr
    assert "write_text" in result.stderr


def test_a_worker_that_reads_a_file_itself_is_rejected(tmp_path: Path) -> None:
    """ADR-017's half. Reading is not forbidden -- reading HERE is. A worker
    that opens the files it was told about looks reasonable and is exactly the
    design the ADR rejected, so the linter has to catch it too."""
    offender = tmp_path / "nodes.py"
    offender.write_text(
        "from pathlib import Path\n"
        "async def worker(state, config=None):\n"
        "    body = Path(state['task'].inputs[0]).read_text()\n"
        "    return {'results': [body]}\n",
        encoding="utf-8",
    )

    result = _lint(offender)

    assert result.returncode == 1
    assert "read_text" in result.stderr


def test_the_store_package_itself_is_allowed_to_write() -> None:
    """The exemption has to work, or the rule is unimplementable and someone
    will delete it rather than the code that violates it."""
    result = _lint(ROOT / "src" / "dynaflows" / "store" / "run_store.py")

    assert result.returncode == 0, result.stderr


def test_dataclasses_replace_is_not_mistaken_for_a_file_rename() -> None:
    """Regression: the first version of this rule listed `replace`, fired four
    times on correct code, and would have been switched off within a day
    (playbook 5.2, Pattern 5). A false positive in a gate is not a small bug."""
    offender = ROOT / "src" / "dynaflows" / "contracts" / "calls.py"

    result = _lint(offender)

    assert result.returncode == 0, result.stderr
