"""ADR-026: the one place dynaflows writes outside `.dynaflows/`.

Against a real repo and real `git apply`, because every property claimed here
is git's: that `--check` leaves the tree untouched, that an apply without
`--index` leaves the change unstaged, that a patch against a moved file fails
rather than half-applying.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from dynaflows.contracts.state import GateDecision, GateOutcome
from dynaflows.executor.apply import apply_to_working_tree

pytestmark = pytest.mark.deterministic

APPROVED = GateOutcome(decision=GateDecision.APPROVE)
REJECTED = GateOutcome(decision=GateDecision.REJECT)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "first")
    return root


@pytest.fixture
def patch(repo: Path) -> str:
    """A real diff, produced by git rather than hand-written."""
    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    text = _git(repo, "diff")
    _git(repo, "checkout", "--", "module.py")
    return text


def test_an_approved_patch_lands_in_the_working_tree(repo: Path, patch: str) -> None:
    result = apply_to_working_tree(repo, patch, APPROVED)

    assert result.ok is True
    assert result.files == ("module.py",)
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_the_change_is_left_unstaged_so_an_editor_shows_it_as_hunks(
    repo: Path, patch: str
) -> None:
    """The whole requirement. Staged changes do not offer hunk-level accept
    and reject in an editor's source-control panel; unstaged ones do."""
    apply_to_working_tree(repo, patch, APPROVED)

    assert _git(repo, "diff", "--name-only").strip() == "module.py"
    assert _git(repo, "diff", "--cached", "--name-only").strip() == ""


def test_a_rejected_gate_writes_nothing(repo: Path, patch: str) -> None:
    result = apply_to_working_tree(repo, patch, REJECTED)

    assert result.ok is False
    assert result.refused is True
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_no_gate_at_all_writes_nothing(repo: Path, patch: str) -> None:
    """Enforced rather than documented: 'call this only after G3' is the kind
    of rule that survives until the second caller."""
    result = apply_to_working_tree(repo, patch, None)

    assert result.refused is True
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_an_empty_patch_is_refused_rather_than_reported_as_applied(repo: Path) -> None:
    result = apply_to_working_tree(repo, "   \n", APPROVED)
    assert result.refused is True


def test_a_stale_patch_leaves_the_tree_exactly_as_it_was(repo: Path, patch: str) -> None:
    """The user committed something else while the run was paused at G3.

    `--check` runs first for this case: a partial application would put a
    half-applied change in the user's own tree, which is the failure the whole
    pipeline exists to prevent, arriving at the last step.
    """
    (repo / "module.py").write_text("VALUE = 99\nEXTRA = True\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "user moved on")

    result = apply_to_working_tree(repo, patch, APPROVED)

    assert result.ok is False
    assert result.refused is False, "this was attempted and failed, not declined"
    assert "no longer applies" in result.reason
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 99\nEXTRA = True\n"
    assert _git(repo, "status", "--porcelain").strip() == "", "the tree must be untouched"


def test_refused_and_failed_are_distinguishable(repo: Path, patch: str) -> None:
    """A conflicted patch and an unapproved one need different advice, and the
    difference is invisible in `ok` alone (AP-20)."""
    assert apply_to_working_tree(repo, patch, REJECTED).refused is True

    (repo / "module.py").write_text("SOMETHING = 'else'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "moved")
    conflicted = apply_to_working_tree(repo, patch, APPROVED)
    assert conflicted.ok is False
    assert conflicted.refused is False


def test_a_new_file_in_the_patch_is_created(repo: Path) -> None:
    (repo / "added.py").write_text("NEW = True\n", encoding="utf-8")
    _git(repo, "add", "-N", "added.py")
    text = _git(repo, "diff")
    (repo / "added.py").unlink()
    _git(repo, "reset", "--quiet")

    result = apply_to_working_tree(repo, text, APPROVED)

    assert result.ok is True
    assert (repo / "added.py").read_text(encoding="utf-8") == "NEW = True\n"


def test_git_apply_lives_in_exactly_one_module() -> None:
    """The guard that replaces the linter rule I could not write.

    ADR-016 is enforced by inspecting Python call sites, and `git apply` is a
    subprocess -- invisible to that. This is weaker and ADR-026 says so: it
    does not make an unauthorised write impossible, it makes it impossible to
    add one without this test failing.

    Matched as `"apply",` -- a quoted argument followed by a comma, which is
    what a git subcommand looks like in an argv list. NOT a bare `apply`
    anywhere in the text: the first version of this test searched for the
    quoted word alone and would have fired on any docstring containing it,
    which is a matcher that sees too much. Its sibling mistake, a matcher that
    sees too little, has cost this project three separate bugs in one day.

    The root is found by walking up for pyproject.toml rather than counting
    `parents[n]`, because a count is a fact about this file's location and
    breaks silently when the file moves.
    """
    here = Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "pyproject.toml").is_file()) / "src"
    pattern = re.compile(r'["\']apply["\']\s*,')
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == ["dynaflows/executor/apply.py"], (
        "a git apply outside executor/apply.py writes to the user's tree "
        "without passing G3; see ADR-026"
    )


def test_an_already_applied_patch_is_reported_as_such_not_as_a_conflict(
    repo: Path, patch: str
) -> None:
    """git says "patch does not apply" for two opposite situations.

    Live, 2026-09-22: a user was told their work was stranded on a branch
    while all eight files already carried it. The forward --check fails
    identically in both cases; a REVERSE check separates them, because a
    patch that applies backwards is a patch already applied forwards.
    """
    first = apply_to_working_tree(repo, patch, APPROVED)
    assert first.ok is True
    assert first.already is False

    second = apply_to_working_tree(repo, patch, APPROVED)
    assert second.ok is True
    assert second.already is True
    assert second.files == ("module.py",)
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_a_genuine_conflict_is_still_a_conflict_not_already_applied(
    repo: Path, patch: str
) -> None:
    """The distinction only helps if it still calls a conflict a conflict."""
    (repo / "module.py").write_text("SOMETHING = 'else'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "moved")

    result = apply_to_working_tree(repo, patch, APPROVED)

    assert result.ok is False
    assert result.already is False
    assert "no longer applies" in result.reason
