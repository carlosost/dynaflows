"""The worktree lifecycle, exercised against real git.

Real git rather than a mock, because every property this module claims is a
property of git's behaviour and not of our call sequence: that a worktree
branches from a commit and therefore misses uncommitted work, that a new file
is invisible to `git diff` until staged, that removing a dirty worktree needs
`--force`. A mock would assert we called the commands we already know we call
and would have caught none of those.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dynaflows.contracts.errors import ErrorCode
from dynaflows.executor import workspace as ws


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit, which is the minimum a worktree needs."""
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "init", "--quiet", "--initial-branch=main")
    _run(root, "config", "user.email", "t@example.invalid")
    _run(root, "config", "user.name", "t")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run(root, "add", "-A")
    _run(root, "commit", "--quiet", "-m", "first")
    return root


@pytest.fixture
def home(repo: Path) -> Path:
    return repo / ".dynaflows"


def test_create_cuts_a_worktree_at_head(repo: Path, home: Path) -> None:
    space = ws.create(repo, home, "r1")
    assert space.path == home / "work" / "r1"
    assert (space.path / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert space.base == ws.git(repo, "rev-parse", "HEAD")
    assert space.branch == "dynaflows/r1"


def test_the_users_working_tree_is_untouched_by_edits_in_the_workspace(
    repo: Path, home: Path
) -> None:
    space = ws.create(repo, home, "r1")
    (space.path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_uncommitted_work_is_reported_because_the_agent_cannot_see_it(
    repo: Path, home: Path
) -> None:
    """The surprise the module exists to make loud.

    The user edits a file, does not commit, and asks for a change to it. The
    worktree branches from HEAD, so the agent reads the committed version.
    The run must be able to say so at the gate.
    """
    (repo / "module.py").write_text("VALUE = 99\n", encoding="utf-8")
    space = ws.create(repo, home, "r1")

    assert space.dirty_paths == ["module.py"]
    assert space.dirty_checked is True
    assert (space.path / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_a_clean_tree_reports_no_dirty_paths_and_says_it_checked(
    repo: Path, home: Path
) -> None:
    """AP-20: 'nothing was dirty' and 'dirtiness unknown' are two facts."""
    space = ws.create(repo, home, "r1")
    assert space.dirty_paths == []
    assert space.dirty_checked is True


def test_an_untracked_file_is_not_a_dirty_path(repo: Path, home: Path) -> None:
    (repo / "scratch.txt").write_text("notes\n", encoding="utf-8")
    space = ws.create(repo, home, "r1")
    assert space.dirty_paths == []


def test_capture_sees_a_new_file_that_an_unstaged_diff_would_miss(
    repo: Path, home: Path
) -> None:
    """The reason `capture` stages rather than leaving it to the caller."""
    space = ws.create(repo, home, "r1")
    (space.path / "added.py").write_text("NEW = True\n", encoding="utf-8")

    unstaged = ws.git(space.path, "diff", "--patch", space.base)
    assert "added.py" not in unstaged

    changes = space.capture()
    assert changes.files == ["added.py"]
    assert "NEW = True" in changes.patch
    assert changes.is_empty is False


def test_capture_reports_a_modification_and_a_deletion(repo: Path, home: Path) -> None:
    (repo / "gone.py").write_text("X = 1\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "--quiet", "-m", "second")

    space = ws.create(repo, home, "r1")
    (space.path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (space.path / "gone.py").unlink()

    changes = space.capture()
    assert changes.files == ["gone.py", "module.py"]
    assert "VALUE = 2" in changes.patch


def test_an_agent_that_changed_nothing_is_empty_not_failed(repo: Path, home: Path) -> None:
    space = ws.create(repo, home, "r1")
    changes = space.capture()
    assert changes.is_empty is True
    assert changes.files == []
    assert changes.patch == ""


def test_the_diff_is_against_the_base_commit_even_after_head_moves(
    repo: Path, home: Path
) -> None:
    """A run paused at G2 while the user commits on main.

    `base` is a resolved sha, so the diff still describes only what the agent
    did -- not the user's commit as well.
    """
    space = ws.create(repo, home, "r1")
    (space.path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    (repo / "unrelated.py").write_text("OTHER = 1\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "--quiet", "-m", "user commits while paused")

    changes = space.capture()
    assert changes.files == ["module.py"]


def test_create_refuses_a_second_attempt_in_an_existing_workspace(
    repo: Path, home: Path
) -> None:
    ws.create(repo, home, "r1")
    with pytest.raises(ws.WorkspaceError) as caught:
        ws.create(repo, home, "r1")
    assert caught.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_create_refuses_a_directory_that_is_not_a_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ws.WorkspaceError) as caught:
        ws.create(plain, plain / ".dynaflows", "r1")
    assert caught.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_adopt_reattaches_with_the_checkpointed_base(repo: Path, home: Path) -> None:
    created = ws.create(repo, home, "r1")
    (created.path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    resumed = ws.adopt(repo, home, "r1", created.base)
    assert resumed.path == created.path
    assert resumed.base == created.base
    assert resumed.capture().files == ["module.py"]


def test_adopt_does_not_claim_to_know_whether_the_tree_was_dirty(
    repo: Path, home: Path
) -> None:
    """The dirty list described the moment of creation. Re-reporting it as
    current would answer a question about a different moment."""
    created = ws.create(repo, home, "r1")
    resumed = ws.adopt(repo, home, "r1", created.base)
    assert resumed.dirty_paths == []
    assert resumed.dirty_checked is False


def test_adopt_fails_loudly_when_the_workspace_is_gone(repo: Path, home: Path) -> None:
    with pytest.raises(ws.WorkspaceError) as caught:
        ws.adopt(repo, home, "r1", "deadbeef")
    assert caught.value.envelope.code is ErrorCode.CONFIG_INVALID


def test_discard_removes_a_dirty_worktree_and_its_branch(repo: Path, home: Path) -> None:
    space = ws.create(repo, home, "r1")
    (space.path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    ws.discard(space)

    assert not space.path.exists()
    assert "dynaflows/r1" not in ws.git(repo, "branch", "--list")
    assert "r1" not in ws.git(repo, "worktree", "list")


def test_discard_is_safe_to_call_twice(repo: Path, home: Path) -> None:
    """Cleanup runs on the failure path, where it may already have run."""
    space = ws.create(repo, home, "r1")
    ws.discard(space)
    ws.discard(space)


def test_git_returns_the_answer_when_check_is_off_and_the_command_fails(
    repo: Path,
) -> None:
    assert ws.git(repo, "rev-parse", "--verify", "no-such-branch", check=False) == ""
