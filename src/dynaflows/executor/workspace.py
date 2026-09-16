"""An isolated checkout the coding agent may write to. ADR-025.

The write pipeline hands an approved brief to a coding agent that edits files.
The question this module answers is *which* files, and the answer is not "the
ones the user is looking at".

A git worktree gives four properties that matter here, and nothing else gives
all four:

  - a half-applied change never touches the user's working tree, so an agent
    that dies mid-edit leaves no mess to clean up by hand;
  - the diff is `git diff <base>` inside the worktree, with no bookkeeping of
    which files were touched -- git already knows;
  - abandoning is `git worktree remove`, which is one command and leaves no
    orphan branch if the branch is deleted with it;
  - it lives under `.dynaflows/`, so ADR-016's boundary survives literally:
    dynaflows still writes only inside its own directory, and the agent it
    invokes writes only inside the worktree.

**The surprise this module exists to make loud.** A worktree branches from a
COMMIT. Uncommitted work in the user's tree is not in it. "Fix the bug in the
file I am editing" would therefore operate on the last committed version of
that file and produce a diff that silently ignores the edit in front of the
user. That is discovered afterwards, if at all, so it is measured up front
and carried on the workspace for the gate to show. The gate can then say
which files differ, and the user decides -- rather than the run deciding for
them by not mentioning it.

The lifecycle is explicit `create` / `adopt` / `discard` rather than a context
manager. A run is interruptible at G2 and resumable in a later process, so the
worktree has to outlive the Python object that made it; a `with` block would
promise cleanup that resume makes wrong.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dynaflows.contracts.errors import DynaflowsError, ErrorCode

__all__ = ["Changes", "Workspace", "WorkspaceError", "adopt", "create", "discard", "git"]

# Both derived from the run id, so a worktree and its branch can never be
# paired with the wrong run, and an orphan of either is traceable to the run
# that leaked it.
WORK_DIR_NAME = "work"
BRANCH_PREFIX = "dynaflows"


class WorkspaceError(DynaflowsError):
    """A git operation the run cannot continue without."""


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run one git command and return its stdout, stripped.

    `check` is a real parameter rather than a flag to be avoided (AP-10):
    several callers ask git a QUESTION whose "no" is a non-zero exit --
    `rev-parse` on a branch that does not exist, `diff --quiet` on a clean
    tree -- and for those a raise would be the wrong answer to a legitimate
    question.
    """
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise WorkspaceError.of(
            ErrorCode.UNKNOWN,
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}",
        )
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class Changes:
    """What the agent did to the worktree, read at one moment.

    `files` is carried alongside the patch rather than parsed back out of it
    so that "the agent changed nothing" is an empty list and not a string
    comparison against an empty patch -- a distinction that matters because a
    run producing no diff is a legitimate outcome (the agent read the code and
    concluded no change was needed) and must be reportable as such rather than
    as a failure.
    """

    patch: str
    stat: str
    files: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.files


@dataclass(frozen=True, slots=True)
class Workspace:
    """Where the agent works, and what it was branched from.

    `base` is a resolved commit sha rather than a ref name on purpose. The
    user may commit on their branch while a run is paused at G2, and a diff
    against a moving ref would then include commits the agent never made.
    """

    run_id: str
    repo_root: Path
    path: Path
    branch: str
    base: str
    # Files that were modified in the user's tree but not committed when the
    # workspace was cut, and are therefore NOT visible to the agent. Empty is
    # the common case and means what it says; see AP-20 -- "no dirty files"
    # and "dirtiness not checked" must not render identically, so the gate
    # reads `dirty_checked` rather than inferring from an empty list.
    dirty_paths: list[str] = field(default_factory=list)
    dirty_checked: bool = True

    def capture(self) -> Changes:
        """Stage everything and read the diff, in that order, once.

        One method rather than `stage_all()` + `diff()` + `is_empty()`,
        because three methods whose answers are only correct in a particular
        order is a footgun the caller has to remember not to trip: a `diff()`
        called before staging reports an empty change set for an agent that
        created twenty new files, and reports it without any error. Staging
        is what makes a new file visible to `git diff` at all, so the two
        steps are not independent and are not offered as if they were.

        Both patch and stat come from the same staged state, so they cannot
        describe different moments.
        """
        git(self.path, "add", "-A")
        return Changes(
            patch=git(self.path, "diff", "--cached", "--patch", self.base),
            stat=git(self.path, "diff", "--cached", "--stat", self.base),
            files=[
                line
                for line in git(
                    self.path, "diff", "--cached", "--name-only", self.base
                ).splitlines()
                if line
            ],
        )


def _worktree_path(home: Path, run_id: str) -> Path:
    return home / WORK_DIR_NAME / run_id


def _dirty_paths(repo_root: Path) -> list[str]:
    """Tracked files with uncommitted modifications, staged or not.

    Untracked files are excluded: an untracked file is not in the base commit
    and not in the worktree either, but it is also not a *divergence* the user
    would be surprised by -- a new scratch file is not work the agent was
    expected to see. Modified tracked files are the surprising case.

    Read with `diff --name-only` rather than `status --porcelain`. The
    porcelain format puts the path at a fixed column offset behind two status
    characters, and `git()` strips its output -- which removes the leading
    space of an unstaged modification and shifts every path on the first line
    by one character. The first run of this module's tests returned
    `odule.py`. A format that needs column arithmetic to read will be misread
    eventually; this one returns the paths and nothing else.
    """
    listed = git(repo_root, "diff", "--name-only", "HEAD")
    return sorted(line for line in listed.splitlines() if line)


def create(repo_root: Path, home: Path, run_id: str) -> Workspace:
    """Cut a fresh worktree for this run at the current HEAD.

    Fails rather than reusing if one already exists for the run id. Reuse
    would silently carry a previous attempt's edits into a new one, and the
    resulting diff would attribute them to this run.
    """
    inside = git(repo_root, "rev-parse", "--is-inside-work-tree", check=False)
    if inside != "true":
        raise WorkspaceError.of(
            ErrorCode.CONFIG_INVALID,
            f"{repo_root} is not a git repository; the write pipeline needs one "
            "because the worktree is how a half-applied edit stays out of your tree",
        )

    path = _worktree_path(home, run_id)
    if path.exists():
        raise WorkspaceError.of(
            ErrorCode.CONFIG_INVALID,
            f"a workspace for run {run_id} already exists at {path}; "
            "resume the run or discard it, do not start a second attempt in it",
        )

    base = git(repo_root, "rev-parse", "HEAD")
    branch = f"{BRANCH_PREFIX}/{run_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo_root, "worktree", "add", "--quiet", "-b", branch, str(path), base)

    return Workspace(
        run_id=run_id,
        repo_root=repo_root,
        path=path,
        branch=branch,
        base=base,
        dirty_paths=_dirty_paths(repo_root),
    )


def adopt(repo_root: Path, home: Path, run_id: str, base: str) -> Workspace:
    """Re-attach to the worktree of a run that is being resumed.

    The base sha comes from the caller (it was checkpointed) rather than being
    re-read from HEAD, because HEAD may have moved while the run was paused
    and the diff must stay against the commit the agent actually started from.

    `dirty_checked=False`: the dirty-file list was measured when the workspace
    was created and is not re-measured here, because the answer would be about
    a different moment. A stale list presented as current is worse than an
    absent one.
    """
    path = _worktree_path(home, run_id)
    if not path.is_dir():
        raise WorkspaceError.of(
            ErrorCode.CONFIG_INVALID,
            f"no workspace at {path} for run {run_id}; it was discarded or never created",
        )
    return Workspace(
        run_id=run_id,
        repo_root=repo_root,
        path=path,
        branch=f"{BRANCH_PREFIX}/{run_id}",
        base=base,
        dirty_paths=[],
        dirty_checked=False,
    )


def discard(workspace: Workspace) -> None:
    """Remove the worktree and its branch.

    `--force` because the worktree is expected to be dirty -- that is the
    point of it -- and a refusal to remove a dirty worktree would leave every
    rejected run's tree on disk forever.

    The branch is deleted second and without `check`: if the worktree removal
    already unregistered it, or the user checked it out somewhere, the run has
    still ended and failing here would turn a cleanup into an error the user
    has to act on.
    """
    git(workspace.repo_root, "worktree", "remove", "--force", str(workspace.path), check=False)
    git(workspace.repo_root, "branch", "-D", workspace.branch, check=False)
