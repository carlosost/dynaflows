"""Landing an approved diff in the user's working tree. ADR-026.

This is the only code in the project that writes outside `.dynaflows/`, and
it exists because of one requirement stated plainly: review the change in an
editor, accept some hunks, reject others, commit later. `git apply` without
`--index` leaves everything unstaged, which is exactly what puts the change
in a source-control panel as hunks rather than as a branch to merge.

**What guards this, and what does not.** ADR-016 is enforced by the
architecture linter, which reads Python call sites -- `Path.write_text`,
`mkdir`, and the rest. It cannot see what a subprocess does, and `git apply`
is a subprocess. **So the boundary that matters most here is the one the
linter cannot check**, and pretending otherwise would be worse than saying it.
Two weaker guards instead, both real:

  - a test asserts the string `git apply` appears in exactly one module in
    `src/`, the same shape as the `SPAWN_MODULES` allowlist;
  - `apply_to_working_tree` takes the approved `GateOutcome` as an argument
    and refuses anything that is not an APPROVE, so a caller cannot reach
    the write without having a G3 decision in hand.

Neither makes it impossible. Together they make it impossible to do by
accident, which is the honest claim -- the same one ADR-025 makes about the
worktree.

**Failure is an outcome, not an exception.** The patch can fail to apply
because the user's tree moved while the run was paused at a gate. The branch
still exists, the work is not lost, and the user needs to hear which of those
happened -- not a traceback.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from dynaflows.contracts.state import GateDecision, GateOutcome

__all__ = ["Applied", "apply_to_working_tree"]

_TAIL_CHARS = 2000


@dataclass(frozen=True, slots=True)
class Applied:
    """What happened to the patch. `ok` is the only field that means success.

    `reason` is populated on every failure INCLUDING the refusals, so a caller
    rendering this never has to distinguish "did not apply" from "was not
    allowed to try" by inspecting which fields are empty (AP-20).
    """

    ok: bool
    reason: str = ""
    files: tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        """True when nothing was attempted, as opposed to attempted and failed.

        A user whose patch conflicted needs different advice from a user whose
        gate answer was not an approval, and the difference is invisible in
        `ok` alone.
        """
        return not self.ok and self.reason.startswith("refused:")


def _refuse(reason: str) -> Applied:
    return Applied(ok=False, reason=f"refused: {reason}")


def apply_to_working_tree(
    repo_root: Path, patch: str, gate: GateOutcome | None
) -> Applied:
    """Apply `patch` to `repo_root`, unstaged. ADR-026.

    `gate` is required and checked rather than documented, because "call this
    only after G3" is exactly the kind of rule that survives until the second
    caller. If you can enforce it, do not ask for it.
    """
    if gate is None or gate.decision is not GateDecision.APPROVE:
        return _refuse("no G3 approval; a diff reaches your tree only after you approve it")
    if not patch.strip():
        return _refuse("the patch is empty, so there is nothing to apply")

    # --check first, so a patch that cannot apply cleanly leaves the tree
    # exactly as it was. Without it a partial application is possible, and a
    # half-applied change in the user's own tree is the failure this entire
    # pipeline was built to avoid -- arriving at the last step.
    checked = _run(repo_root, ["apply", "--check", "-"], patch)
    if checked.returncode != 0:
        return Applied(
            ok=False,
            reason=(
                "the patch no longer applies to your tree -- it probably moved while the run "
                f"was paused. The branch still has the work. git said: {_tail(checked)}"
            ),
        )

    applied = _run(repo_root, ["apply", "-"], patch)
    if applied.returncode != 0:
        # Reachable: --check passed and the real apply failed. Rare, and worth
        # its own branch rather than an `assert`, because the difference
        # between "cannot apply" and "checked clean then failed anyway" is the
        # difference between a stale patch and something wrong on disk.
        return Applied(
            ok=False,
            reason=f"the patch passed --check and then failed to apply: {_tail(applied)}",
        )

    listed = _run(repo_root, ["apply", "--numstat", "-"], patch)
    files = tuple(
        line.split("\t")[-1]
        for line in listed.stdout.splitlines()
        if line.strip() and "\t" in line
    )
    return Applied(ok=True, files=files)


def _run(repo_root: Path, args: list[str], patch: str) -> subprocess.CompletedProcess[str]:
    """Feed the patch on stdin rather than via a temp file.

    A temp file would be a second place the patch exists, and the two could
    disagree if anything rewrote one of them. stdin also means no cleanup
    path that can fail after a successful apply.
    """
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), *args],
        input=patch,
        capture_output=True,
        text=True,
        check=False,
    )


def _tail(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr.strip() or completed.stdout.strip())[-_TAIL_CHARS:]
