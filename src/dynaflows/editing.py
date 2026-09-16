"""Opening the user's editor on a gate's text, and the fallback when there is none.

A module of its own rather than four functions in `cli.py`, for one reason
that is about enforcement and not about tidiness.

ADR-025 bans importing `subprocess` outside `executor/`, because a node that
can spawn a process can do everything the filesystem rule forbids and more.
`cli.py` legitimately needs to spawn exactly one thing -- the editor the user
asked for, in the foreground, at a gate. Exempting `cli.py` would have granted
a thousand-line module the right to spawn anything, to serve seventy lines
that need it, and `cli.py` is about to grow a `change` command. The allowlist
comment in `scripts/lint_architecture.py` already says exemptions are listed
by name "rather than by a pattern that would quietly admit the next module
too"; a module-wide exemption for a function-sized need is that pattern
wearing a name.

So the need moved to where the exemption can be small. This file may spawn a
process. `cli.py` may not, and neither may anything else outside `executor/`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

__all__ = ["edit_text", "find_editor", "read_multiline"]

# stderr: at a gate, stdout belongs to the brief, because the brief is what
# the user pipes somewhere else.
_err = Console(stderr=True)
_out = Console()


def find_editor() -> list[str] | None:
    """The editor to open, or None.

    Order matters. An explicit $VISUAL/$EDITOR is the user's own choice and
    wins. Otherwise nano before vim before vi: someone who never set $EDITOR
    is unlikely to be a vi user, and dropping them into modal editing with no
    warning is its own kind of trap.
    """
    for variable in ("VISUAL", "EDITOR"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value.split()
    for candidate in ("nano", "vim", "vi"):
        if shutil.which(candidate):
            return [candidate]
    return None


def _edit_in_editor(initial: str) -> str | None:
    """Open the text pre-filled. None if no editor, or the editor failed."""
    editor = find_editor()
    if editor is None:
        return None
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as handle:
        handle.write(initial)
        path = handle.name
    try:
        _err.print(f"[dim]opening {editor[0]}…[/]")
        result = subprocess.run([*editor, path], check=False)  # noqa: S603
        if result.returncode != 0:
            return None
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def read_multiline(current: str) -> str:
    """Last resort when no editor exists at all.

    A one-line `prompt` was the first version of this and it was bad twice
    over: retyping a paragraph into a single line is miserable, and with no
    default an empty Enter re-asked forever. Empty input now KEEPS the text,
    because "I changed my mind about editing" is the likeliest reason someone
    submits nothing.
    """
    _out.print(
        "[dim]No editor found. Paste the replacement brief, then a line containing only '.'[/]"
    )
    _out.print("[dim]Submit nothing to keep the text above unchanged.[/]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip() or current


def edit_text(initial: str) -> str:
    """Always returns usable text. Never loops, never returns empty."""
    edited = _edit_in_editor(initial)
    if edited is None:
        return read_multiline(initial)
    return edited.strip() or initial
