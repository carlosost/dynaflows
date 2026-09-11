"""Reading the files a plan names. ADR-017: once, here, never in a worker.

Four refusals, counted and reported apart (AP-20). "We could not show the
worker that file" is not one fact: a path outside the project, a symlink
leaving it, something that looks like a credential, and a file too large to
send are four different problems with four different answers, and a single
"skipped" count answers none of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dynaflows.contracts.playbook import Chunk
from dynaflows.playbook.tokens import estimate_tokens

# A worker reasons about code and prose. Anything else is bytes it cannot read
# and tokens the run pays for, so the cap is small on purpose.
MAX_FILE_BYTES = 200_000
# PLACEHOLDER (playbook 4.5): chosen so a directory input cannot silently
# become the whole repository. Step 1.8 measures what a real audit needs.
MAX_FILES_PER_INPUT = 25

# Not a security boundary -- ADR-016 already forbids writes and tools. This is
# so an "audit auth" task does not post the project's own credentials to a
# third-party model, which is the exact shape of task most likely to name the
# directory they live in.
_SECRET_NAMES = frozenset({".env", ".netrc", ".npmrc", ".pypirc", "credentials", "id_rsa"})
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")
_SECRET_STEMS = (".env.",)

_SKIP_DIRS = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", ".dynaflows", ".mypy_cache"}
)

_TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".txt",
        ".toml",
        ".cfg",
        ".ini",
        ".json",
        ".yaml",
        ".yml",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".html",
        ".css",
        ".sql",
        ".sh",
        ".rs",
        ".go",
        ".java",
        ".rb",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".env.example",
        "",
    }
)


class Refusal(StrEnum):
    NOT_FOUND = "not found"
    OUTSIDE_PROJECT = "outside the project root"
    LOOKS_SECRET = "looks like a credential file"
    TOO_LARGE = "larger than the size cap"
    NOT_TEXT = "not readable as text"
    TOO_MANY = "directory had more files than the per-input cap"


@dataclass(frozen=True, slots=True)
class SourceRefusal:
    requested: str
    reason: Refusal
    detail: str = ""

    def render(self) -> str:
        return f"{self.requested}: {self.reason}{f' ({self.detail})' if self.detail else ''}"


def _is_secret(path: Path) -> bool:
    name = path.name
    return (
        name in _SECRET_NAMES
        or name.endswith(_SECRET_SUFFIXES)
        or any(name.startswith(stem) for stem in _SECRET_STEMS)
    )


def _inside(path: Path, root: Path) -> bool:
    """Resolved on both sides, so a symlink pointing out of the tree is caught
    by the same check as a literal `../`. Doing it any other way means two
    rules that can disagree."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError, OSError:
        return False
    return True


def _chunk_of(path: Path, root: Path) -> Chunk | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return None
    relative = str(path.resolve().relative_to(root.resolve()))
    return Chunk(
        id="src:" + hashlib.sha256(relative.encode()).hexdigest()[:16],
        source_path=relative,
        heading_path=relative,
        anchors=(relative,),
        body=text,
        tokens=estimate_tokens(text),
    )


def _files_under(directory: Path) -> list[Path]:
    found: list[Path] = []
    for candidate in sorted(directory.rglob("*")):
        if any(part in _SKIP_DIRS for part in candidate.parts):
            continue
        if candidate.is_file():
            found.append(candidate)
    return found


def read_sources(inputs: list[str], root: Path) -> tuple[list[Chunk], list[SourceRefusal]]:
    """Resolve a plan task's `inputs` into chunks, plus why anything is missing.

    Returns chunks in the order requested: `pack()` fills a budget in the order
    it is handed, so the planner's priority has to survive the round trip.

    Never raises. This runs at dispatch, and an unreadable path is a fact about
    one task, not a reason to end a run.
    """
    chunks: list[Chunk] = []
    refusals: list[SourceRefusal] = []
    seen: set[str] = set()

    for requested in inputs:
        candidate = (root / requested).expanduser()
        if not _inside(candidate, root):
            refusals.append(SourceRefusal(requested, Refusal.OUTSIDE_PROJECT))
            continue
        if not candidate.exists():
            refusals.append(SourceRefusal(requested, Refusal.NOT_FOUND))
            continue

        targets = _files_under(candidate) if candidate.is_dir() else [candidate]
        if len(targets) > MAX_FILES_PER_INPUT:
            refusals.append(
                SourceRefusal(
                    requested,
                    Refusal.TOO_MANY,
                    f"{len(targets)} files, cap is {MAX_FILES_PER_INPUT}; naming files is better",
                )
            )
            targets = targets[:MAX_FILES_PER_INPUT]

        for target in targets:
            label = str(target.relative_to(root)) if _inside(target, root) else str(target)
            if not _inside(target, root):
                refusals.append(SourceRefusal(label, Refusal.OUTSIDE_PROJECT, "symlink"))
                continue
            if _is_secret(target):
                refusals.append(SourceRefusal(label, Refusal.LOOKS_SECRET))
                continue
            if target.suffix not in _TEXT_SUFFIXES:
                refusals.append(
                    SourceRefusal(label, Refusal.NOT_TEXT, target.suffix or "no suffix")
                )
                continue
            try:
                size = target.stat().st_size
            except OSError:
                refusals.append(SourceRefusal(label, Refusal.NOT_FOUND))
                continue
            if size > MAX_FILE_BYTES:
                refusals.append(SourceRefusal(label, Refusal.TOO_LARGE, f"{size} bytes"))
                continue
            chunk = _chunk_of(target, root)
            if chunk is None:
                refusals.append(SourceRefusal(label, Refusal.NOT_TEXT))
                continue
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chunks.append(chunk)

    return chunks, refusals
