"""What the planner is shown about the code. ADR-018.

ADR-009 gave the planner a compact map of the playbook so one frontier call
could route context for every worker. This is the same idea for source: the
planner cannot name a file it has never heard of, and run `w1` proved what it
does instead -- it declines to name anything, and a worker fills the vacuum.

Built from the SAME rules that govern reading (`sources.py`), so the planner
cannot name a file the resolver would then refuse. Two rule sets here would be
two rule sets that disagree, and the disagreement would show up as a refusal at
dispatch with nothing explaining it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from dynaflows.playbook.tokens import estimate_tokens
from dynaflows.store.sources import MAX_FILE_BYTES, is_listable

# PLACEHOLDER (playbook 4.5). The playbook catalogue measured 3,956 tokens, so
# this doubles the planner's map without doubling it again. Step 1.8 measures
# what a real repository needs.
CATALOGUE_BUDGET_TOKENS = 4_000

_SUMMARY_CHARS = 90


@dataclass(frozen=True, slots=True)
class SourceCatalogue:
    """The map, and honestly how much of the territory it covers."""

    text: str
    paths: frozenset[str]
    listed: int
    total: int

    @property
    def truncated(self) -> bool:
        return self.listed < self.total

    def render(self) -> str:
        """What goes in the prompt.

        Truncation is stated, not implied. A planner shown a partial tree it
        believes is complete will plan confidently against files it cannot see
        -- AP-19 relocated into a prompt, and strictly worse there because
        nobody reviews a prompt.
        """
        if not self.total:
            return "(no source files found under the project root)"
        if not self.truncated:
            return self.text
        return (
            f"{self.text}\n\n"
            f"... TRUNCATED: {self.listed} of {self.total} files listed. The rest were "
            f"omitted for space. Do NOT assume a file is absent because it is missing "
            f"from this list; name only paths you can see here."
        )


def _summary(path: Path) -> str:
    """One line about a file, cheaply.

    A path alone tells the planner a file exists; `client.py -- the resiliency
    ladder` tells it which file to send a worker to. Parsing is best-effort and
    silent on failure: a catalogue that raises on one odd file is a catalogue
    nobody can build.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return ""
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:
            doc = ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
    elif path.suffix == ".md":
        first = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), ""
        )
    else:
        first = ""
    return first[:_SUMMARY_CHARS]


def build_catalogue(root: Path, *, budget_tokens: int = CATALOGUE_BUDGET_TOKENS) -> SourceCatalogue:
    """Every listable file under `root`, until the budget runs out.

    Never raises: it runs inside the planner node, and an unreadable directory
    is a fact about the map rather than a reason to end a run.
    """
    try:
        candidates = sorted(p for p in root.rglob("*") if p.is_file() and is_listable(p, root))
    except OSError:
        return SourceCatalogue("(the project root could not be read)", frozenset(), 0, 0)

    lines: list[str] = []
    paths: list[str] = []
    used = 0
    for candidate in candidates:
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            continue
        relative = str(candidate.relative_to(root))
        summary = _summary(candidate)
        line = f"{relative} ({size} bytes)" + (f" -- {summary}" if summary else "")
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens:
            break
        used += cost
        lines.append(line)
        paths.append(relative)

    return SourceCatalogue(
        text="\n".join(lines),
        paths=frozenset(paths),
        listed=len(paths),
        total=len(candidates),
    )


def unknown_paths(requested: list[str], catalogue: SourceCatalogue) -> list[str]:
    """Which of these the planner invented. ADR-018's second half.

    A directory is accepted when the catalogue lists anything beneath it: the
    catalogue is a file list, and `src/dynaflows/gateway` is a legitimate input
    even though no line says exactly that.
    """
    missing: list[str] = []
    for item in requested:
        normalised = item.strip().strip("/")
        if not normalised:
            missing.append(item)
            continue
        if normalised in catalogue.paths:
            continue
        prefix = normalised + "/"
        if any(known.startswith(prefix) for known in catalogue.paths):
            continue
        missing.append(item)
    return missing
