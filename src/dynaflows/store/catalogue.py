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
# MEASURED, not guessed: this repository's 90 files with their symbol lists
# come to 3,913 tokens. 4,000 truncated it the moment symbols were added, so
# the budget is raised with headroom rather than to the measurement. The
# planner's whole system prompt was 6,309 tokens against a 32,000-token floor,
# so this is affordable; a repository large enough to truncate at 6,000 is the
# one that needs source SEARCH rather than a longer list.
CATALOGUE_BUDGET_TOKENS = 6_000

_SUMMARY_CHARS = 90

# Enough to name what a module is for, not so many that one file's API drowns
# the rest of the repository. PLACEHOLDER (playbook 4.5): the number that
# should set this is the catalogue's total token cost against its budget.
_MAX_SYMBOLS = 14


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


def _is_test(path: Path) -> bool:
    return path.name.startswith("test_") or "tests" in path.parts


def _symbols(tree: ast.Module) -> list[str]:
    """Top-level function and class names, public ones first.

    A docstring says what a file is FOR; these say what is IN it, and the
    difference decided a live run. Asked to locate a bug in `resume`, the
    enhancer was shown `cli.py (53417 bytes) -- Terminal entry point.` and
    picked a test file instead -- correctly, because the only catalogue lines
    mentioning resume were tests. `cli.py` contains `def resume(` and the
    catalogue had no way to say so.
    """
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    public = [n for n in names if not n.startswith("_")]
    private = [n for n in names if n.startswith("_")]
    return (public + private)[:_MAX_SYMBOLS]


def _summary(path: Path) -> str:
    """One line about a file, cheaply.

    A path alone tells the planner a file exists; `client.py -- the resiliency
    ladder` tells it which file to send a worker to, and the symbol list tells
    it which file defines the thing the user actually named. Parsing is
    best-effort and silent on failure: a catalogue that raises on one odd file
    is a catalogue nobody can build.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return ""
    symbols: list[str] = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return ""
        doc = ast.get_docstring(tree) or ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        # Symbols exist to locate a DEFINITION. A test module defines tests,
        # not the thing under test, so its list is fourteen `test_*` names --
        # the longest lines in the catalogue and the least useful. Its
        # docstring already says what it covers. Skipping them took this
        # repository from 82 of 90 files listed to all 90.
        if not _is_test(path):
            symbols = _symbols(tree)
    elif path.suffix == ".md":
        first = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), ""
        )
    else:
        first = ""
    line = first[:_SUMMARY_CHARS]
    if symbols:
        line = (
            f"{line} · defines: {', '.join(symbols)}" if line else f"defines: {', '.join(symbols)}"
        )
    return line


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
    exhausted = False
    for candidate in candidates:
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            continue
        relative = str(candidate.relative_to(root))
        summary = _summary(candidate)
        # The path is recorded whether or not it fits the DISPLAY budget.
        # `paths` is the validation set -- a plan naming a real file must not
        # be rejected because the rendered list ran out of room. Conflating the
        # two made a budget into a correctness rule, and adding symbols to the
        # summary was enough to trip it: 90 files became 71 listed.
        paths.append(relative)
        if exhausted:
            continue
        line = f"{relative} ({size} bytes)" + (f" -- {summary}" if summary else "")
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens:
            exhausted = True
            continue
        used += cost
        lines.append(line)

    return SourceCatalogue(
        text="\n".join(lines),
        paths=frozenset(paths),
        listed=len(lines),
        total=len(paths),
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
