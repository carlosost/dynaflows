"""Markdown -> Chunk. ADR-009.

Chunked on **heading structure, never fixed token windows**. A window boundary
falls in the middle of an argument; a heading boundary is where the author
already decided one idea ends.

Pure: text in, chunks out. No filesystem, no database, no network -- which is
what lets the whole rule live in the deterministic tier.
"""

from __future__ import annotations

import hashlib
import re

from markdown_it import MarkdownIt

from dynaflows.contracts.playbook import Chunk
from dynaflows.playbook.tokens import estimate_tokens

_MD = MarkdownIt("commonmark")

# Anchors are taken from the HEADING ONLY, never the body.
#
# `by_anchor("AP-11")` must return the section that DEFINES AP-11, not the
# dozen sections that mention it in passing. Mentions are what search() is for.
# Conflating the two would make the primary retrieval path noisy exactly where
# it is supposed to be exact.
_AP_RE = re.compile(r"\bAP-(\d+)\b")
_ADR_RE = re.compile(r"\bADR-(\d+)\b")
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)[.\s]")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_STRIP.sub("-", text.lower()).strip("-")


def anchors_for(title: str) -> tuple[str, ...]:
    """Addressable ids for one heading, in stable order.

    Every chunk gets a slug so it is nameable even without a formal id. Slugs
    can collide across documents; `by_anchor` then returns both, which is the
    honest answer to an ambiguous request -- the catalogue shows heading paths,
    so a caller that needs one can pick the formal anchor instead.
    """
    found: list[str] = []
    for match in _AP_RE.finditer(title):
        found.append(f"AP-{int(match.group(1)):02d}")
    for match in _ADR_RE.finditer(title):
        found.append(f"ADR-{int(match.group(1)):03d}")
    if section := _SECTION_RE.match(title.strip()):
        found.append(f"§{section.group(1)}")
    if slug := slugify(title):
        found.append(slug)
    # dict.fromkeys: dedupe while preserving order. A set would not be stable.
    return tuple(dict.fromkeys(found))


def chunk_id(source_path: str, heading_path: str) -> str:
    return hashlib.sha256(f"{source_path}\x00{heading_path}".encode()).hexdigest()[:16]


def _headings(text: str) -> list[tuple[int, int, str]]:
    """(line, level, title) for every ATX/setext heading, in document order."""
    tokens = _MD.parse(text)
    out: list[tuple[int, int, str]] = []
    for i, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        inline = tokens[i + 1] if i + 1 < len(tokens) else None
        title = inline.content.strip() if inline is not None else ""
        out.append((token.map[0], int(token.tag[1:]), title))
    return out


def chunk_markdown(text: str, source_path: str) -> list[Chunk]:
    """One chunk per heading, holding only the text directly beneath it.

    A parent section does not swallow its children -- the heading path is what
    restores the ancestry, at a cost of a few tokens instead of duplicating
    every subsection into its parent.
    """
    lines = text.splitlines()
    headings = _headings(text)
    chunks: list[Chunk] = []

    def emit(heading_path: str, title: str, body_lines: list[str]) -> None:
        body = "\n".join(body_lines).strip()
        if not body:
            return  # a heading whose children carry all the text
        chunks.append(
            Chunk(
                id=chunk_id(source_path, heading_path),
                source_path=source_path,
                heading_path=heading_path,
                anchors=anchors_for(title),
                body=body,
                tokens=estimate_tokens(body),
            )
        )

    if headings and headings[0][0] > 0:
        emit(source_path, source_path, lines[: headings[0][0]])
    elif not headings:
        emit(source_path, source_path, lines)
        return chunks

    stack: list[tuple[int, str]] = []
    for index, (line, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        emit(" > ".join(t for _, t in stack), title, lines[line + 1 : end])

    return chunks
