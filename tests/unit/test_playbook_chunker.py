"""Chunking, anchors and token estimation. ADR-009.

The chunker slices the ORIGINAL source lines rather than re-rendering the AST,
so fidelity of code fences and tables is a contract, not a hope.
"""

from __future__ import annotations

import pytest

from dynaflows.playbook.chunker import anchors_for, chunk_id, chunk_markdown, slugify
from dynaflows.playbook.tokens import estimate_tokens

pytestmark = pytest.mark.deterministic

NESTED = """# Playbook

Intro paragraph.

## 1. Foundation

Foundation text.

### 1.3 The ADR Process

Every structural decision is recorded.

## 4. Production

### 4.4 Anti-Pattern Catalog

#### AP-11: Parallel Abstraction Exercised Only by Tests

Symptom: green in the suite, never runs in production.
"""


def test_chunks_follow_heading_structure() -> None:
    chunks = chunk_markdown(NESTED, "playbook.md")
    paths = [c.heading_path for c in chunks]
    assert "Playbook > 1. Foundation > 1.3 The ADR Process" in paths
    assert (
        "Playbook > 4. Production > 4.4 Anti-Pattern Catalog"
        " > AP-11: Parallel Abstraction Exercised Only by Tests" in paths
    )


def test_a_parent_holding_only_subheadings_emits_nothing() -> None:
    """'## 4. Production' has no prose of its own. A chunk with a heading and
    no body reads as though the section were empty."""
    paths = [c.heading_path for c in chunk_markdown(NESTED, "p.md")]
    assert "Playbook > 4. Production" not in paths


def test_a_parent_with_its_own_prose_is_kept_separately() -> None:
    """'1. Foundation' has prose AND a child. It is not swallowed by the child,
    and it does not swallow the child either."""
    chunks = {c.heading_path: c for c in chunk_markdown(NESTED, "p.md")}
    parent = chunks["Playbook > 1. Foundation"]
    assert parent.body == "Foundation text."
    assert "ADR Process" not in parent.body


def test_text_before_the_first_heading_is_not_lost() -> None:
    chunks = chunk_markdown("Preamble line.\n\n# Title\n\nBody.\n", "doc.md")
    assert chunks[0].body == "Preamble line."
    assert chunks[0].heading_path == "doc.md"


def test_a_document_with_no_headings_is_one_chunk() -> None:
    chunks = chunk_markdown("Just prose.\n\nMore prose.\n", "notes.md")
    assert len(chunks) == 1
    assert chunks[0].body == "Just prose.\n\nMore prose."


def test_code_fences_survive_verbatim() -> None:
    """The chunker slices raw lines. Re-rendering an AST would reformat this."""
    source = "# T\n\n```python\ndef f(  x ):\n    return   x\n```\n"
    body = chunk_markdown(source, "d.md")[0].body
    assert "def f(  x ):" in body
    assert "    return   x" in body


def test_tables_survive_verbatim() -> None:
    source = "# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert "|---|---|" in chunk_markdown(source, "d.md")[0].body


# --- anchors -------------------------------------------------------------


def test_anchors_come_from_the_heading_not_the_body() -> None:
    """`by_anchor('AP-11')` must return the section that DEFINES AP-11, not the
    dozen that mention it. Mentions are what search() is for."""
    source = "# T\n\n## Some Section\n\nThis paragraph discusses AP-11 and ADR-006 at length.\n"
    chunk = chunk_markdown(source, "d.md")[-1]
    assert not any(a.startswith(("AP-", "ADR-")) for a in chunk.anchors)


def test_ap_and_adr_numbers_are_zero_padded_for_stable_sorting() -> None:
    assert "AP-01" in anchors_for("AP-1: Something")
    assert "AP-11" in anchors_for("AP-11: Something")
    assert "ADR-006" in anchors_for("ADR-6: Something")


def test_section_numbers_become_paragraph_anchors() -> None:
    assert "§4.4" in anchors_for("4.4 Anti-Pattern Catalog")
    assert "§1" in anchors_for("1. Project Foundation")


def test_every_heading_gets_a_slug_so_it_is_always_nameable() -> None:
    assert "the-conflict-check" in anchors_for("The Conflict Check")
    assert slugify("4.5 Numeric Thresholds!") == "4-5-numeric-thresholds"


def test_anchor_order_is_stable_and_deduplicated() -> None:
    first = anchors_for("AP-11: AP-11 again")
    assert first == anchors_for("AP-11: AP-11 again")
    assert first.count("AP-11") == 1


def test_chunk_ids_are_stable_and_distinguish_heading_paths() -> None:
    assert chunk_id("a.md", "X > Y") == chunk_id("a.md", "X > Y")
    assert chunk_id("a.md", "X > Y") != chunk_id("a.md", "X > Z")
    assert chunk_id("a.md", "X > Y") != chunk_id("b.md", "X > Y")


# --- token estimate ------------------------------------------------------


def test_the_estimate_leans_high_on_purpose() -> None:
    """Packing under budget is safe; packing over gets truncated by the
    provider. The constant is provisional (4.5) and must never under-count."""
    text = "word " * 200
    assert estimate_tokens(text) > len(text) / 4


def test_the_estimate_is_never_zero_for_non_empty_text() -> None:
    assert estimate_tokens("a") >= 1
