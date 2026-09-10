"""The FTS5 index and the retrieval interface. ADR-009.

Uses a real SQLite file under tmp_path. That is still the deterministic tier by
this project's definition -- no network, no provider, no filesystem outside tmp
-- and it is the right call here: ADR-009's central claim is that SQLite's own
FTS5 is sufficient, and mocking SQLite would test the mock instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.contracts.errors import DynaflowsError
from dynaflows.playbook import store
from dynaflows.playbook.chunker import chunk_markdown
from dynaflows.playbook.repository import (
    InMemoryPlaybookRepository,
    SqlitePlaybookRepository,
    catalog_line,
    index_corpus,
)

pytestmark = pytest.mark.deterministic

CORPUS = {
    "playbook.md": (
        "# Playbook\n\n"
        "## 3.3 Service Integration Patterns\n\n"
        "Idempotency keys on every mutating operation. Without them a retry double charges.\n\n"
        "## 4.4 Anti-Pattern Catalog\n\n"
        "### AP-11: Parallel Abstraction Exercised Only by Tests\n\n"
        "Coverage looks healthy but the entry point calls a different implementation.\n\n"
        "### AP-02: Mocking at the Wrong Layer\n\n"
        "Tests mocked the implementation class instead of the factory function.\n"
    ),
    "notes.md": (
        "# Notes\n\n"
        "## Overview\n\n"
        "A second document, so slug collisions and multi-source ordering are exercised.\n\n"
        "## Retry Policy\n\n"
        "We discuss AP-11 here without defining it. Mentions must not become anchors.\n"
    ),
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def repo(corpus: Path, tmp_path: Path):  # noqa: ANN201
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)
    yield SqlitePlaybookRepository(connection, corpus)
    connection.close()


# --- indexing ------------------------------------------------------------


def test_indexing_populates_chunks_from_every_file(repo) -> None:  # noqa: ANN001
    assert repo.count() > 0
    assert {c.source_path for c in repo.by_anchor(["overview", "AP-11"])} == {
        "notes.md",
        "playbook.md",
    }


def test_reindexing_replaces_rather_than_duplicates(corpus: Path, tmp_path: Path) -> None:
    connection = store.connect(tmp_path / "p.db")
    first = index_corpus(connection, corpus)
    second = index_corpus(connection, corpus)
    assert first == second
    repo = SqlitePlaybookRepository(connection, corpus)
    assert repo.count() == first
    connection.close()


def test_a_missing_corpus_is_a_config_error_not_a_crash(tmp_path: Path) -> None:
    connection = store.connect(tmp_path / "p.db")
    with pytest.raises(DynaflowsError):
        index_corpus(connection, tmp_path / "absent")
    connection.close()


# --- by_anchor: the primary path ----------------------------------------


def test_anchor_lookup_returns_the_defining_section(repo) -> None:  # noqa: ANN001
    chunks = repo.by_anchor(["AP-11"])
    assert len(chunks) == 1
    assert chunks[0].source_path == "playbook.md"
    assert "Coverage looks healthy" in chunks[0].body


def test_anchor_lookup_ignores_mere_mentions(repo) -> None:  # noqa: ANN001
    """notes.md discusses AP-11 in its body. That must not make it an AP-11
    result, or the exact path becomes as noisy as the fuzzy one."""
    assert [c.source_path for c in repo.by_anchor(["AP-11"])] == ["playbook.md"]


def test_anchor_lookup_preserves_the_requested_order(repo) -> None:  # noqa: ANN001
    """pack() fills in the order it is handed, so priority has to survive."""
    assert [c.anchors[0] for c in repo.by_anchor(["AP-02", "AP-11"])] == ["AP-02", "AP-11"]
    assert [c.anchors[0] for c in repo.by_anchor(["AP-11", "AP-02"])] == ["AP-11", "AP-02"]


def test_repeated_anchors_are_returned_once(repo) -> None:  # noqa: ANN001
    assert len(repo.by_anchor(["AP-11", "AP-11"])) == 1


def test_an_unknown_anchor_returns_nothing_rather_than_guessing(repo) -> None:  # noqa: ANN001
    assert repo.by_anchor(["AP-99"]) == []


def test_section_anchors_work(repo) -> None:  # noqa: ANN001
    chunks = repo.by_anchor(["§3.3"])
    assert len(chunks) == 1
    assert "Idempotency" in chunks[0].body


# --- search: the fallback ------------------------------------------------


def test_search_finds_a_section_by_its_body(repo) -> None:  # noqa: ANN001
    hits = repo.search("idempotency retry charge", k=3)
    assert hits
    assert "§3.3" in hits[0].anchors


def test_search_ranks_a_section_about_a_term_above_one_mentioning_it(repo) -> None:  # noqa: ANN001
    hits = repo.search("mocking factory", k=3)
    assert hits and "AP-02" in hits[0].anchors


def test_search_survives_fts5_hostile_input(repo) -> None:  # noqa: ANN001
    """A planner's phrase is DATA. Bare quotes, NEAR, * and - are FTS5 syntax
    and would raise -- the same lesson Rich taught, in a different parser."""
    for hostile in ['NOT "unbalanced', "foo* -bar", 'a AND (b OR "', "^", "   "]:
        assert isinstance(repo.search(hostile, k=3), list)


def test_search_with_no_usable_terms_returns_nothing(repo) -> None:  # noqa: ANN001
    assert repo.search("!!! ?", k=3) == []


# --- catalogue -----------------------------------------------------------


def test_the_catalogue_lists_every_chunk_once(repo) -> None:  # noqa: ANN001
    assert len(repo.catalog().splitlines()) == repo.count()


def test_a_catalogue_row_prefers_formal_anchors_over_the_slug(repo) -> None:  # noqa: ANN001
    """Every byte is paid for on every planner call; a slug restates the
    heading printed two columns over."""
    row = next(r for r in repo.catalog().splitlines() if r.startswith("AP-11"))
    assert "parallel-abstraction" not in row


def test_a_catalogue_row_falls_back_to_the_slug_when_there_is_no_formal_anchor(
    repo,  # noqa: ANN001
) -> None:
    assert any(r.startswith("overview |") for r in repo.catalog().splitlines())


def test_catalogue_rows_are_trimmed_to_the_last_two_heading_levels(repo) -> None:  # noqa: ANN001
    chunk = repo.by_anchor(["AP-11"])[0]
    assert chunk.heading_path.count(" > ") >= 2
    assert catalog_line(chunk).split(" | ")[1].count(" > ") == 1


# --- drift ---------------------------------------------------------------


def test_a_fresh_index_reports_clean(repo) -> None:  # noqa: ANN001
    report = repo.drift()
    assert report.clean
    assert "matches every source" in report.summary()


def test_an_edited_source_is_reported_as_changed(repo, corpus: Path) -> None:  # noqa: ANN001
    (corpus / "playbook.md").write_text("# Playbook\n\nrewritten\n", encoding="utf-8")
    report = repo.drift()
    assert report.changed == ("playbook.md",)
    assert report.added == () and report.removed == ()


def test_a_new_file_is_reported_as_unindexed_not_changed(repo, corpus: Path) -> None:  # noqa: ANN001
    """AP-20: 'nobody has indexed this yet' and 'this was edited behind our
    back' need different actions and must not share a counter."""
    (corpus / "third.md").write_text("# Third\n\nnew\n", encoding="utf-8")
    report = repo.drift()
    assert report.added == ("third.md",)
    assert report.changed == ()


def test_a_deleted_source_is_reported_as_removed(repo, corpus: Path) -> None:  # noqa: ANN001
    (corpus / "notes.md").unlink()
    report = repo.drift()
    assert report.removed == ("notes.md",)
    assert report.changed == () and report.added == ()


def test_reindexing_after_a_deletion_forgets_its_chunks(corpus: Path, tmp_path: Path) -> None:
    connection = store.connect(tmp_path / "p.db")
    index_corpus(connection, corpus)
    (corpus / "notes.md").unlink()
    index_corpus(connection, corpus)
    repo = SqlitePlaybookRepository(connection, corpus)
    assert repo.by_anchor(["overview"]) == []
    assert repo.drift().clean
    connection.close()


# --- the in-memory implementation honours the same contract --------------


def test_the_in_memory_repository_matches_the_sqlite_one_on_anchors(repo) -> None:  # noqa: ANN001
    chunks = chunk_markdown(CORPUS["playbook.md"], "playbook.md")
    memory = InMemoryPlaybookRepository(chunks)
    assert [c.heading_path for c in memory.by_anchor(["AP-11"])] == [
        c.heading_path for c in repo.by_anchor(["AP-11"])
    ]
