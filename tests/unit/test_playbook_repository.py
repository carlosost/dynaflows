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
    index_corpus,
    section_line,
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
    assert len(repo.section_map().splitlines()) == repo.count()


def test_a_catalogue_row_prefers_formal_anchors_over_the_slug(repo) -> None:  # noqa: ANN001
    """Every byte is paid for on every planner call; a slug restates the
    heading printed two columns over."""
    row = next(r for r in repo.section_map().splitlines() if r.startswith("AP-11"))
    assert "parallel-abstraction" not in row


def test_a_catalogue_row_falls_back_to_the_slug_when_there_is_no_formal_anchor(
    repo,  # noqa: ANN001
) -> None:
    assert any(r.startswith("overview |") for r in repo.section_map().splitlines())


def test_catalogue_rows_are_trimmed_to_the_last_two_heading_levels(repo) -> None:  # noqa: ANN001
    chunk = repo.by_anchor(["AP-11"])[0]
    assert chunk.heading_path.count(" > ") >= 2
    assert section_line(chunk).split(" | ")[1].count(" > ") == 1


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


def _orphan_postings(connection, term: str) -> list[int]:  # noqa: ANN001
    """FTS rowids matching `term` that no live chunk owns.

    The fault is asserted here rather than through `search()` because
    `search()` JOINs orphans away: with the current whole-corpus reindex the
    rowids climb, the JOIN finds nothing, and a test written against the
    symptom passes over a broken index. Assert the index, not the query.
    """
    live = {r[0] for r in connection.execute("SELECT rowid FROM chunks")}
    matched = connection.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (term,)
    ).fetchall()
    return [r[0] for r in matched if r[0] not in live]


def test_reindexing_leaves_no_orphan_postings(corpus: Path, tmp_path: Path) -> None:
    """External-content FTS5 subtracts the column values handed to 'delete'.
    Empty strings subtract nothing, so every posting survived the reindex and
    the index grew without bound -- silently, and `integrity-check` passes.

    Today the damage is a search that returns FEWER results than it should,
    because `search`'s JOIN drops the orphans. It is worse than that latent:
    see `test_an_incremental_reindex_does_not_resurrect_a_deleted_term`.
    """
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)
    assert not _orphan_postings(connection, "idempotency")

    (corpus / "playbook.md").write_text(
        "# Playbook\n\n## 3.3 Service Integration Patterns\n\nLighting conduits only.\n",
        encoding="utf-8",
    )
    index_corpus(connection, corpus)

    assert _orphan_postings(connection, "idempotency") == []


def test_forgetting_a_source_leaves_no_orphan_postings(corpus: Path, tmp_path: Path) -> None:
    """`forget_sources` carried the same empty-string delete. Two call sites,
    one fault -- fixing the one the repro used would leave the other to be
    found again later, by someone with less context."""
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)

    (corpus / "playbook.md").unlink()
    index_corpus(connection, corpus)

    assert _orphan_postings(connection, "idempotency") == []


def test_an_incremental_reindex_does_not_resurrect_a_deleted_term(
    corpus: Path, tmp_path: Path
) -> None:
    """The consequence that is latent today and armed by the obvious optimisation.

    `index_corpus` currently rebuilds every file, so rowids climb and orphans
    are merely dropped by the JOIN. Reindex ONE file -- the change everybody
    proposes when they notice the waste -- and SQLite reuses the freed rowids.
    The surviving postings then point at whatever now occupies them, and a
    search for a term that exists nowhere in the corpus returns a confident
    hit on unrelated text.

    So the performance fix must not land before this one. This test is what
    says so, in the place where somebody would make that mistake.
    """
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)
    repository = SqlitePlaybookRepository(connection, corpus)
    assert repository.search("idempotency"), "precondition: the term is indexed"

    source = corpus / "playbook.md"
    source.write_text(
        "# Playbook\n\n## 3.3 Service Integration Patterns\n\nLighting conduits only.\n",
        encoding="utf-8",
    )
    store.replace_source(
        connection, source, "playbook.md", chunk_markdown(source.read_text(), "playbook.md")
    )

    for chunk in repository.search("idempotency"):
        raise AssertionError(
            f"a term that exists nowhere in the corpus matched {chunk.source_path}: {chunk.body!r}"
        )


def test_an_index_that_disagrees_with_its_chunks_is_detected(corpus: Path, tmp_path: Path) -> None:
    """A database indexed before the fix cannot heal by reindexing -- the
    surviving postings have no chunk row left to subtract them -- so the
    condition needs its own check. Drift asks "is the index stale"; this asks
    "is the index a lie" (AP-20).

    Note which check: FTS5's `integrity-check` at its DEFAULT rank passes on
    this fault, and an anti-join against `chunks_fts` also passes, because an
    external-content table reports the content table's rowids. Both were
    tried. Only rank 1 cross-checks the two.
    """
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)
    assert store.fts_is_consistent(connection)

    # Exactly what the old delete did: remove the chunk, leave the posting.
    connection.execute("DELETE FROM chunks WHERE source_path = 'playbook.md'")
    connection.commit()

    assert not store.fts_is_consistent(connection)


def test_rebuild_repairs_an_index_corrupted_before_the_fix(corpus: Path, tmp_path: Path) -> None:
    """The remedy for the databases already on disk."""
    connection = store.connect(tmp_path / "playbook.db")
    index_corpus(connection, corpus)
    connection.execute("DELETE FROM chunks WHERE source_path = 'playbook.md'")
    connection.commit()
    assert not store.fts_is_consistent(connection)

    store.rebuild_fts(connection)

    assert store.fts_is_consistent(connection)
