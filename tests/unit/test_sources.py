"""ADR-017's read boundary. Four refusals, counted apart (AP-20).

"We could not show the worker that file" is not one fact. A path outside the
project, a symlink leaving it, something that looks like a credential and a
file too large to send are four different problems with four different
answers, and a single `skipped` count answers none of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.store.sources import MAX_FILE_BYTES, MAX_FILES_PER_INPUT, Refusal, read_sources

pytestmark = pytest.mark.deterministic


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def login(): ...\n", encoding="utf-8")
    (tmp_path / "src" / "db.py").write_text("ENGINE = 'pg'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-real-secret\n", encoding="utf-8")
    return tmp_path


def _reasons(refusals: list) -> set[Refusal]:
    return {r.reason for r in refusals}


def test_a_named_file_is_read_and_carries_its_path(root: Path) -> None:
    chunks, refusals = read_sources(["src/auth.py"], root)

    assert refusals == []
    assert len(chunks) == 1
    assert chunks[0].source_path == "src/auth.py"
    assert "def login" in chunks[0].body
    # The heading is the path: pack() prepends it, so the worker always knows
    # which file it is looking at even after truncation.
    assert chunks[0].heading_path == "src/auth.py"


def test_a_directory_expands_in_a_stable_order(root: Path) -> None:
    chunks, _ = read_sources(["src"], root)

    assert [c.source_path for c in chunks] == ["src/auth.py", "src/db.py"]


def test_the_requested_order_survives(root: Path) -> None:
    """pack() fills a budget in the order it is handed, so the planner's
    priority has to survive the round trip or a tight budget drops the wrong
    file."""
    chunks, _ = read_sources(["src/db.py", "src/auth.py"], root)

    assert [c.source_path for c in chunks] == ["src/db.py", "src/auth.py"]


def test_the_same_file_named_twice_is_read_once(root: Path) -> None:
    chunks, _ = read_sources(["src/auth.py", "src", "src/auth.py"], root)

    assert [c.source_path for c in chunks].count("src/auth.py") == 1


# --- the four refusals ---------------------------------------------------


def test_a_path_outside_the_project_is_refused(root: Path) -> None:
    _, refusals = read_sources(["../../etc/passwd"], root)

    assert _reasons(refusals) == {Refusal.OUTSIDE_PROJECT}


def test_a_symlink_leaving_the_project_is_refused(root: Path, tmp_path: Path) -> None:
    """The check resolves both sides, so a symlink is caught by the same rule
    as a literal `../`. Two rules here would be two rules that can disagree."""
    outside = tmp_path.parent / "outside_secret.py"
    outside.write_text("TOKEN = 'x'\n", encoding="utf-8")
    (root / "src" / "link.py").symlink_to(outside)

    chunks, refusals = read_sources(["src/link.py"], root)

    assert chunks == []
    assert _reasons(refusals) == {Refusal.OUTSIDE_PROJECT}


def test_a_credential_file_is_refused_even_when_explicitly_named(root: Path) -> None:
    """An 'audit auth' task is the one most likely to name the directory the
    secrets live in. The planner asking for it is not permission to send it."""
    chunks, refusals = read_sources([".env"], root)

    assert chunks == []
    assert _reasons(refusals) == {Refusal.LOOKS_SECRET}


def test_a_secret_inside_a_requested_directory_is_refused_too(root: Path) -> None:
    """The dangerous case is not naming .env -- it is naming the folder."""
    chunks, refusals = read_sources(["."], root)

    assert not any("sk-real-secret" in c.body for c in chunks)
    assert Refusal.LOOKS_SECRET in _reasons(refusals)


def test_a_file_over_the_cap_is_refused_and_says_how_big(root: Path) -> None:
    (root / "src" / "huge.py").write_text("x" * (MAX_FILE_BYTES + 1), encoding="utf-8")

    _, refusals = read_sources(["src/huge.py"], root)

    assert _reasons(refusals) == {Refusal.TOO_LARGE}
    assert str(MAX_FILE_BYTES + 1) in refusals[0].detail


def test_a_binary_file_is_refused_by_suffix_not_by_crashing(root: Path) -> None:
    (root / "src" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01")

    chunks, refusals = read_sources(["src/logo.png"], root)

    assert chunks == []
    assert _reasons(refusals) == {Refusal.NOT_TEXT}


def test_a_missing_path_is_not_found_rather_than_silence(root: Path) -> None:
    _, refusals = read_sources(["src/nope.py"], root)

    assert _reasons(refusals) == {Refusal.NOT_FOUND}


def test_an_oversized_directory_is_truncated_and_says_so(root: Path) -> None:
    big = root / "big"
    big.mkdir()
    for i in range(MAX_FILES_PER_INPUT + 5):
        (big / f"m{i:03d}.py").write_text("pass\n", encoding="utf-8")

    chunks, refusals = read_sources(["big"], root)

    assert len(chunks) == MAX_FILES_PER_INPUT
    assert _reasons(refusals) == {Refusal.TOO_MANY}
    # AP-20 again: the refusal has to say it was a cap, not just that files are
    # missing, because the fix is "name the files" and nothing else says that.
    assert "cap" in refusals[0].detail


def test_the_skip_list_keeps_vendored_trees_out(root: Path) -> None:
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("x\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    chunks, _ = read_sources(["."], root)

    assert not any("node_modules" in c.source_path for c in chunks)
    assert not any(".git" in c.source_path for c in chunks)


def test_reading_never_raises(root: Path) -> None:
    """It runs at dispatch. An unreadable path is a fact about one task, not a
    reason to end a run."""
    chunks, refusals = read_sources(["", "..", "/etc", "src/auth.py"], root)

    assert any(c.source_path == "src/auth.py" for c in chunks)
    assert refusals
