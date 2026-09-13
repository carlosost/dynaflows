"""ADR-018: the planner is shown what exists, and cannot name what does not.

Live run `w1` is the subject. The planner had never been shown the repository,
so `inputs` was guesswork -- right by luck on one thread, abandoned entirely on
the next, and a worker filled the vacuum with four invented filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.store.catalogue import build_catalogue, unknown_paths
from dynaflows.store.sources import read_sources

pytestmark = pytest.mark.deterministic


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        '"""Login and session handling."""\n\ndef login(): ...\n', encoding="utf-8"
    )
    (tmp_path / "src" / "db.py").write_text('"""Engine config."""\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("# The Project\n\nWords.\n", encoding="utf-8")
    (tmp_path / ".env").write_text("KEY=secret\n", encoding="utf-8")
    return tmp_path


def test_a_file_is_listed_with_what_it_is(repo: Path) -> None:
    """A path alone says a file exists. The docstring says which file to send a
    worker to, which is the whole point of showing the planner anything."""
    text = build_catalogue(repo).render()

    assert "src/auth.py" in text
    assert "Login and session handling." in text
    assert "The Project" in text


def test_cache_directories_do_not_fill_the_catalogue(repo: Path) -> None:
    """Regression. The first catalogue built from this repo came out 60%
    .pytest_cache and .ruff_cache, because the skip rule was a deny-list and a
    deny-list only knows the tools that existed when it was written."""
    for cache in (".pytest_cache", ".ruff_cache/0.16.6", ".mypy_cache", "node_modules/pkg"):
        directory = repo / cache
        directory.mkdir(parents=True)
        (directory / "junk.json").write_text("{}", encoding="utf-8")

    catalogue = build_catalogue(repo)

    assert not any("cache" in path for path in catalogue.paths)
    assert not any("node_modules" in path for path in catalogue.paths)


def test_ci_configuration_survives_the_dot_directory_rule(repo: Path) -> None:
    """The allow-list has to actually allow something, or 'audit the CI' is
    unanswerable."""
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")

    assert ".github/workflows/ci.yml" in build_catalogue(repo).paths


def test_a_credential_file_is_never_advertised(repo: Path) -> None:
    """The catalogue must not name a file the resolver would refuse: the
    planner would name it, and the refusal would arrive at dispatch with
    nothing explaining it."""
    assert ".env" not in build_catalogue(repo).paths


def test_the_catalogue_and_the_resolver_agree(repo: Path) -> None:
    """One predicate, checked from both sides. Two rule sets here would be two
    rule sets that can disagree."""
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "j.json").write_text("{}", encoding="utf-8")

    catalogue = build_catalogue(repo)
    chunks, _ = read_sources(["."], repo)

    assert {c.source_path for c in chunks} == set(catalogue.paths)


def test_truncation_is_stated_rather_than_implied(repo: Path) -> None:
    """A planner shown a partial tree it believes is complete plans
    confidently against files it cannot see -- AP-19 relocated into a prompt,
    and worse there, because nobody reviews a prompt."""
    for i in range(60):
        (repo / "src" / f"m{i:03d}.py").write_text('"""A module."""\n', encoding="utf-8")

    catalogue = build_catalogue(repo, budget_tokens=120)

    assert catalogue.truncated
    assert "TRUNCATED" in catalogue.render()
    assert f"of {catalogue.total} files" in catalogue.render()


def test_a_complete_catalogue_says_nothing_about_truncation(repo: Path) -> None:
    """The warning must not fire on the ordinary path or it becomes noise."""
    catalogue = build_catalogue(repo)

    assert not catalogue.truncated
    assert "TRUNCATED" not in catalogue.render()


def test_an_empty_tree_says_so_instead_of_rendering_nothing(tmp_path: Path) -> None:
    assert "no source files" in build_catalogue(tmp_path).render()


# --- the half that makes a bad plan impossible to run silently ------------


def test_an_exact_path_is_known(repo: Path) -> None:
    assert unknown_paths(["src/auth.py"], build_catalogue(repo)) == []


def test_a_directory_is_known_when_anything_lives_under_it(repo: Path) -> None:
    """The catalogue lists files; `src` is a legitimate input even though no
    line says exactly that."""
    catalogue = build_catalogue(repo)

    assert unknown_paths(["src"], catalogue) == []
    assert unknown_paths(["src/"], catalogue) == []


def test_an_invented_path_is_reported(repo: Path) -> None:
    """`w1` again: gateway/logger.py, gateway/middleware.py, gateway/handlers.py
    and gateway/metrics.py, none of which existed."""
    invented = ["src/logger.py", "src/middleware.py"]

    assert unknown_paths([*invented, "src/auth.py"], build_catalogue(repo)) == invented


def test_an_empty_string_is_not_a_path(repo: Path) -> None:
    assert unknown_paths(["", "   "], build_catalogue(repo)) == ["", "   "]


# --- a docstring says what a file is FOR, not what is IN it --------------


def test_a_module_lists_what_it_defines(repo: Path) -> None:
    """Live run `a1`: asked to locate a bug in `resume`, the enhancer was shown
    `cli.py (53417 bytes) -- Terminal entry point.` and picked a test file.
    Correctly — the only catalogue lines mentioning resume WERE tests. The file
    containing `def resume(` had no way to say so."""
    (repo / "src" / "cli.py").write_text(
        '"""Terminal entry point."""\n\ndef run(): ...\n\ndef resume(): ...\n\nclass Runner: ...\n',
        encoding="utf-8",
    )

    line = next(ln for ln in build_catalogue(repo).text.splitlines() if "cli.py" in ln)

    assert "Terminal entry point." in line
    assert "defines:" in line
    assert "resume" in line
    assert "Runner" in line


def test_public_names_come_before_private_ones(repo: Path) -> None:
    (repo / "src" / "m.py").write_text(
        "def _helper(): ...\n\ndef public(): ...\n", encoding="utf-8"
    )

    line = next(ln for ln in build_catalogue(repo).text.splitlines() if "m.py" in ln)

    assert line.index("public") < line.index("_helper")


def test_a_test_module_lists_no_symbols(repo: Path) -> None:
    """A test module defines tests, not the thing under test. Fourteen `test_*`
    names were the longest lines in the catalogue and the least useful; its
    docstring already says what it covers."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_auth.py").write_text(
        '"""Login behaviour."""\n\ndef test_one(): ...\n\ndef test_two(): ...\n',
        encoding="utf-8",
    )

    line = next(ln for ln in build_catalogue(repo).text.splitlines() if "test_auth" in ln)

    assert "Login behaviour." in line
    assert "defines:" not in line


def test_a_file_with_no_docstring_still_lists_its_symbols(repo: Path) -> None:
    (repo / "src" / "bare.py").write_text("def handler(): ...\n", encoding="utf-8")

    line = next(ln for ln in build_catalogue(repo).text.splitlines() if "bare.py" in ln)

    assert "defines: handler" in line


# --- the display budget is not a validation set --------------------------


def test_a_file_truncated_out_of_the_listing_is_still_a_known_path(repo: Path) -> None:
    """Conflating the two made a budget into a correctness rule: a plan naming
    a real file would be rejected because the rendered list ran out of room.
    Adding symbols to the summary was enough to trip it."""
    for i in range(40):
        (repo / "src" / f"mod{i:02d}.py").write_text(
            f'"""Module {i} does a thing worth describing at length."""\n\ndef f{i}(): ...\n',
            encoding="utf-8",
        )

    catalogue = build_catalogue(repo, budget_tokens=200)

    assert catalogue.truncated
    assert catalogue.listed < catalogue.total
    assert len(catalogue.paths) == catalogue.total
    # Every real file validates, listed or not.
    assert unknown_paths(["src/mod39.py"], catalogue) == []
    # And invention is still caught.
    assert unknown_paths(["src/nope.py"], catalogue) == ["src/nope.py"]
