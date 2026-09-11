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
