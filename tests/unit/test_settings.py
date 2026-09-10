"""Contract tests for environment handling.

The .env parser is hand-rolled to keep the dependency surface honest, which
means it has to be tested like any other parser -- quotes, comments, export
prefixes, and the precedence rule that a real env var beats the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynaflows.settings import REQUIRED_VARS, check_env, get_settings, load_dotenv

pytestmark = pytest.mark.deterministic


def test_check_env_lists_every_missing_var() -> None:
    assert check_env({}) == list(REQUIRED_VARS)


def test_check_env_treats_empty_string_as_missing() -> None:
    env = dict.fromkeys(REQUIRED_VARS, "x") | {REQUIRED_VARS[0]: ""}
    assert check_env(env) == [REQUIRED_VARS[0]]


def test_check_env_passes_when_all_set() -> None:
    assert check_env(dict.fromkeys(REQUIRED_VARS, "x")) == []


def test_load_dotenv_handles_comments_quotes_and_export(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=one",
                'QUOTED="two"',
                "SINGLE='three'",
                "export EXPORTED=four",
                "  SPACED  =  five  ",
                "NOEQUALS",
            ]
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    loaded = load_dotenv(path, environ=env)
    assert loaded == {
        "PLAIN": "one",
        "QUOTED": "two",
        "SINGLE": "three",
        "EXPORTED": "four",
        "SPACED": "five",
    }
    assert "NOEQUALS" not in env


def test_load_dotenv_does_not_override_real_environment(tmp_path: Path) -> None:
    """A CI-injected secret must never be shadowed by a stale local file."""
    path = tmp_path / ".env"
    path.write_text("KEY=from-file", encoding="utf-8")
    env = {"KEY": "from-real-env"}
    assert load_dotenv(path, environ=env) == {}
    assert env["KEY"] == "from-real-env"


def test_load_dotenv_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env", environ={}) == {}


def test_settings_resolve_relative_home_against_project_root(project: Path) -> None:
    settings = get_settings({"DYNAFLOWS_HOME": ".dynaflows"}, root=project)
    assert settings.home == project / ".dynaflows"
    assert settings.state_db == project / ".dynaflows" / "state.db"


def test_settings_keep_absolute_home(project: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    settings = get_settings({"DYNAFLOWS_HOME": str(elsewhere)}, root=project)
    assert settings.home == elsewhere


def test_missing_api_keys_become_none_not_empty_string(project: Path) -> None:
    settings = get_settings({"OPENROUTER_API_KEY": ""}, root=project)
    assert settings.openrouter_api_key is None
