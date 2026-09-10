"""Contract tests for environment handling.

The .env parser is hand-rolled to keep the dependency surface honest, which
means it has to be tested like any other parser -- quotes, comments, export
prefixes, and the precedence rule that a real env var beats the file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dynaflows.settings import REQUIRED_VARS, check_env, get_settings, parse_dotenv, resolve_env

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
    loaded = parse_dotenv(path)
    assert loaded == {
        "PLAIN": "one",
        "QUOTED": "two",
        "SINGLE": "three",
        "EXPORTED": "four",
        "SPACED": "five",
    }
    assert "NOEQUALS" not in loaded


def test_parse_dotenv_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert parse_dotenv(tmp_path / "nope.env") == {}


def test_parse_dotenv_does_not_touch_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: parsing used to write into os.environ, so one call in a
    test session leaked a developer's real .env into every later test."""
    monkeypatch.delenv("DF_PROBE_VAR", raising=False)
    (tmp_path / ".env").write_text("DF_PROBE_VAR=leaked", encoding="utf-8")
    parse_dotenv(tmp_path / ".env")
    assert os.environ.get("DF_PROBE_VAR") is None


def test_real_environment_wins_over_the_dotenv_file(tmp_path: Path) -> None:
    """A CI-injected secret must never be shadowed by a stale local file."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("PATH=from-file\nONLY_IN_FILE=yes", encoding="utf-8")
    merged = resolve_env(tmp_path)
    assert merged["PATH"] == os.environ["PATH"]
    assert merged["ONLY_IN_FILE"] == "yes"


def test_an_injected_environment_is_used_verbatim(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SHOULD_NOT_APPEAR=1", encoding="utf-8")
    assert resolve_env(tmp_path, {"ONLY": "this"}) == {"ONLY": "this"}


def test_settings_report_missing_vars_from_the_injected_environment(project: Path) -> None:
    """doctor's environment check reads settings.missing_required. It used to
    call check_env() with no argument and silently consult os.environ."""
    assert get_settings({}, root=project).missing_required == REQUIRED_VARS
    filled = dict.fromkeys(REQUIRED_VARS, "x")
    assert get_settings(filled, root=project).missing_required == ()


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
