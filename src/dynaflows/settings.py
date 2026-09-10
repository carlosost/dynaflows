"""Environment configuration and the startup check.

Playbook AP-05: an SDK renames an environment variable, the old name is
silently ignored, and a working feature stops working with no error at all.
The fix is a startup check that refuses to run with required vars missing --
`check_env()` below, surfaced by `dynaflows doctor`.

Every numeric default here is a PLACEHOLDER, not a measurement (playbook 4.5).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REQUIRED_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "LANGSMITH_API_KEY",
)


def find_project_root(start: Path | None = None) -> Path:
    """Walk up to the directory holding pyproject.toml.

    AP-13: the production launcher does not share the test runner's working
    directory. Resolve paths from a marker file, never from os.getcwd().
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found above {current}")


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict.

    PURE: returns values, mutates nothing. An earlier version wrote straight
    into os.environ, which made `get_settings()` a function with a hidden
    global side effect -- one call anywhere in a test session leaked a
    developer's real .env into every later test in the process. Composition
    now happens in `resolve_env`, where it is visible.
    """
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            loaded[key] = value
    return loaded


def resolve_env(project_root: Path, environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """The environment this process reads, as one explicit mapping.

    The real environment wins over the .env file, so a CI-injected secret is
    never shadowed by a stale local file.
    """
    if environ is not None:
        return environ
    return {**parse_dotenv(project_root / ".env"), **os.environ}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    openrouter_api_key: str | None
    openrouter_base_url: str
    langsmith_api_key: str | None
    langsmith_tracing: bool
    langsmith_project: str
    langsmith_endpoint: str
    home: Path
    models_config: Path
    max_concurrent: int
    timeout_seconds: float
    missing_required: tuple[str, ...]

    @property
    def state_db(self) -> Path:
        return self.home / "state.db"

    @property
    def playbook_db(self) -> Path:
        return self.home / "playbook.db"

    @property
    def calls_db(self) -> Path:
        """ADR-013.1: separate from state.db, so clearing the cache to force
        fresh calls can never destroy a resumable run."""
        return self.home / "calls.db"

    @property
    def playbook_root(self) -> Path:
        """The runtime retrieval corpus. ADR-009 amendment: docs/ only.

        A single root, not a configurable list. The second root arrives with a
        notes vault, and so does the knob -- adding it now would be a setting
        nothing sets.
        """
        return self.project_root / "docs"


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_settings(environ: Mapping[str, str] | None = None, root: Path | None = None) -> Settings:
    """Sole construction path for configuration (playbook 3.1, factory).

    Tests patch `dynaflows.<module>.get_settings`, never os.environ directly.
    """
    project_root = root or find_project_root()
    env = resolve_env(project_root, environ)

    home_raw = env.get("DYNAFLOWS_HOME", ".dynaflows")
    home = Path(home_raw)
    if not home.is_absolute():
        home = project_root / home

    return Settings(
        project_root=project_root,
        openrouter_api_key=env.get("OPENROUTER_API_KEY") or None,
        openrouter_base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        langsmith_api_key=env.get("LANGSMITH_API_KEY") or None,
        langsmith_tracing=_flag(env, "LANGSMITH_TRACING", True),
        langsmith_project=env.get("LANGSMITH_PROJECT", "dynaflows"),
        langsmith_endpoint=env.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        home=home,
        models_config=project_root / "config" / "models.toml",
        max_concurrent=int(env.get("OPENROUTER_MAX_CONCURRENT", "8")),
        timeout_seconds=float(env.get("OPENROUTER_TIMEOUT_SECONDS", "60")),
        # Computed from the SAME mapping every other field was read from.
        # doctor previously called check_env() with no argument, which fell
        # through to os.environ and ignored the injected environment entirely.
        missing_required=tuple(check_env(env)),
    )


def check_env(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return the names of required vars that are unset. Empty list means OK."""
    env = os.environ if environ is None else environ
    return [name for name in REQUIRED_VARS if not env.get(name)]
