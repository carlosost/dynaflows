"""`dynaflows doctor` -- the Phase 0 exit gate.

Playbook AP-19 habit 3: verify claims about the environment by running them.
A negative result usually has several possible causes and no diagnostic value
on its own, so every check here reports *what it found*, not just pass/fail.

Checks are pure data producers. Rendering lives in cli.py.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from dynaflows.contracts.errors import DynaflowsError
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.registry import context_violations, get_model_registry
from dynaflows.gateway.telemetry import check_langsmith, configure_tracing
from dynaflows.settings import REQUIRED_VARS, Settings, get_settings


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str

    @property
    def blocking(self) -> bool:
        return self.status is Status.FAIL


def check_required_env(settings: Settings) -> Check:
    missing = settings.missing_required
    if missing:
        return Check(
            "environment",
            Status.FAIL,
            f"unset: {', '.join(missing)} -- copy .env.example to .env",
        )
    return Check("environment", Status.OK, f"all {len(REQUIRED_VARS)} required vars set")


def check_state_dir(settings: Settings) -> Check:
    """SQLite writable, and FTS5 present.

    FTS5 is not decoration: ADR-009 chose BM25-over-FTS5 precisely so that
    retrieval needs no new dependency. If this interpreter's sqlite3 lacks it,
    that decision is invalid on this machine and must be known now.
    """
    try:
        settings.home.mkdir(parents=True, exist_ok=True)
        probe_db = settings.home / ".doctor-probe.db"
        connection = sqlite3.connect(probe_db)
        try:
            connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _probe USING fts5(a)")
            connection.execute("INSERT INTO _probe(a) VALUES ('handshake')")
            row = connection.execute(
                "SELECT bm25(_probe) FROM _probe WHERE _probe MATCH 'handshake'"
            ).fetchone()
        finally:
            connection.close()
            probe_db.unlink(missing_ok=True)
    except sqlite3.OperationalError as exc:
        return Check("sqlite", Status.FAIL, f"FTS5/bm25 unavailable -- ADR-009 invalid here: {exc}")
    except OSError as exc:
        return Check("sqlite", Status.FAIL, f"{settings.home} not writable: {exc}")
    if row is None:
        return Check("sqlite", Status.FAIL, "FTS5 table created but bm25() returned nothing")
    return Check(
        "sqlite",
        Status.OK,
        f"{sqlite3.sqlite_version}, FTS5 + bm25 available, {settings.home} writable",
    )


def check_playbook_index(settings: Settings) -> Check:
    """Present, populated, and matching the corpus on disk.

    AP-19 habit 2: a generated artifact plus a drift check makes staleness
    impossible rather than unlikely. An index silently one edit behind is the
    document-that-lies failure in database form -- retrieval would keep
    returning text that is no longer in the playbook.
    """
    from dynaflows.playbook import store
    from dynaflows.playbook.repository import SqlitePlaybookRepository

    if not settings.playbook_root.is_dir():
        return Check("playbook index", Status.FAIL, f"corpus missing: {settings.playbook_root}")
    try:
        connection = store.connect(settings.playbook_db)
    except sqlite3.Error as exc:
        return Check("playbook index", Status.FAIL, str(exc))
    try:
        repository = SqlitePlaybookRepository(connection, settings.playbook_root)
        total = repository.count()
        report = repository.drift()
    finally:
        connection.close()

    if total == 0:
        return Check("playbook index", Status.WARN, "empty -- run `dynaflows index`")
    if not report.clean:
        return Check("playbook index", Status.FAIL, f"{report.summary()} -- run `dynaflows index`")
    return Check("playbook index", Status.OK, f"{total} chunks, matches every source")


def check_models_config(settings: Settings) -> Check:
    try:
        registry = get_model_registry(settings.models_config)
    except DynaflowsError as exc:
        return Check("models.toml", Status.FAIL, str(exc))

    unpopulated = registry.unpopulated
    if unpopulated:
        names = ", ".join(t.value for t in unpopulated)
        return Check(
            "models.toml",
            Status.WARN,
            f"unpopulated tiers: {names} -- run `dynaflows models --suggest` (OQ-01)",
        )
    violation = registry.diversity_violation()
    if violation:
        return Check("models.toml", Status.FAIL, violation)
    return Check("models.toml", Status.OK, f"version {registry.version}, all 4 tiers populated")


def check_langsmith_connectivity(settings: Settings) -> Check:
    enabled = configure_tracing(settings)
    if not enabled:
        return Check("langsmith", Status.FAIL, "tracing disabled or LANGSMITH_API_KEY unset")
    status = check_langsmith(settings)
    return Check(
        "langsmith",
        Status.OK if status.reachable else Status.FAIL,
        status.detail,
    )


def check_openrouter(settings: Settings) -> Check:
    """Auth plus the live structured-output catalogue, in one request."""
    from dynaflows.gateway.probe import catalogue

    try:
        models = catalogue(settings)
    except DynaflowsError as exc:
        return Check("openrouter", Status.FAIL, str(exc))
    return Check(
        "openrouter",
        Status.OK,
        f"authenticated; {len(models)} endpoints support structured outputs",
    )


def check_tier_capability(settings: Settings) -> Check:
    """Every configured model still exists and still supports json_schema.

    ADR-006's capability-homogeneity rule is only true if something checks it.
    This is that something.
    """
    from dynaflows.gateway.probe import catalogue

    try:
        registry = get_model_registry(settings.models_config)
    except DynaflowsError as exc:
        return Check("tier capability", Status.FAIL, str(exc))
    if registry.unpopulated:
        return Check("tier capability", Status.WARN, "skipped -- tiers unpopulated")
    try:
        supported = {m.id for m in catalogue(settings)}
    except DynaflowsError as exc:
        return Check("tier capability", Status.FAIL, str(exc))

    offenders: list[str] = []
    for tier in Tier:
        for model_id in registry.chain(tier).chain:
            if model_id not in supported:
                offenders.append(f"{tier.value}:{model_id}")
    if offenders:
        return Check(
            "tier capability",
            Status.FAIL,
            "no structured-output support: " + ", ".join(offenders),
        )
    return Check("tier capability", Status.OK, "every configured model supports json_schema")


def check_context_homogeneity(settings: Settings) -> Check:
    """A fallback must be able to handle prompts the primary handles.

    ADR-006 requires capability-homogeneous chains, and context length is a
    capability. An 8k fallback behind a 400k primary does not degrade
    gracefully under a rate limit -- it fails on the exact prompt the primary
    was chosen for, which is the failure ADR-006 exists to prevent.

    The floor is absolute and defaults to 0 (disabled), because the right
    number is a measurement nobody has taken yet (playbook 4.5). Disabled is
    reported as WARN, not OK -- a skipped check must not read as a passing one.
    """
    from dynaflows.gateway.probe import catalogue

    try:
        registry = get_model_registry(settings.models_config)
    except DynaflowsError as exc:
        return Check("context homogeneity", Status.FAIL, str(exc))
    if registry.unpopulated:
        return Check("context homogeneity", Status.WARN, "skipped -- tiers unpopulated")
    if registry.min_context_tokens <= 0:
        return Check(
            "context homogeneity",
            Status.WARN,
            "min_context_tokens is 0 (disabled) -- set it in config/models.toml once "
            "you know the largest prompt a tier actually sends (4.5)",
        )
    try:
        context = {m.id: m.context_length for m in catalogue(settings)}
    except DynaflowsError as exc:
        return Check("context homogeneity", Status.FAIL, str(exc))

    offenders = context_violations(registry, context)
    if offenders:
        return Check("context homogeneity", Status.FAIL, "; ".join(offenders))
    return Check(
        "context homogeneity",
        Status.OK,
        f"every configured model holds >= {registry.min_context_tokens // 1000}k context",
    )


def check_handshake(settings: Settings) -> Check:
    """One real, traced, schema-enforced call. The end-to-end proof."""
    from dynaflows.gateway.probe import probe_structured

    try:
        registry = get_model_registry(settings.models_config)
    except DynaflowsError as exc:
        return Check("handshake", Status.FAIL, str(exc))
    if not registry.chain(Tier.SMALL).is_populated:
        return Check("handshake", Status.WARN, "skipped -- small tier unpopulated")
    result = probe_structured(settings, registry.chain(Tier.SMALL).preferred)
    return Check(
        "handshake",
        Status.OK if result.ok else Status.FAIL,
        f"{result.model_id}: {result.detail}",
    )


def run_checks(settings: Settings | None = None, *, offline: bool = False) -> Iterator[Check]:
    """Cheapest and most local first, so a missing key fails before a timeout."""
    settings = settings or get_settings()
    yield check_required_env(settings)
    yield check_state_dir(settings)
    yield check_playbook_index(settings)
    yield check_models_config(settings)
    if offline:
        return
    yield check_langsmith_connectivity(settings)
    yield check_openrouter(settings)
    yield check_tier_capability(settings)
    yield check_context_homogeneity(settings)
    yield check_handshake(settings)
