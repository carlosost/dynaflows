"""LangSmith wiring. ADR-011.

Tracing is configured before any provider call exists, not bolted on later:
the LangChain/LangGraph integrations read these variables at call time, so if
they are set wrong the traces are simply absent, with no error -- the same
silent-failure shape as AP-05.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dynaflows.settings import Settings


def configure_tracing(settings: Settings) -> bool:
    """Export the variables the LangChain integrations read. Returns enabled state."""
    enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    return enabled


@dataclass(frozen=True, slots=True)
class TelemetryStatus:
    reachable: bool
    detail: str


def check_langsmith(settings: Settings) -> TelemetryStatus:
    """One authenticated round trip to LangSmith. Never raises."""
    if not settings.langsmith_api_key:
        return TelemetryStatus(False, "LANGSMITH_API_KEY is not set")
    try:
        from langsmith import Client

        client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
        client.info  # noqa: B018 -- property performs the request
        return TelemetryStatus(True, f"project '{settings.langsmith_project}'")
    except Exception as exc:  # noqa: BLE001 -- a doctor check reports, never propagates
        return TelemetryStatus(False, f"{type(exc).__name__}: {exc}")
