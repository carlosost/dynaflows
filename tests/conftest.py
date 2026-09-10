"""Shared fixtures.

Playbook 2.2: the deterministic tier touches no network, no provider and no
filesystem outside tmp. Nothing here reads os.environ or the real .env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal project root: just the marker file settings.py looks for."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    return tmp_path


@pytest.fixture
def models_toml(project: Path) -> Path:
    return project / "config" / "models.toml"


class FakeGateway:
    """A gateway that records what it was asked and never touches a network.

    Shared by every graph test. The call COUNT is the assertion in the ADR-007
    tests, so this lives in one place: two copies would let a fix to one drift
    away from the other.
    """

    def __init__(
        self,
        enhanced: str = "a precise brief",
        assumptions: list[str] | None = None,
        *,
        echo: bool = False,
    ):
        # echo=True returns the prompt unchanged. Tests about topology,
        # reducers and resume need the state they put in to be identifiable
        # coming out; a fixed string would make them assert the fake instead.
        self.enhanced = enhanced
        self.echo = echo
        self.assumptions = assumptions or []
        self.requests: list[Any] = []

    async def call(self, request: Any, **_: object) -> Any:
        from dynaflows.contracts.calls import CallResult
        from dynaflows.graph.prompts import EnhancedPrompt

        self.requests.append(request)
        return CallResult(
            payload=EnhancedPrompt(
                enhanced=request.prompt if self.echo else self.enhanced,
                assumptions=self.assumptions,
            ),
            model_id="acme/small",
            tier=request.tier,
            raw="{}",
            tokens_in=40,
            tokens_out=12,
            cost_usd=0.002,
            fallback_depth=0,
            attempts=1,
            cache_hit=False,
        )

    @property
    def calls(self) -> int:
        return len(self.requests)


@pytest.fixture
def fake_gateway() -> FakeGateway:
    return FakeGateway()
