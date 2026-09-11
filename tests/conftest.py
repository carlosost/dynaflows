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
        self.plan_draft: Any = _default_draft

    async def call(self, request: Any, **_: object) -> Any:
        """Answers according to the schema it was asked for.

        One fake for the whole graph: a second one would be the AP-11 shape in
        test code, where a fix to one silently misses the other.
        """
        from dynaflows.contracts.calls import CallResult
        from dynaflows.graph.prompts import EnhancedPrompt, PlanDraft

        self.requests.append(request)
        if request.schema is PlanDraft:
            payload: Any = self.plan_draft() if callable(self.plan_draft) else self.plan_draft
        else:
            payload = EnhancedPrompt(
                enhanced=request.prompt if self.echo else self.enhanced,
                assumptions=self.assumptions,
            )
        return CallResult(
            payload=payload,
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

    @property
    def enhancer_calls(self) -> int:
        """Counted by ROLE, not in total.

        Several tests assert the enhancer ran exactly once across a resume
        (ADR-007). Once the planner also used the gateway, a total count
        stopped answering that question -- and a test whose number changes for
        an unrelated reason is a test people start editing instead of reading.
        """
        from dynaflows.graph.prompts import EnhancedPrompt

        return sum(1 for r in self.requests if r.schema is EnhancedPrompt)

    @property
    def planner_calls(self) -> int:
        from dynaflows.graph.prompts import PlanDraft

        return sum(1 for r in self.requests if r.schema is PlanDraft)


@pytest.fixture
def fake_gateway() -> FakeGateway:
    return FakeGateway()


def draft_with(count: int) -> Any:
    """A plan of exactly `count` tasks, for tests about fan-out width."""
    from dynaflows.graph.prompts import PlanDraft, PlannedTask

    return PlanDraft(
        rationale="split by surface area",
        tasks=[
            PlannedTask(
                task_id=f"task-{i}",
                capability="analyse",
                objective=f"objective {i}",
                playbook_anchors=["AP-11"],
            )
            for i in range(1, count + 1)
        ],
    )


def _default_draft() -> Any:

    return draft_with(3)


@pytest.fixture
def fake_playbook() -> Any:
    return make_playbook()


def make_playbook() -> Any:
    """A two-section corpus. Real repository class, no filesystem."""
    from dynaflows.contracts.playbook import Chunk
    from dynaflows.playbook.repository import InMemoryPlaybookRepository

    return InMemoryPlaybookRepository(
        [
            Chunk(
                id="c1",
                source_path="p.md",
                heading_path="Playbook > AP-11: Parallel Abstraction",
                anchors=("AP-11",),
                body="Coverage looks healthy but the entry point calls another implementation.",
                tokens=12,
            ),
            Chunk(
                id="c2",
                source_path="p.md",
                heading_path="Playbook > 3.3 Service Integration",
                anchors=("§3.3",),
                body="Idempotency keys on every mutating operation.",
                tokens=8,
            ),
        ]
    )
