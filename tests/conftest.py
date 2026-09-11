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
        # Overridden by tests that need a worker to fail, come back short, or
        # admit it lacked context. Default is a clean success.
        self.worker_report: Any = _default_report

    async def call(self, request: Any, **_: object) -> Any:
        """Answers according to the schema it was asked for.

        One fake for the whole graph: a second one would be the AP-11 shape in
        test code, where a fix to one silently misses the other.
        """
        from dynaflows.contracts.calls import CallResult
        from dynaflows.graph.prompts import EnhancedPrompt, PlanDraft, WorkerReport

        self.requests.append(request)
        if request.schema is PlanDraft:
            payload: Any = self.plan_draft() if callable(self.plan_draft) else self.plan_draft
        elif request.schema is WorkerReport:
            payload = self.worker_report(request)
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

    @property
    def worker_calls(self) -> int:
        from dynaflows.graph.prompts import WorkerReport

        return sum(1 for r in self.requests if r.schema is WorkerReport)


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


def _default_report(request: Any) -> Any:
    """A clean worker answer, identifiable per task.

    The task id is echoed into the findings so a fan-out test can prove the N
    artifacts are N DIFFERENT artifacts rather than one written N times.
    """
    from dynaflows.graph.prompts import WorkerReport

    task_id = dict(request.metadata).get("task_id", "?")
    return WorkerReport(
        summary=f"findings for {task_id}",
        findings=f"# {task_id}\n\nEvidence.",
        context_was_sufficient=True,
    )


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


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A source root for ADR-017's input resolution, with one readable file."""
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("def login():\n    return True\n", encoding="utf-8")
    return root


def graph_config(thread: str, gateway: Any, root: Path, **extra: Any) -> dict[str, Any]:
    """The `configurable` every graph test passes.

    Shared rather than copied. It was copied -- once per test module -- and the
    moment the worker node needed two more dependencies, every copy had to
    learn about them independently. That is the AP-11 shape in test code.
    """
    from dynaflows.store.run_store import RunStore

    return {
        "configurable": {
            "thread_id": thread,
            "gateway": gateway,
            "playbook": make_playbook(),
            "store": RunStore(root / ".dynaflows"),
            "source_root": root,
            # No default. Which gate a module skips is the thing that module
            # is about -- g1 skips plan, g2 skips prompt -- and picking one
            # here made every g1 test sail past its own subject.
            "auto_approve": list(extra.pop("auto_approve", []) or []),
            **extra,
        }
    }


def cfg_factory(workspace: Path, skip: list[str]) -> Any:
    """A `cfg(thread, gateway, **extra)` that skips `skip` by default.

    The shared part -- gateway, playbook, store, source root -- lives in
    graph_config; the gate default does not, because it is what distinguishes
    the modules. Merged rather than overridden, so a test adding a gate does
    not silently lose the module's own.
    """

    def _cfg(thread: str, gateway: Any, **extra: Any) -> dict[str, Any]:
        gates = [*skip, *(extra.pop("auto_approve", []) or [])]
        return graph_config(thread, gateway, workspace, auto_approve=gates, **extra)

    return _cfg
