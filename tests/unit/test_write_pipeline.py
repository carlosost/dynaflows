"""The WRITE pipeline's nodes, against a real git repo and a fake agent.

Real git because every claim about the worktree is a claim about git's
behaviour. A fake agent because the real one costs a subscription turn per
test and its output shape is already covered by `test_agent.py`.

The focus here is the ORDER and the failure decisions -- which is where the
dangerous bugs are. An `execute` that runs after a failed `prepare` edits the
USER'S tree, and that is the one thing this pipeline exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from dynaflows.contracts.state import GateDecision, WorkflowState
from dynaflows.executor.agent import AgentRun
from dynaflows.executor import workspace as ws
from dynaflows.graph import builder, write_nodes
from dynaflows.store.run_store import RunStore

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "first")
    return root


class FakeAgent:
    """Records what it was handed; edits the worktree if told to."""

    def __init__(self, *, edit: str | None = None, run: AgentRun | None = None) -> None:
        self.edit = edit
        self.brief: str | None = None
        self.cwd: Path | None = None
        self.context: str = ""
        self._run = run

    def run(self, workspace: Any, brief: str, *, context: str = "") -> AgentRun:
        self.brief = brief
        self.cwd = workspace.path
        self.context = context
        if self.edit is not None:
            (workspace.path / "module.py").write_text(self.edit, encoding="utf-8")
        return self._run or AgentRun(
            session_id="s1",
            exit_code=0,
            timed_out=False,
            duration_s=1.0,
            raw="",
            parsed=True,
            result_text="edited module.py",
            cost_usd=0.2,
            num_turns=3,
        )


class FakeOracle:
    """Stands in for `uv sync` and the test suite, which take minutes."""

    def __init__(self, *, sync_code: int = 0, before: list[str], after: list[str]) -> None:
        self.sync_code = sync_code
        self.before = before
        self.after = after
        self.runs = 0

    def prepare(self, space: Any) -> Any:
        from dynaflows.contracts.state import SuiteRun

        return SuiteRun(command=["uv", "sync"], exit_code=self.sync_code, tail="sync output")

    def run_tests(self, space: Any, *args: Any, **kwargs: Any) -> Any:
        from dynaflows.contracts.state import SuiteRun

        failing = self.before if self.runs == 0 else self.after
        self.runs += 1
        return SuiteRun(
            command=["pytest"],
            exit_code=1 if failing else 0,
            failing=sorted(failing),
            parsed=True,
            tail="suite output",
        )

    @staticmethod
    def compare(before: Any, after: Any) -> Any:
        from dynaflows.executor.verify import compare

        return compare(before, after)


def _config(repo: Path, agent: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "source_root": str(repo),
            "home": str(repo / ".dynaflows"),
            "agent": agent,
            "store": RunStore(repo / ".dynaflows"),
        }
    }


def _state(**over: Any) -> WorkflowState:
    base: dict[str, Any] = {
        "run_id": "r1",
        "thread_id": "t1",
        "raw_prompt": "fix it",
        "enhanced_prompt": "THE APPROVED BRIEF",
    }
    base.update(over)
    return base  # type: ignore[return-value]


@pytest.fixture
def oracle(monkeypatch: pytest.MonkeyPatch) -> FakeOracle:
    fake = FakeOracle(before=[], after=[])
    monkeypatch.setattr(write_nodes, "oracle", fake)
    return fake


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


async def test_prepare_cuts_a_worktree_and_takes_a_baseline(
    repo: Path, oracle: FakeOracle
) -> None:
    oracle.before = ["tests/x.py::already_red"]
    update = await write_nodes.prepare(_state(), _config(repo, FakeAgent()))

    assert update["workspace"].branch == "dynaflows/r1"
    assert update["baseline"].failing == ["tests/x.py::already_red"]
    assert Path(update["workspace"].path).is_dir()


async def test_prepare_halts_when_the_environment_cannot_be_built(
    repo: Path, oracle: FakeOracle
) -> None:
    """A worktree that cannot run its tests cannot verify anything, and a
    change handed back unverified by a tool whose premise is 'and then it runs
    your tests' is worse than no change."""
    oracle.sync_code = 1
    update = await write_nodes.prepare(_state(), _config(repo, FakeAgent()))

    assert "could not be built" in update["halted"]
    assert "workspace" not in update
    assert not (repo / ".dynaflows" / "work" / "r1").exists(), "the worktree must be cleaned up"


async def test_prepare_halts_rather_than_raising_outside_a_repository(
    tmp_path: Path, oracle: FakeOracle
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    update = await write_nodes.prepare(_state(), _config(plain, FakeAgent()))
    assert "halted" in update


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------


async def test_execute_hands_the_agent_the_approved_brief_in_the_worktree(
    repo: Path, oracle: FakeOracle
) -> None:
    """Not the raw prompt and not the plan objective: the human approved one
    string at G1, and that is the string the agent gets."""
    agent = FakeAgent(edit="VALUE = 2\n")
    config = _config(repo, agent)
    state = _state(**await write_nodes.prepare(_state(), config))

    update = await write_nodes.execute(state, config)

    assert agent.brief == "THE APPROVED BRIEF"
    assert agent.cwd == Path(state["workspace"].path)
    assert update["changes"].files == ["module.py"]
    assert update["changes"].captured is True
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n", (
        "the user's own tree must be untouched"
    )


async def test_the_patch_goes_to_the_run_store_not_into_state(
    repo: Path, oracle: FakeOracle
) -> None:
    """ADR-008. A diff can be megabytes and state is re-serialised every
    superstep."""
    config = _config(repo, FakeAgent(edit="VALUE = 2\n"))
    state = _state(**await write_nodes.prepare(_state(), config))

    update = await write_nodes.execute(state, config)
    reference = update["changes"].patch

    assert reference is not None
    assert "VALUE = 2" in Path(reference.path).read_text(encoding="utf-8")


async def test_an_agent_that_changed_nothing_is_captured_as_empty(
    repo: Path, oracle: FakeOracle
) -> None:
    """Measured live: `acceptEdits` exits 0 having silently refused a command.
    `usable` is true and the diff is empty, and only the diff catches it."""
    config = _config(repo, FakeAgent(edit=None))
    state = _state(**await write_nodes.prepare(_state(), config))

    update = await write_nodes.execute(state, config)

    assert update["agent"].usable is True
    assert update["changes"].is_empty is True
    assert update["changes"].captured is True


async def test_an_unavailable_agent_halts_and_says_so(repo: Path, oracle: FakeOracle) -> None:
    unavailable = AgentRun(
        session_id="s1",
        exit_code=1,
        timed_out=False,
        duration_s=0.1,
        raw="",
        parsed=True,
        result_text="Not logged in · Please run /login",
    )
    config = _config(repo, FakeAgent(run=unavailable))
    state = _state(**await write_nodes.prepare(_state(), config))

    update = await write_nodes.execute(state, config)

    assert update["agent"].unavailable is True
    assert "could not be used" in update["halted"]
    assert "Not logged in" in update["halted"]


async def test_the_agents_cost_never_reaches_the_dollar_ledger(
    repo: Path, oracle: FakeOracle
) -> None:
    """Two currencies in one number is AP-20 in a cost line: under
    subscription auth this figure is an estimate and the run spent quota."""
    config = _config(repo, FakeAgent(edit="VALUE = 2\n"))
    state = _state(**await write_nodes.prepare(_state(), config))

    update = await write_nodes.execute(state, config)

    assert update["agent"].cost_usd_equivalent == 0.2
    assert "cost" not in update


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


async def test_verify_compares_against_the_baseline_not_against_zero(
    repo: Path, oracle: FakeOracle
) -> None:
    oracle.before = ["tests/x.py::already_red"]
    oracle.after = ["tests/x.py::already_red", "tests/y.py::broken_by_the_change"]

    config = _config(repo, FakeAgent(edit="VALUE = 2\n"))
    state = _state(**await write_nodes.prepare(_state(), config))
    state.update(await write_nodes.execute(state, config))

    update = await write_nodes.verify(state, config)
    verdict = update["verdict"]

    assert verdict.newly_failing == ["tests/y.py::broken_by_the_change"]
    assert verdict.still_failing == ["tests/x.py::already_red"]
    assert verdict.clean is False


async def test_an_already_red_suite_does_not_block_a_clean_change(
    repo: Path, oracle: FakeOracle
) -> None:
    oracle.before = ["tests/x.py::already_red"]
    oracle.after = ["tests/x.py::already_red"]

    config = _config(repo, FakeAgent(edit="VALUE = 2\n"))
    state = _state(**await write_nodes.prepare(_state(), config))
    state.update(await write_nodes.execute(state, config))

    assert (await write_nodes.verify(state, config))["verdict"].clean is True


async def test_verify_is_skipped_when_nothing_changed(repo: Path, oracle: FakeOracle) -> None:
    """A green verdict beside an empty diff reads as success. The outcome the
    user needs is 'it changed nothing', so no verdict is produced at all."""
    config = _config(repo, FakeAgent(edit=None))
    state = _state(**await write_nodes.prepare(_state(), config))
    state.update(await write_nodes.execute(state, config))

    assert await write_nodes.verify(state, config) == {}
    assert oracle.runs == 1, "the suite must not be run a second time for an empty diff"


# --------------------------------------------------------------------------
# G3
# --------------------------------------------------------------------------


async def test_g3_auto_approves_when_told_to(repo: Path, oracle: FakeOracle) -> None:
    config = _config(repo, FakeAgent(edit="VALUE = 2\n"))
    config["configurable"]["auto_approve"] = ("change",)
    state = _state(**await write_nodes.prepare(_state(), config))
    state.update(await write_nodes.execute(state, config))

    update = await write_nodes.approve_change(state, config)
    assert update["change_gate"].decision is GateDecision.APPROVE


async def test_g3_rejects_when_there_is_no_diff_to_review(repo: Path) -> None:
    update = await write_nodes.approve_change(_state(), _config(repo, FakeAgent()))
    assert update["change_gate"].decision is GateDecision.REJECT
    assert "halted" in update


# --------------------------------------------------------------------------
# The graph's own edges. These are the dangerous ones.
# --------------------------------------------------------------------------


def test_a_halted_prepare_never_reaches_execute() -> None:
    """Without this edge the agent would run with no worktree -- and edit the
    user's own tree, which is the failure this whole pipeline prevents."""
    assert builder.after_prepare({"halted": "no worktree"}) == "__end__"
    assert builder.after_prepare({}) == "execute"


def test_a_halted_execute_never_reaches_verify() -> None:
    assert builder.after_execute({"halted": "agent unavailable"}) == "__end__"
    assert builder.after_execute({}) == "verify"


def test_a_g2_rejection_ends_the_run_before_a_worktree_is_cut() -> None:
    from dynaflows.contracts.state import GateOutcome

    rejected = {"plan_gate": GateOutcome(decision=GateDecision.REJECT)}
    assert builder.after_plan_gate_write(rejected) == "__end__"
    assert builder.after_plan_gate_write({}) == "prepare"


def test_the_write_graph_shares_the_read_pipelines_front_half() -> None:
    """ADR-023 says everything before G2 is one implementation. Asserted
    against the node objects, so a copy-paste would fail here rather than
    drift quietly."""
    from dynaflows.graph import nodes

    assert builder.build_write_graph() is not None
    assert write_nodes.prepare is not nodes.worker
    for name in ("enhance_prompt", "approve_prompt", "plan", "approve_plan"):
        assert getattr(nodes, name) is not None


# --------------------------------------------------------------------------
# G1's paths reach the agent. They used to stop at the gate.
# --------------------------------------------------------------------------


class RecordingAgent(FakeAgent):
    """`FakeAgent` records `context` itself now.

    It did not when these tests were written, and widening the `Agent`
    Protocol broke every implementation of it at once -- caught only because
    a fake exists. Nothing static checks that an implementation still
    satisfies the Protocol, because the agent travels through config as
    `Any`. Worth knowing: the seam's shape is guarded by tests, not by types.
    """

    def __init__(self) -> None:
        super().__init__(edit="VALUE = 2\n")


async def test_the_files_g1_identified_are_given_to_the_agent(
    repo: Path, oracle: FakeOracle
) -> None:
    """Measured on the first live run: the agent got the brief and nothing
    else, so it began file discovery from zero -- which made a `change` run
    close to `claude -p` with a lightly edited prompt."""
    agent = RecordingAgent()
    config = _config(repo, agent)
    state = _state(relevant_paths=["src/a.py", "src/b.py"])
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert "src/a.py" in agent.context
    assert "src/b.py" in agent.context


async def test_the_paths_never_change_the_brief_the_human_approved(
    repo: Path, oracle: FakeOracle
) -> None:
    """They travel as system context. Appending them to the brief would hand
    the agent a string nobody read at G1."""
    agent = RecordingAgent()
    config = _config(repo, agent)
    state = _state(relevant_paths=["src/a.py"])
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert agent.brief == "THE APPROVED BRIEF"
    assert "src/a.py" not in (agent.brief or "")


async def test_the_paths_are_a_starting_point_not_a_boundary(
    repo: Path, oracle: FakeOracle
) -> None:
    """A wrong list must not cripple a capable agent, and a wrong list should
    become a visible outcome rather than a quietly worse change (§7: ADR-018's
    narrowing rule fires on legitimate work)."""
    agent = RecordingAgent()
    config = _config(repo, agent)
    state = _state(relevant_paths=["src/a.py"])
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert "not a boundary" in agent.context
    assert "say so" in agent.context


async def test_no_paths_means_no_context_rather_than_an_empty_heading(
    repo: Path, oracle: FakeOracle
) -> None:
    agent = RecordingAgent()
    config = _config(repo, agent)
    state = _state(relevant_paths=[])
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert agent.context == ""


# --------------------------------------------------------------------------
# The planner must plan a CHANGE. Found on the first live `change` run.
# --------------------------------------------------------------------------


def test_the_write_planner_is_shown_implement_and_not_the_read_capabilities() -> None:
    """The bug: `nodes.plan` called `render_capabilities()` with no argument,
    which defaults to the read catalogue. The write planner could not see
    `implement`, so it planned four `analyse` tasks -- and the WRITE pipeline
    has no worker node to run them."""
    from dynaflows.graph.capabilities import render_capabilities

    assert "implement" in render_capabilities("write")
    assert "analyse" not in render_capabilities("write")


def test_every_caller_of_render_capabilities_names_a_pipeline() -> None:
    """A test on `render_capabilities` proves nothing about its CALLERS.

    The suite already asserted `"implement" not in render_capabilities("read")`
    and passed while `nodes.plan` ignored the parameter entirely, so the write
    planner was shown the read catalogue and planned four analysis tasks.

    Checked by PARSING, not by searching text. The first version of this test
    asserted `"render_capabilities()" not in source` and failed on a COMMENT
    that quoted the bug it was describing. That is the fifth matcher in two
    days to look at the wrong thing -- an anchored grep that could not see an
    indented import, a column parse fed stripped output, a phrase rule that
    could not cross a line wrap, a quoted word that matched any docstring, and
    now this. **The AST knows what a call is; a substring does not.**
    """
    import ast

    root = Path(__file__).resolve().parents[2] / "src"
    bare: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "render_capabilities"
                and not node.args
                and not node.keywords
            ):
                bare.append(f"{path.relative_to(root)}:{node.lineno}")
    assert bare == [], (
        "render_capabilities() without a pipeline defaults to the READ catalogue; "
        f"a WRITE run planned with it cannot see `implement`: {bare}"
    )


def test_a_write_plan_with_several_tasks_is_rejected() -> None:
    from dynaflows.graph.nodes import _write_plan_violation

    class _Task:
        capability = "implement"

    class _Draft:
        tasks = [_Task(), _Task()]

    reason = _write_plan_violation(_Draft())
    assert "ONE task" in reason
    assert "2" in reason


def test_a_write_plan_that_reads_instead_of_writing_is_rejected() -> None:
    from dynaflows.graph.nodes import _write_plan_violation

    class _Task:
        capability = "analyse"

    class _Draft:
        tasks = [_Task()]

    assert "implement" in _write_plan_violation(_Draft())


def test_one_implement_task_is_accepted() -> None:
    from dynaflows.graph.nodes import _write_plan_violation

    class _Task:
        capability = "implement"

    class _Draft:
        tasks = [_Task()]

    assert _write_plan_violation(_Draft()) == ""


# --------------------------------------------------------------------------
# The planner's anchors reach the agent. This is what it is FOR.
# --------------------------------------------------------------------------


class _Chunk:
    def __init__(self, heading_path: str, body: str) -> None:
        self.heading_path = heading_path
        self.body = body


class _Playbook:
    def __init__(self, chunks: list[_Chunk] | None = None) -> None:
        self.chunks = chunks or []
        self.asked: list[str] = []

    def by_anchor(self, anchors: list[str]) -> list[_Chunk]:
        self.asked = list(anchors)
        return self.chunks


def _planned(anchors: list[str]) -> Any:
    from dynaflows.contracts.state import Plan, PlanTask

    return Plan(
        tasks=[
            PlanTask(
                task_id="t", capability="implement", objective="o", playbook_anchors=anchors
            )
        ]
    )


async def test_the_playbook_sections_the_planner_chose_reach_the_agent(
    repo: Path, oracle: FakeOracle
) -> None:
    """The planner's one irreplaceable contribution in the WRITE pipeline.

    The agent reads files better than any plan can describe them, but it
    cannot know which of this project's decisions govern the change it is
    about to make -- that is not discoverable from the code.
    """
    agent = RecordingAgent()
    config = _config(repo, agent)
    config["configurable"]["playbook"] = _Playbook(
        [_Chunk("Playbook > AP-20", "two facts, two counters")]
    )
    state = _state(plan=_planned(["AP-20"]))
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert "two facts, two counters" in agent.context
    assert "AP-20" in agent.context


async def test_a_broken_playbook_costs_context_not_the_change(
    repo: Path, oracle: FakeOracle
) -> None:
    """Degraded, not dead: the agent loses context it would have liked and
    still makes the change."""

    class _Broken:
        def by_anchor(self, anchors: list[str]) -> list[_Chunk]:
            raise RuntimeError("index unavailable")

    agent = RecordingAgent()
    config = _config(repo, agent)
    config["configurable"]["playbook"] = _Broken()
    state = _state(plan=_planned(["AP-20"]), relevant_paths=["src/a.py"])
    state.update(await write_nodes.prepare(state, config))

    update = await write_nodes.execute(state, config)

    assert "halted" not in update
    assert update["changes"].files == ["module.py"]
    assert "src/a.py" in agent.context


async def test_no_anchors_means_no_playbook_section_in_the_context(
    repo: Path, oracle: FakeOracle
) -> None:
    agent = RecordingAgent()
    config = _config(repo, agent)
    config["configurable"]["playbook"] = _Playbook([])
    state = _state(plan=_planned([]))
    state.update(await write_nodes.prepare(state, config))

    await write_nodes.execute(state, config)

    assert "govern the change" not in agent.context


async def test_an_agent_that_never_started_leaves_no_worktree_behind(
    repo: Path, oracle: FakeOracle
) -> None:
    """An `execute` halt normally KEEPS the worktree -- an agent that ran and
    failed may have left work worth looking at. One that never started cannot
    have, and two such runs left empty worktrees and branches on disk in
    September with nothing to clean them up."""
    unavailable = AgentRun(
        session_id="s1", exit_code=1, timed_out=False, duration_s=0.1, raw="",
        parsed=True, result_text="Failed to authenticate: OAuth session expired",
        fields={"usage": {"input_tokens": 0}, "modelUsage": {}},
    )
    config = _config(repo, FakeAgent(run=unavailable))
    state = _state(**await write_nodes.prepare(_state(), config))
    path = Path(state["workspace"].path)
    assert path.is_dir()

    update = await write_nodes.execute(state, config)

    assert "halted" in update
    assert not path.exists(), "an agent that never ran leaves nothing worth keeping"
    assert "dynaflows/r1" not in ws.git(repo, "branch", "--list")


async def test_an_agent_that_ran_and_failed_keeps_its_worktree(
    repo: Path, oracle: FakeOracle
) -> None:
    """The other half of the rule: work that exists is not thrown away."""
    failed = AgentRun(
        session_id="s1", exit_code=1, timed_out=False, duration_s=0.1, raw="",
        parsed=True, result_text="hit the spend limit",
        fields={"usage": {"input_tokens": 5000, "output_tokens": 200}},
    )
    config = _config(repo, FakeAgent(edit="VALUE = 2\n", run=failed))
    state = _state(**await write_nodes.prepare(_state(), config))
    path = Path(state["workspace"].path)

    await write_nodes.execute(state, config)

    assert path.is_dir(), "an agent that produced work keeps its worktree"
