"""The WRITE pipeline's three nodes. ADR-023, ADR-025.

    G2 approved -> prepare -> execute -> verify -> G3

Each node does one thing and records what it found, including when what it
found is "I could not tell". The nodes are deliberately thin: the worktree,
the agent and the test oracle are all in `executor/`, tested on their own
against real git and real processes. What lives here is the ORDER, the
failure decisions, and the translation into state.

**Why these are not `Send` branches.** The READ pipeline fans out because N
readers are independent. There is one executor here, and the three steps are
strictly sequential -- you cannot compare a suite run against a baseline you
have not taken. So they are plain edges, and a node MAY raise: ADR-010's
never-raise rule exists because an exception inside a Send branch aborts the
whole superstep, and there is no superstep to abort. They still do not raise,
for a different reason: a halt carries a message to the gate, and an exception
carries a traceback to the terminal.

**The one opinionated decision here is that `prepare` halts when the worktree
cannot be built.** A run whose environment failed to sync can still make a
change; it just cannot verify one. Handing back an unverified diff from a tool
whose premise is "and then it runs your tests" is worse than stopping, because
the user has no way to see which of the two they got. Reversal condition: a
user who wants the change without the check -- at which point it is a flag on
`change`, named for what it removes.
"""

# NOT `from __future__ import annotations`, for the same reason as `nodes.py`,
# and the warning was live before this comment was written: LangGraph inspects
# node signatures at add_node() time, and postponed annotations leave it
# comparing the STRING "RunnableConfig | None" against a type it cannot
# resolve, so it warns once per node on every run. The warning it emits names
# the annotation it wanted and the annotation it got, and they are identical
# text -- which is why it is easy to read as noise and ignore. A warning that
# fires on correct work is a warning people learn to scroll past (§5.1).
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.state import (
    AgentOutcome,
    ChangeSet,
    SuiteRun,
    WorkflowState,
    WorkspaceRef,
)
from dynaflows.executor import verify as oracle
from dynaflows.executor import workspace as ws
from langgraph.types import interrupt

from dynaflows.contracts.state import GateDecision, GateOutcome
from dynaflows.graph.gates import gate_outcome
from dynaflows.graph.deps import (
    agent_from,
    auto_approved,
    home_from,
    playbook_from,
    source_root_from,
    store_from,
)

__all__ = ["approve_change", "execute", "prepare", "verify"]

# The task id every write artifact is filed under. One task per plan (OQ-02),
# so this is a constant rather than a field -- and a constant that is wrong
# the day OQ-02 reverses, which is where it will be noticed.
TASK_ID = "implement"


def _ref(space: ws.Workspace) -> WorkspaceRef:
    return WorkspaceRef(
        path=str(space.path),
        branch=space.branch,
        base=space.base,
        dirty_paths=list(space.dirty_paths),
        dirty_checked=space.dirty_checked,
    )


def _governing_rules(state: WorkflowState, config: RunnableConfig | None) -> str:
    """The playbook sections the planner said govern this change. ADR-009.

    This is the planner earning its cost in the WRITE pipeline. The coding
    agent reads files better than any plan can describe them, so naming its
    steps is worthless -- but it starts with no idea which of this project's
    decisions bear on the change it is about to make, and it cannot find that
    out by reading code. Routing retrieval is the one thing the planner does
    that the agent cannot do for itself.

    Failure here is not fatal. An anchor the planner named that no longer
    exists, or a repository that cannot be reached, costs the agent context
    it would have liked -- it does not cost it the change. Returning "" and
    proceeding is right; raising would turn a degraded run into no run.
    """
    plan = state.get("plan")
    if plan is None or not plan.tasks:
        return ""
    anchors = list(plan.tasks[0].playbook_anchors)
    if not anchors:
        return ""
    try:
        chunks = playbook_from(config).by_anchor(anchors)
    except Exception:  # noqa: BLE001 - see the docstring: context is optional
        return ""
    if not chunks:
        return ""
    sections = "\n\n".join(
        f"### {chunk.heading_path}\n{chunk.body}" for chunk in chunks
    )
    return (
        "These sections of this project's engineering playbook govern the change "
        "you are about to make. They were selected for this specific request. "
        "Follow them; where one conflicts with the brief, say so rather than "
        "silently picking one.\n\n"
        f"{sections}"
    )


def _context(state: WorkflowState) -> str:
    """What the run knows that the brief does not say.

    G1 produces the files it believes the request concerns, checked against
    the source catalogue (ADR-023). Until now that list stopped at the gate:
    the agent was handed the brief alone and began file discovery from
    nothing, which made a `change` run close to `claude -p` with a lightly
    edited prompt -- and the paths are the most useful thing G1 produces.

    It travels as system context rather than appended to the brief, because
    the brief is the exact string a human approved and editing it would hand
    the agent something nobody read.

    Framed as a STARTING POINT and not a boundary, deliberately. A wrong
    path list would otherwise cripple a capable agent, and this project has
    already recorded that narrowing rules fire on legitimate work
    (ADR-018's empty-inputs rule, §7). The agent is asked to say so when the
    list is wrong, which turns a bad enhancer into a visible outcome instead
    of a quietly worse change.
    """
    paths = [p for p in (state.get("relevant_paths") or []) if p]
    if not paths:
        return ""
    listed = "\n".join(f"  - {path}" for path in paths)
    return (
        "Before you were invoked, the request was read against this repository "
        "and these files were identified as the ones it concerns:\n"
        f"{listed}\n"
        "Treat this as a starting point, not a boundary: read whatever else the "
        "change requires. If one of these turns out to be irrelevant, or the list "
        "misses the real subject, say so explicitly in your final message."
    )


def _attached(state: WorkflowState, config: RunnableConfig | None) -> ws.Workspace:
    """Re-attach to the worktree this run already cut.

    Reads the base sha from state rather than from HEAD, because the run may
    have been paused at a gate while the user committed (ADR-025). A diff
    against a moved HEAD would attribute the user's commits to the agent.
    """
    reference = state.get("workspace")
    if reference is None:
        raise DynaflowsError.of(
            ErrorCode.CONFIG_INVALID,
            "no workspace in state; `prepare` did not run or did not finish",
        )
    return ws.adopt(
        Path(source_root_from(config)),
        Path(home_from(config)),
        state["run_id"],
        reference.base,
    )


async def prepare(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Cut the worktree, build its environment, and take the baseline.

    The baseline is the half people skip. Without it the verdict can only say
    "three tests fail", which is the same sentence for a change that broke
    three and for a project that was already red -- and only one of those is
    the user's problem (ADR-025).
    """
    try:
        root = Path(source_root_from(config))
        home = Path(home_from(config))
        space = ws.create(root, home, state["run_id"])
    except DynaflowsError as exc:
        return {"halted": str(exc.envelope)}

    synced = oracle.prepare(space)
    if synced.exit_code != 0:
        ws.discard(space)
        return {
            "halted": (
                "the worktree's environment could not be built, so no test run from it "
                f"would mean anything: {synced.tail[-400:] or 'no output'}"
            )
        }

    baseline = oracle.run_tests(space)
    return {"workspace": _ref(space), "baseline": baseline}


async def execute(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Hand the approved brief to the coding agent, then read the diff.

    The brief is `enhanced_prompt` -- the text the user approved at G1 and
    nothing else. Not the raw prompt, and not the plan's objective: the human
    approved one string, and that is the string the agent gets.
    """
    try:
        space = _attached(state, config)
        agent = agent_from(config)
        store = store_from(config)
    except DynaflowsError as exc:
        return {"halted": str(exc.envelope)}

    brief = (state.get("enhanced_prompt") or state.get("raw_prompt") or "").strip()
    if not brief:
        return {"halted": "no approved brief to hand the agent"}

    # Two kinds of context, joined, and both optional. The paths say where to
    # look; the playbook sections say what the change must not break.
    context = "\n\n".join(
        part for part in (_context(state), _governing_rules(state, config)) if part
    )
    result = agent.run(space, brief, context=context)
    outcome = AgentOutcome(
        session_id=result.session_id,
        exit_code=result.exit_code,
        usable=result.usable,
        unavailable=result.unavailable,
        timed_out=result.timed_out,
        duration_s=result.duration_s,
        cost_usd_equivalent=result.cost_usd,
        num_turns=result.num_turns,
        result_text=result.result_text[:2000],
    )

    if result.unavailable:
        # A configuration problem, not a failed change (ADR-025 amendment).
        # Telling the user their change failed would send them to read a diff
        # that was never produced.
        return {
            "agent": outcome,
            "halted": f"the coding agent could not be used: {result.result_text or result.raw[-300:]}",
        }

    changes = space.capture()
    patch = (
        store.write(state["run_id"], TASK_ID, changes.patch, kind="diff")
        if changes.patch
        else None
    )
    return {
        "agent": outcome,
        "changes": ChangeSet(
            files=changes.files,
            stat=changes.stat[:4000],
            patch=patch,
            captured=True,
        ),
    }


async def verify(state: WorkflowState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Run the suite again and compare it against the baseline.

    Skipped entirely when the agent changed nothing. Re-running a suite to
    prove that an empty diff breaks nothing costs minutes and produces a
    `clean` verdict, which is true and misleading: the outcome the user needs
    is "it changed nothing", and a green verdict next to an empty diff reads
    as success (AP-20). The gate reads `changes.is_empty` first.
    """
    changes = state.get("changes")
    if changes is None or changes.is_empty:
        return {}

    baseline = state.get("baseline")
    if baseline is None:
        return {"after": None, "verdict": None}

    try:
        space = _attached(state, config)
    except DynaflowsError as exc:
        return {"halted": str(exc.envelope)}

    after: SuiteRun = oracle.run_tests(space)
    return {"after": after, "verdict": oracle.compare(baseline, after)}


async def approve_change(
    state: WorkflowState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Gate G3 -- the diff, the verdict, and what the agent says it did.

    PURE, like G1 and G2: it reads state and interrupts. Everything it shows
    was measured by an earlier node, so re-running the gate after a resume
    cannot produce a different picture than the one the user was shown.

    The payload leads with the three facts that are easy to conflate, in the
    order that stops them being conflated:

      1. `empty` -- the agent reported success and changed nothing. Measured
         live: `acceptEdits` exits 0 having silently refused a command. This
         is FIRST because every other field below reads as success when it is
         true, and the verdict would say `clean`.
      2. `comparable` -- whether the suite comparison is an answer at all.
         An empty `newly_failing` from a run that collected no tests looks
         exactly like a clean change.
      3. `newly_failing` / `newly_passing` / `still_failing` -- three facts,
         three fields. "Three failures" before and after can be disjoint sets.

    `dirty_paths` is here because this is the last moment it can still matter:
    the worktree was cut from HEAD, so a file the user had edited but not
    committed was invisible to the agent, and applying this diff over it is
    where that collides (ADR-025, ADR-026).
    """
    changes = state.get("changes")
    if changes is None:
        return {
            "change_gate": GateOutcome(
                decision=GateDecision.REJECT, note="no diff was captured"
            ),
            "halted": "nothing to review at G3",
        }

    if auto_approved(config, "change"):
        return {"change_gate": GateOutcome(decision=GateDecision.APPROVE, note="--yes-change")}

    agent = state.get("agent")
    verdict = state.get("verdict")
    after = state.get("after")
    baseline = state.get("baseline")
    reference = state.get("workspace")

    answer = interrupt(
        {
            "gate": "change",
            "empty": changes.is_empty,
            "files": list(changes.files),
            "stat": changes.stat,
            "patch": None if changes.patch is None else changes.patch.path,
            "branch": None if reference is None else reference.branch,
            "dirty_paths": [] if reference is None else list(reference.dirty_paths),
            "dirty_checked": True if reference is None else reference.dirty_checked,
            "agent_said": "" if agent is None else agent.result_text,
            "agent_turns": None if agent is None else agent.num_turns,
            # Named for what it is. NOT added to the run's dollar ledger:
            # under subscription auth this is an estimate of equivalent API
            # spend, and the run consumed quota instead. Two currencies in one
            # number is AP-20 in a cost line.
            "agent_cost_equivalent": None if agent is None else agent.cost_usd_equivalent,
            "comparable": False if verdict is None else verdict.comparable,
            "newly_failing": [] if verdict is None else list(verdict.newly_failing),
            "newly_passing": [] if verdict is None else list(verdict.newly_passing),
            "still_failing": [] if verdict is None else list(verdict.still_failing),
            "baseline_failing": [] if baseline is None else list(baseline.failing),
            "suite_tail": "" if after is None else after.tail,
        }
    )
    outcome = gate_outcome(answer)
    update: dict[str, Any] = {"change_gate": outcome}
    if outcome.decision is GateDecision.REJECT:
        update["halted"] = "rejected by human at gate G3"
    return update
