"""Graph assembly. ADR-001: the topology is static and declared here.

Everything the planner "decides" is plan DATA. The only runtime-variable
dimensions are fan-out width (`Send`), path selection (conditional edges) and
iteration count -- and step 1.3 wires the first of those with a width the plan
supplies.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from dynaflows.contracts.state import GateDecision, WorkflowState
from dynaflows.graph import nodes, write_nodes

# Static breakpoints for step 1.3, so resume can be proven before `interrupt()`
# exists. Steps 1.4/1.5 replace these with real gates; the constant is here so
# the change is one line and visible in a diff.
GATE_NODES = ("approve_prompt", "approve_plan")


def after_prompt_gate(state: WorkflowState) -> str:
    """A rejection ends the run before anything is spent.

    ADR-005: G1 is cheap to fail. The whole value of gating here is that a
    "no" costs one small-tier call and nothing else.
    """
    gate = state.get("prompt_gate")
    if gate is not None and gate.decision is GateDecision.REJECT:
        return END
    return "plan"


def after_plan_gate(state: WorkflowState) -> str:
    """A rejection at G2 ends the run before the fan-out spends anything."""
    gate = state.get("plan_gate")
    if gate is not None and gate.decision is GateDecision.REJECT:
        return END
    return "dispatch"


def dispatch_workers(state: WorkflowState) -> list[Send] | str:
    """Fan out one branch per plan task. ADR-001's variable dimension.

    An absent or empty plan goes straight to evaluate rather than dispatching
    nothing: a fan-out of zero would leave the barrier waiting on a superstep
    that never runs.
    """
    plan = state.get("plan")
    if plan is None or not plan.tasks:
        return "evaluate"
    return [Send("worker", {**state, "task": task}) for task in plan.tasks]


def build_graph(checkpointer: Any = None, *, interrupt_before: tuple[str, ...] = ()) -> Any:
    builder = StateGraph(WorkflowState)
    builder.add_node("enhance_prompt", nodes.enhance_prompt)
    builder.add_node("approve_prompt", nodes.approve_prompt)
    builder.add_node("plan", nodes.plan)
    builder.add_node("approve_plan", nodes.approve_plan)
    builder.add_node("worker", nodes.worker)
    builder.add_node("evaluate", nodes.evaluate)
    builder.add_node("synthesize", nodes.synthesize)

    builder.add_edge(START, "enhance_prompt")
    builder.add_edge("enhance_prompt", "approve_prompt")
    builder.add_conditional_edges("approve_prompt", after_prompt_gate, ["plan", END])
    builder.add_edge("plan", "approve_plan")
    # Two hops on purpose: the gate decides whether to proceed at all, and
    # only then does the plan decide the fan-out width. Folding them into one
    # predicate would mix a human decision with a data-shape decision.
    builder.add_node("dispatch", lambda state: {})
    builder.add_conditional_edges("approve_plan", after_plan_gate, ["dispatch", END])
    builder.add_conditional_edges("dispatch", dispatch_workers, ["worker", "evaluate"])
    builder.add_edge("worker", "evaluate")
    builder.add_edge("evaluate", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before),
    )


# ---------------------------------------------------------------------------
# The WRITE pipeline. ADR-023: a separate graph, not a flag on the read one.
# ---------------------------------------------------------------------------

# Both pipelines interrupt at G1 and G2; only this one has a G3.
WRITE_GATE_NODES = ("approve_prompt", "approve_plan", "approve_change")


def after_plan_gate_write(state: WorkflowState) -> str:
    """G2 rejection ends the run before a worktree is cut or an agent spends."""
    gate = state.get("plan_gate")
    if gate is not None and gate.decision is GateDecision.REJECT:
        return END
    return "prepare"


def after_prepare(state: WorkflowState) -> str:
    """A halt in `prepare` must not reach `execute`.

    Without this the run would hand a brief to an agent with no worktree to
    work in, and the agent would edit the USER'S tree -- the one failure this
    whole pipeline is built to prevent. So the check is an edge in the graph
    rather than a guard inside the node: a guard can be forgotten by the next
    node added, an edge cannot.
    """
    return END if state.get("halted") else "execute"


def after_execute(state: WorkflowState) -> str:
    """A halt in `execute` means the agent never ran usefully. Skip the suite.

    Running the tests after an agent that could not be used would take minutes
    to prove that nothing changed, and would produce a verdict about a change
    that does not exist.
    """
    return END if state.get("halted") else "verify"


def build_write_graph(
    checkpointer: Any = None, *, interrupt_before: tuple[str, ...] = ()
) -> Any:
    """G1 -> G2 -> prepare -> execute -> verify -> G3.

    The first four nodes are the SAME objects as the read pipeline's, not
    copies: ADR-023 says everything before G2 is one implementation, and
    sharing the node functions is what makes that true in code rather than in
    a paragraph.
    """
    builder = StateGraph(WorkflowState)
    builder.add_node("enhance_prompt", nodes.enhance_prompt)
    builder.add_node("approve_prompt", nodes.approve_prompt)
    builder.add_node("plan", nodes.plan)
    builder.add_node("approve_plan", nodes.approve_plan)
    builder.add_node("prepare", write_nodes.prepare)
    builder.add_node("execute", write_nodes.execute)
    builder.add_node("verify", write_nodes.verify)
    builder.add_node("approve_change", write_nodes.approve_change)

    builder.add_edge(START, "enhance_prompt")
    builder.add_edge("enhance_prompt", "approve_prompt")
    builder.add_conditional_edges("approve_prompt", after_prompt_gate, ["plan", END])
    builder.add_edge("plan", "approve_plan")
    builder.add_conditional_edges("approve_plan", after_plan_gate_write, ["prepare", END])
    builder.add_conditional_edges("prepare", after_prepare, ["execute", END])
    builder.add_conditional_edges("execute", after_execute, ["verify", END])
    builder.add_edge("verify", "approve_change")
    builder.add_edge("approve_change", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before),
    )
