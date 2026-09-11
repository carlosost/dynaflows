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
from dynaflows.graph import nodes

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
