"""The enhancer and gate G1. Step 1.4, ADR-005 and ADR-007.

The load-bearing test here is `resuming does not re-run the enhancer`. ADR-007
exists because LangGraph re-runs an interrupted node from its first line, so
an enhancer sharing a node with its gate would be paid for twice AND could
show a human different text than the text they approved. This is the test that
proves the split works.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from dynaflows.contracts.state import GateDecision, initial_state
from dynaflows.contracts.tiers import Tier
from dynaflows.graph import build_graph, open_checkpointer
from dynaflows.graph.prompts import EnhancedPrompt
from tests.conftest import FakeGateway, make_playbook

pytestmark = [pytest.mark.deterministic, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def cfg(thread: str, gateway: FakeGateway, **extra: object) -> dict:
    return {
        "configurable": {
            "thread_id": thread,
            "gateway": gateway,
            "playbook": make_playbook(),
            # G1 is the subject here; G2 is auto-approved so it never appears.
            # Merged, not overridden: a test that adds "prompt" must not
            # accidentally re-enable G2 and hang on a gate it never asked for.
            "auto_approve": ["plan", *(extra.pop("auto_approve", []) or [])],  # type: ignore[misc]
            **extra,
        }
    }


async def run_to_gate(saver, thread: str, gateway: FakeGateway, prompt: str = "audit auth"):  # noqa: ANN001, ANN201
    graph = build_graph(saver)
    out = await graph.ainvoke(initial_state("r", thread, prompt), cfg(thread, gateway))
    return graph, out


# --- the enhancer --------------------------------------------------------


async def test_the_enhancer_runs_at_small_tier_and_halts_at_the_gate(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        _, out = await run_to_gate(saver, "t1", gateway)
    assert gateway.enhancer_calls == 1
    assert gateway.requests[0].tier is Tier.SMALL
    assert gateway.requests[0].schema is EnhancedPrompt
    assert "__interrupt__" in out


async def test_the_gate_payload_shows_the_human_what_was_assumed(tmp_path: Path) -> None:
    """A human approving a rewrite needs to see what was assumed on their
    behalf, not just the result."""
    gateway = FakeGateway("rewritten", ["assumed the HTTP layer, not the DB"])
    async with open_checkpointer(tmp_path / "s.db") as saver:
        _, out = await run_to_gate(saver, "t2", gateway, "audit auth")
    payload = out["__interrupt__"][0].value
    assert payload["gate"] == "prompt"
    assert payload["original"] == "audit auth"
    assert payload["enhanced"] == "rewritten"
    assert payload["assumptions"] == ["assumed the HTTP layer, not the DB"]


async def test_the_enhancer_records_its_own_cost(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t3", gateway)
        snapshot = await graph.aget_state(cfg("t3", gateway))
    ledger = snapshot.values["cost"]
    assert (ledger.calls_made, ledger.tokens_in, ledger.tokens_out) == (1, 40, 12)
    assert ledger.usd_spent == pytest.approx(0.002)


# --- ADR-007: the reason the gate is its own node ------------------------


async def test_resuming_does_not_re_run_the_enhancer(tmp_path: Path) -> None:
    """The whole point of ADR-007.

    A resumed graph re-runs the interrupted node from its first line. If the
    enhancer shared a node with its gate, this count would be 2 -- paid twice,
    and the second answer might not be the one the human approved.
    """
    gateway = FakeGateway()
    db = tmp_path / "s.db"
    async with open_checkpointer(db) as saver:
        graph, out = await run_to_gate(saver, "t4", gateway)
        shown = out["__interrupt__"][0].value["enhanced"]

    # A fresh process resumes the same thread.
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("t4", gateway))
    assert gateway.enhancer_calls == 1, "the enhancer was paid for twice"
    assert final["enhanced_prompt"] == shown, "the approved text is not the text that proceeded"


async def test_the_text_that_proceeds_is_the_text_that_was_shown(tmp_path: Path) -> None:
    gateway = FakeGateway("EXACT TEXT SHOWN")
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, out = await run_to_gate(saver, "t5", gateway)
        assert out["__interrupt__"][0].value["enhanced"] == "EXACT TEXT SHOWN"
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("t5", gateway))
    assert final["enhanced_prompt"] == "EXACT TEXT SHOWN"


# --- the three answers ---------------------------------------------------


async def test_approve_lets_the_run_continue(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t6", gateway)
        final = await graph.ainvoke(Command(resume={"decision": "approve"}), cfg("t6", gateway))
    assert final["prompt_gate"].decision is GateDecision.APPROVE
    assert final["evaluation"] is not None
    assert final.get("halted") is None


async def test_edit_replaces_the_model_text_without_asking_again(tmp_path: Path) -> None:
    """The human's text beats the model's. Re-asking would defeat the option."""
    gateway = FakeGateway("model wording")
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t7", gateway)
        final = await graph.ainvoke(
            Command(resume={"decision": "edit", "replacement": "my wording"}),
            cfg("t7", gateway),
        )
    assert final["enhanced_prompt"] == "my wording"
    assert gateway.enhancer_calls == 1


async def test_reject_ends_the_run_before_anything_else_is_spent(tmp_path: Path) -> None:
    """ADR-005: the value of gating here is that a 'no' costs one small-tier
    call and nothing else."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t8", gateway)
        final = await graph.ainvoke(Command(resume={"decision": "reject"}), cfg("t8", gateway))
    assert final["prompt_gate"].decision is GateDecision.REJECT
    assert final["halted"] == "rejected by human at gate G1"
    assert final.get("evaluation") is None, "the run continued past a rejection"
    assert gateway.enhancer_calls == 1


async def test_an_unparseable_answer_is_a_rejection_not_an_approval(tmp_path: Path) -> None:
    """Defaulting an unreadable answer to approval would let a typo authorise
    a fan-out."""
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t9", gateway)
        final = await graph.ainvoke(Command(resume=12345), cfg("t9", gateway))
    assert final["prompt_gate"].decision is GateDecision.REJECT
    assert "unparseable" in (final["prompt_gate"].note or "")


async def test_a_bare_string_answer_is_accepted(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph, _ = await run_to_gate(saver, "t10", gateway)
        final = await graph.ainvoke(Command(resume="approve"), cfg("t10", gateway))
    assert final["prompt_gate"].decision is GateDecision.APPROVE


# --- --yes-prompt (ADR-005) ---------------------------------------------


async def test_auto_approve_skips_the_gate_entirely(tmp_path: Path) -> None:
    gateway = FakeGateway()
    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            initial_state("r", "t11", "audit auth"),
            cfg("t11", gateway, auto_approve=["prompt"]),
        )
    assert "__interrupt__" not in final
    assert final["prompt_gate"].note == "--yes-prompt"
    assert final["evaluation"] is not None


async def test_a_missing_gateway_is_a_typed_config_error(tmp_path: Path) -> None:
    from dynaflows.contracts.errors import DynaflowsError, ErrorCode

    async with open_checkpointer(tmp_path / "s.db") as saver:
        graph = build_graph(saver)
        with pytest.raises(DynaflowsError) as excinfo:
            await graph.ainvoke(
                initial_state("r", "t12", "x"), {"configurable": {"thread_id": "t12"}}
            )
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_INVALID


async def test_a_gateway_in_config_does_not_break_strict_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependencies ride in `configurable`, and checkpoints are strict.

    Worth an explicit test: if LangGraph ever tried to serialise the config
    alongside the state, a GatewayClient would be exactly the unregistered type
    that strict msgpack refuses -- and every gated run would stop resuming.
    """
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    gateway = FakeGateway()
    db = tmp_path / "s.db"
    async with open_checkpointer(db) as saver:
        graph, out = await run_to_gate(saver, "strict-1", gateway)
        assert "__interrupt__" in out
    async with open_checkpointer(db) as saver:
        graph = build_graph(saver)
        final = await graph.ainvoke(
            Command(resume={"decision": "approve"}), cfg("strict-1", gateway)
        )
    assert final["prompt_gate"].decision is GateDecision.APPROVE
    assert gateway.enhancer_calls == 1
