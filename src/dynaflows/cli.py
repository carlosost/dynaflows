"""Terminal entry point.

Phase 0 ships two commands and no workflow: `doctor` proves the environment,
`models` answers OQ-01 with data instead of opinion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.table import Table
from rich.text import Text

from dynaflows import __version__
from dynaflows.contracts.errors import DynaflowsError
from dynaflows.contracts.tiers import Tier
from dynaflows.doctor import Status, run_checks
from dynaflows.settings import get_settings

app = typer.Typer(
    name="dynaflows",
    help="Dynamic workflow patterns on LangGraph, with per-node model routing.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_GLYPH = {Status.OK: "[green]OK  [/]", Status.WARN: "[yellow]WARN[/]", Status.FAIL: "[red]FAIL[/]"}


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"dynaflows {__version__}")


@app.command()
def doctor(
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip every check that needs the network.")
    ] = False,
) -> None:
    """Verify the environment. Exits non-zero if anything blocking failed."""
    settings = get_settings()
    console.print(f"[dim]project root:[/] {settings.project_root}")

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("", width=4)
    table.add_column("check", style="bold", no_wrap=True)
    table.add_column("detail", overflow="fold")

    blocking = 0
    warnings = 0
    for check in run_checks(settings, offline=offline):
        # Text() not str: a model id ending ':free' is Rich emoji shorthand,
        # and a detail containing '[' would be read as a style tag.
        table.add_row(_GLYPH[check.status], Text(check.name), Text(check.detail))
        blocking += check.blocking
        warnings += check.status is Status.WARN

    console.print(table)
    if blocking:
        console.print(f"\n[red]{blocking} blocking failure(s).[/] Phase 0 gate is not met.")
        raise typer.Exit(code=1)
    if warnings:
        console.print(f"\n[yellow]{warnings} warning(s).[/] Nothing blocking.")
        return
    console.print("\n[green]All checks passed.[/]")


@app.command()
def index(
    check: Annotated[
        bool,
        typer.Option("--check", help="Report drift and exit non-zero. Builds nothing."),
    ] = False,
) -> None:
    """Build the retrieval index over docs/, or verify it still matches.

    `--check` is the drift guard AP-19 asks for: a generated artifact plus a
    check that it still matches its source makes staleness impossible rather
    than unlikely. Wire it into CI and a save hook.
    """
    from dynaflows.playbook import store
    from dynaflows.playbook.repository import SqlitePlaybookRepository, index_corpus

    settings = get_settings()
    connection = store.connect(settings.playbook_db)
    try:
        repository = SqlitePlaybookRepository(connection, settings.playbook_root)
        if check:
            report = repository.drift()
            if report.clean:
                console.print(f"[green]OK[/] {repository.count()} chunks, {report.summary()}")
                return
            console.print(f"[red]DRIFT[/] {report.summary()}")
            raise typer.Exit(code=1)

        total = index_corpus(connection, settings.playbook_root)
    except DynaflowsError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        connection.close()

    console.print(
        f"[green]Indexed[/] {total} chunks from {settings.playbook_root.name}/ "
        f"→ {settings.playbook_db}"
    )


@app.command()
def context(
    anchors: Annotated[list[str], typer.Argument(help="Anchors, e.g. AP-11 ADR-006 §4.5")],
    budget: Annotated[int, typer.Option(help="Token budget for the pack.")] = 2000,
    search: Annotated[str, typer.Option(help="Also BM25-search this text.")] = "",
) -> None:
    """Print exactly what a worker would receive for these anchors.

    This is the production caller for `pack()`. It exists now, in the same step
    as the packer, so there is never a window where the assembly code is
    exercised only by its own tests (AP-11) -- and step 1.6's worker will call
    the same function rather than growing a second one.
    """
    from dynaflows.playbook import store
    from dynaflows.playbook.pack import pack
    from dynaflows.playbook.repository import SqlitePlaybookRepository

    settings = get_settings()
    connection = store.connect(settings.playbook_db)
    try:
        repository = SqlitePlaybookRepository(connection, settings.playbook_root)
        chunks = repository.by_anchor(anchors)
        hits = len(chunks)
        if search:
            chunks += repository.search(search, k=5)
        packed = pack(chunks, budget)
    finally:
        connection.close()

    if not packed.included_ids and not packed.dropped_ids:
        console.print(f"[yellow]No chunk matches {anchors}.[/] Try `dynaflows index` first.")
        raise typer.Exit(code=1)

    console.print(packed.text, markup=False, highlight=False, soft_wrap=True)
    console.print(
        f"\n[dim]{packed.tokens}/{budget} tokens · {hits} anchor hit(s) · "
        f"{len(packed.included_ids)} included · {len(packed.dropped_ids)} dropped"
        + (f" · truncated {packed.truncated_id}" if packed.truncated_id else "")
        + "[/]"
    )


def _render_plan_gate(payload: dict[str, Any]) -> None:
    """The task list, in full. ADR-005: this gate exists because approving a
    well-worded brief tells you nothing about the twelve workers about to run
    on the wrong twelve files -- so it shows the tasks, not a count of them."""
    from rich.table import Table

    tasks = payload.get("tasks") or []
    console.print()
    console.print(f"[dim]{payload.get('rationale', '')}[/]")
    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("#", width=3)
    table.add_column("task", no_wrap=True)
    table.add_column("does", style="bold")
    table.add_column("objective", overflow="fold")
    table.add_column("playbook", no_wrap=True)
    for index, task in enumerate(tasks, start=1):
        table.add_row(
            str(index),
            Text(task.get("task_id", "")),
            Text(task.get("capability", "")),
            Text(task.get("objective", "")),
            Text(" ".join(task.get("anchors") or []) or "—"),
        )
    console.print(table)
    console.print(
        f"[yellow]{len(tasks)} parallel worker(s)[/], "
        f"[dim]~{payload.get('estimated_tokens', 0):,} tokens estimated · "
        f"plan {payload.get('plan_hash', '?')}[/]"
    )


def _render_gate(payload: dict[str, Any]) -> None:
    """Show the human what they are approving, and what it cost them nothing to see."""
    from rich.panel import Panel

    if payload.get("gate") == "plan":
        _render_plan_gate(payload)
        return

    console.print()
    console.print(
        Panel(
            Text(payload.get("original", "")),
            title="[dim]you asked[/]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    console.print(
        Panel(
            Text(payload.get("enhanced", "")),
            title="[yellow]the workflow will run this[/]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    assumptions = payload.get("assumptions") or []
    if assumptions:
        console.print("[dim]assumed on your behalf:[/]")
        for item in assumptions:
            console.print(Text(f"  · {item}"))
    else:
        console.print("[dim]no assumptions declared[/]")


def _find_editor() -> list[str] | None:
    """The editor to open, or None.

    Order matters. An explicit $VISUAL/$EDITOR is the user's own choice and
    wins. Otherwise nano before vim before vi: someone who never set $EDITOR
    is unlikely to be a vi user, and dropping them into modal editing with no
    warning is its own kind of trap.
    """
    import os
    import shutil

    for variable in ("VISUAL", "EDITOR"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value.split()
    for candidate in ("nano", "vim", "vi"):
        if shutil.which(candidate):
            return [candidate]
    return None


def _edit_in_editor(initial: str) -> str | None:
    """Open the text pre-filled. None if no editor, or the editor failed."""
    import subprocess
    import tempfile

    editor = _find_editor()
    if editor is None:
        return None
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as handle:
        handle.write(initial)
        path = handle.name
    try:
        console.print(f"[dim]opening {editor[0]}…[/]")
        result = subprocess.run([*editor, path], check=False)  # noqa: S603
        if result.returncode != 0:
            return None
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def _read_multiline(current: str) -> str:
    """Last resort when no editor exists at all.

    A one-line `prompt` was the first version of this and it was bad twice
    over: retyping a paragraph into a single line is miserable, and with no
    default an empty Enter re-asked forever. Empty input now KEEPS the text,
    because "I changed my mind about editing" is the likeliest reason someone
    submits nothing.
    """
    console.print(
        "[dim]No editor found. Paste the replacement brief, then a line containing only '.'[/]"
    )
    console.print("[dim]Submit nothing to keep the text above unchanged.[/]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip() or current


def _edit_text(initial: str) -> str:
    """Always returns usable text. Never loops, never returns empty."""
    edited = _edit_in_editor(initial)
    if edited is None:
        return _read_multiline(initial)
    return edited.strip() or initial


def _ask_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """approve / edit / reject.

    `edit` opens the text in $EDITOR pre-filled, because a rewritten brief is
    a paragraph and a one-line prompt would make editing it worse than
    rejecting. If no editor is available it falls back to a prompt rather than
    failing.
    """
    choice = typer.prompt("\n[a]pprove  [e]dit  [r]eject", default="a").strip().lower()[:1]
    if choice == "r":
        return {"decision": "reject"}
    if choice == "e":
        original = payload.get("enhanced", "")
        edited = _edit_text(original)
        if edited == original:
            console.print("[dim]unchanged — treating as approve[/]")
            return {"decision": "approve"}
        return {"decision": "edit", "replacement": edited}
    return {"decision": "approve"}


async def _drive(graph: Any, cfg: dict[str, Any], first_input: Any) -> dict[str, Any]:
    """Run until the graph finishes, answering each gate as it appears.

    A loop, not a single round trip: 1.5 adds gate G2, and a workflow that
    could only ever stop once would have to be rewritten to add the second.
    """
    payload_in: Any = first_input
    while True:
        out = await graph.ainvoke(payload_in, cfg)
        interrupts = out.get("__interrupt__") if isinstance(out, dict) else None
        if not interrupts:
            return dict(out) if isinstance(out, dict) else {}
        gate_payload = interrupts[0].value
        _render_gate(gate_payload)
        payload_in = Command(resume=_ask_gate(gate_payload))


def _fail(exc: DynaflowsError) -> None:
    """A typed error is a message, not a stack trace.

    Playbook §1.4: errors are typed precisely so a caller can act on them. A
    200-line LangChain traceback for "you are out of credits" throws that away
    and makes the user rediscover what the response already said.
    """
    envelope = exc.envelope
    console.print(f"\n[red]{envelope.code}[/] {envelope.message}")
    remedy = getattr(exc, "remedy", None)
    if remedy:
        console.print(f"[yellow]→ {remedy}[/]")
    if envelope.attempts > 1:
        console.print(f"[dim]after {envelope.attempts} attempt(s)[/]")
    raise typer.Exit(code=3)


def _report(values: dict[str, Any]) -> None:
    """How a finished run is described. Shared, so `run` and `resume` cannot
    drift into describing the same state differently."""
    halted = values.get("halted")
    if halted:
        console.print(f"[red]Stopped.[/] {halted}")
        ledger = values.get("cost")
        if ledger is not None:
            console.print(f"[dim]spent ${ledger.usd_spent:.4f} before stopping[/]")
        raise typer.Exit(code=2)

    console.print("[green]Completed.[/]")
    report = values.get("evaluation")
    if report is not None:
        console.print(
            f"[dim]{report.task_count} task(s), {report.ok_count} ok, "
            f"{report.failed_count} failed, passed={report.passed}[/]"
        )
    ledger = values.get("cost")
    if ledger is not None:
        console.print(
            f"[dim]${ledger.usd_spent:.4f} spent, ${ledger.usd_avoided:.4f} avoided by cache, "
            f"{ledger.calls_made} call(s)[/]"
        )


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="What you want the workflow to do.")],
    thread: Annotated[str, typer.Option(help="Thread id. Reuse it to resume.")] = "",
    stop_before: Annotated[str, typer.Option(help="Halt before this node. Debugging aid.")] = "",
    yes_prompt: Annotated[
        bool, typer.Option("--yes-prompt", help="Skip gate G1 (ADR-005).")
    ] = False,
    yes_plan: Annotated[
        bool, typer.Option("--yes-plan", help="Skip gate G2. The fan-out runs unreviewed.")
    ] = False,
) -> None:
    """Execute the workflow graph.

    Step 1.4: the enhancer is real and gate G1 asks before anything else runs.
    The planner, workers and synthesizer are still pass-throughs (1.5-1.7).
    """
    import asyncio
    import uuid

    from dynaflows.contracts.state import initial_state
    from dynaflows.gateway.client import get_gateway
    from dynaflows.gateway.telemetry import configure_tracing
    from dynaflows.graph import build_graph, open_checkpointer
    from dynaflows.playbook import get_playbook_repository

    settings = get_settings()
    # ADR-011. LangChain and LangGraph read os.environ directly, and
    # `get_settings()` is pure -- it does not export anything. Without this
    # call a run emits NO traces while `doctor` keeps reporting LangSmith as
    # OK, which is the silent-failure shape the whole ADR exists to prevent.
    configure_tracing(settings)
    thread_id = thread or f"run-{uuid.uuid4().hex[:8]}"

    async def _go() -> dict[str, Any]:
        async with open_checkpointer(settings.state_db) as saver:
            graph = build_graph(saver, interrupt_before=(stop_before,) if stop_before else ())
            cfg = {
                "configurable": {
                    "thread_id": thread_id,
                    # Dependencies ride in `configurable` -- the documented
                    # place for them, and the reason a test injects a fake by
                    # passing a config rather than patching an import.
                    "gateway": get_gateway(settings=settings),
                    "playbook": get_playbook_repository(),
                    "auto_approve": (["prompt"] if yes_prompt else [])
                    + (["plan"] if yes_plan else []),
                }
            }
            state = initial_state(uuid.uuid4().hex[:8], thread_id, prompt)
            await _drive(graph, cfg, state)
            snapshot = await graph.aget_state(cfg)
            return {"next": snapshot.next, "values": snapshot.values}

    try:
        outcome = asyncio.run(_go())
    except DynaflowsError as exc:
        console.print(
            f"[dim]thread[/] {thread_id}  [dim](resume with: dynaflows resume {thread_id})[/]"
        )
        _fail(exc)
    console.print(f"[dim]thread[/] {thread_id}")
    if outcome["next"]:
        console.print(f"[yellow]HALTED[/] before {', '.join(outcome['next'])}")
        console.print(f"[dim]resume with:[/] dynaflows resume {thread_id}")
        return
    _report(outcome["values"])


@app.command()
def resume(
    thread: Annotated[str, typer.Argument(help="The thread id to continue.")],
) -> None:
    """Continue a halted or crashed run from its last checkpoint (ADR-008)."""
    import asyncio

    from dynaflows.gateway.client import get_gateway
    from dynaflows.gateway.telemetry import configure_tracing
    from dynaflows.graph import build_graph, open_checkpointer
    from dynaflows.playbook import get_playbook_repository

    settings = get_settings()
    configure_tracing(settings)  # ADR-011; see the note in `run`.

    async def _go() -> dict[str, Any]:
        async with open_checkpointer(settings.state_db) as saver:
            graph = build_graph(saver)
            cfg = {
                "configurable": {
                    "thread_id": thread,
                    "gateway": get_gateway(settings=settings),
                    "playbook": get_playbook_repository(),
                    "auto_approve": [],
                }
            }
            before = await graph.aget_state(cfg)
            if not before.created_at:
                return {"missing": True}
            if not before.next:
                # Already finished. Saying "Completed." here would report work
                # that did not happen -- the same two-facts-one-word confusion
                # AP-20 describes, in a status line.
                return {"already_done": True, "values": before.values}
            # None as input means "continue from the checkpoint" rather than
            # "start again" -- the whole point of resume.
            await _drive(graph, cfg, None)
            snapshot = await graph.aget_state(cfg)
            return {"next": snapshot.next, "values": snapshot.values}

    try:
        outcome = asyncio.run(_go())
    except DynaflowsError as exc:
        _fail(exc)
    if outcome.get("missing"):
        console.print(f"[red]No checkpoint for thread {thread}.[/]")
        raise typer.Exit(code=1)
    if outcome.get("already_done"):
        console.print(f"[dim]Thread {thread} already finished. Nothing to resume.[/]")
        _report(outcome["values"])
        return
    if outcome["next"]:
        console.print(f"[yellow]HALTED[/] before {', '.join(outcome['next'])}")
        return
    _report(outcome["values"])


@app.command()
def probe(
    model: Annotated[str, typer.Argument(help="Model id to probe, e.g. openai/gpt-5.6-luna-pro")],
    max_tokens: Annotated[int, typer.Option(help="Output allowance to request.")] = 4096,
    structured: Annotated[
        bool, typer.Option("--structured/--plain", help="Ask for json_schema output.")
    ] = True,
) -> None:
    """Make ONE raw call to a model and print exactly what comes back.

    No LangChain, no ladder, no cache -- a bare chat/completions POST. When the
    stack cannot explain a response, the only honest next step is to look at
    the response rather than reason about it (AP-19 habit 3).

    Diagnostics it answers directly: does this model work at all on this key,
    does it honour json_schema, what `finish_reason` does it return, and how
    many of your max_tokens went to reasoning rather than to the answer.
    """
    import json as json_module

    from dynaflows.gateway.invoker import raw_completion

    settings = get_settings()
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "model_said": {"type": "string"}},
        "required": ["ok", "model_said"],
        "additionalProperties": False,
    }
    status, body = raw_completion(
        settings, model, max_tokens=max_tokens, schema=schema if structured else None
    )
    console.print(f"[dim]HTTP[/] {status}")
    if isinstance(body, str):
        console.print(Text(body[:2000]), markup=False)
        raise typer.Exit(code=0 if status == 200 else 1)

    choices = body.get("choices")
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        console.print(f"[dim]finish_reason[/] {first.get('finish_reason')}")
        content = message.get("content")
        console.print(f"[dim]content[/] {'<empty>' if not content else ''}")
        if content:
            console.print(Text(str(content)[:1200]), markup=False)
        if message.get("reasoning"):
            console.print("[yellow]this model returned reasoning content[/]")
    else:
        console.print("[red]no choices in the response[/]")
    if usage := body.get("usage"):
        console.print(f"[dim]usage[/] {usage}")
    if error := body.get("error"):
        console.print(f"[red]error[/] {error}")
    console.print("\n[dim]full body:[/]")
    console.print(Text(json_module.dumps(body, indent=2)[:3000]), markup=False)


@app.command()
def cache(
    clear: Annotated[bool, typer.Option("--clear", help="Delete every cached response.")] = False,
) -> None:
    """Inspect or clear the response cache (ADR-013).

    Clearing is the manual remedy for the limitation the ADR records: a
    provider can change the model behind a pinned id without changing the id
    -- AP-05's exact shape -- and nothing detects a silent behavioural change.
    """
    from dynaflows.gateway.cache import ResponseCache, connect_cache

    settings = get_settings()
    connection = connect_cache(settings.calls_db)
    try:
        store_ = ResponseCache(connection)
        if clear:
            removed = store_.clear()
            console.print(f"[green]Cleared[/] {removed} cached response(s).")
            return
        console.print(f"{store_.count()} cached response(s) in {settings.calls_db}")
    finally:
        connection.close()


@app.command()
def models(
    suggest: Annotated[
        bool, typer.Option("--suggest", help="Print a models.toml block from the live catalogue.")
    ] = False,
    limit: Annotated[int, typer.Option(help="Candidates to show per tier.")] = 5,
) -> None:
    """Inspect the live structured-output catalogue. Answers OQ-01 with data.

    Model ids are never hardcoded from memory: a renamed id fails silently
    (AP-05), so the only trustworthy source is the provider's live list.
    """
    from dynaflows.gateway.probe import ModelInfo, catalogue

    settings = get_settings()
    try:
        available = catalogue(settings)
    except DynaflowsError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    by_cost = sorted(available, key=lambda m: (m.prompt_usd_per_mtok, m.completion_usd_per_mtok))
    if not suggest:
        table = Table(title=f"{len(available)} endpoints supporting structured outputs")
        table.add_column("model id")
        table.add_column("in $/Mtok", justify="right")
        table.add_column("out $/Mtok", justify="right")
        table.add_column("context", justify="right")
        for model in by_cost:
            table.add_row(
                Text(model.id),
                f"{model.prompt_usd_per_mtok:.2f}",
                f"{model.completion_usd_per_mtok:.2f}",
                f"{model.context_length // 1000}k",
            )
        console.print(table)
        return

    # --suggest must not make a decision it cannot justify from this data.
    # Price tells you cost. It does not tell you reasoning quality, and it does
    # not tell you rate limits -- both of which decide two of the four tiers.
    #
    # Everything below is emitted with markup disabled. A TOML header like
    # [tiers.small] is valid Rich markup and gets swallowed as a style tag, and
    # a model id ending ":free" contains Rich's :emoji: shorthand. Output that
    # is meant to be pasted must be printed literally.
    # A :batch endpoint routes to a provider's asynchronous batch API. Its
    # latency class is different by design, and the gateway calls behind a 60s
    # timeout with a human waiting at gate G2. Whether such an endpoint blocks
    # for hours or times out immediately is UNVERIFIED -- both are useless
    # here, so it never appears as a candidate. This is ADR-006's homogeneity
    # rule in a third dimension the registry does not check.
    batch = [m for m in by_cost if m.id.endswith(":batch")]
    usable = [m for m in by_cost if not m.id.endswith(":batch")]
    free = [m for m in usable if m.id.endswith(":free")]
    paid = [m for m in usable if not m.id.endswith(":free")]

    small = (free + paid)[:limit]
    # Workers run N times per plan. A :free endpoint is rate-limited hard, so
    # at fan-out it produces 429s, not savings (OQ-03).
    mid = paid[:limit]
    mid_families = {m.family for m in mid}
    # ADR-006: diversity by construction, not by hoping the user notices.
    mid_high = [
        m for m in sorted(paid, key=lambda m: -m.context_length) if m.family not in mid_families
    ][:limit]

    lines: list[str] = [
        "# Paste into config/models.toml and bump `version`.",
        "# Order within a chain is preference, then fallback.",
        "# If you set min_context_tokens in [constraints], every model listed here",
        "# must hold at least that many tokens or `dynaflows doctor` fails.",
    ]
    # State the filter. A silent exclusion is indistinguishable from absence,
    # and the reader cannot tell what they were not shown (AP-20).
    if batch:
        lines += [
            f"# {len(batch)} :batch endpoint(s) excluded from every tier below.",
            "# They route to an asynchronous batch API; this CLI calls synchronously",
            "# with a human waiting at a gate. `dynaflows models` still lists them.",
        ]

    def entry(model: ModelInfo, *, commented: bool = False) -> str:
        prefix = '    # "' if commented else '    "'
        return (
            f'{prefix}{model.id}",'
            f"  # in ${model.prompt_usd_per_mtok:.2f}"
            f" out ${model.completion_usd_per_mtok:.2f}"
            f" ctx {model.context_length // 1000}k"
        )

    for tier, candidates, note in (
        (Tier.SMALL, small, "cheapest available; the enhancer and router are easy tasks"),
        (
            Tier.MID,
            mid,
            "cheapest PAID; :free endpoints are rate-limited and this tier runs N times",
        ),
        (Tier.MID_HIGH, mid_high, "longest context, family-disjoint from mid per ADR-006"),
    ):
        lines += ["", f"[tiers.{tier.value}]  # {note}", "chain = ["]
        lines += [entry(m) for m in candidates]
        lines.append("]")

    # The planner is the one tier this command refuses to choose for you.
    lines += [
        "",
        f"[tiers.{Tier.FRONTIER.value}]",
        "# NOT SUGGESTED. The planner needs reasoning quality, and nothing in the",
        "# catalogue measures that. Sorting by price would name the most expensive",
        "# model, which is a legacy-premium heuristic, not a good one.",
        "# Uncomment one or two deliberately. ADR-006: this tier runs ONCE per plan",
        "# but decides the cost and correctness of N worker calls.",
        "chain = [",
    ]
    lines += [
        entry(m, commented=True) for m in sorted(paid, key=lambda m: -m.context_length)[:limit]
    ]
    lines.append("]")

    for line in lines:
        # soft_wrap: Rich hard-wraps at the terminal width by default, which
        # would split a model id across two lines and break the paste on any
        # narrow terminal.
        console.print(line, markup=False, highlight=False, soft_wrap=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
