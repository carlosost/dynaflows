"""Terminal entry point.

Phase 0 ships two commands and no workflow: `doctor` proves the environment,
`models` answers OQ-01 with data instead of opinion.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from dynaflows import __version__
from dynaflows.contracts.errors import DynaflowsError, ErrorCode
from dynaflows.contracts.tiers import Tier
from dynaflows.doctor import Status, run_checks
from dynaflows.settings import get_settings
from dynaflows.store.run_store import get_run_store

app = typer.Typer(
    name="dynaflows",
    help="Dynamic workflow patterns on LangGraph, with per-node model routing.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
# Everything a human reads while a command works, when stdout is
# reserved for the command's actual output (see `brief`).
_err = Console(stderr=True)

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
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Reconstruct the FTS index. Repairs orphan postings."),
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
            consistent = store.fts_is_consistent(connection)
            if report.clean and consistent:
                console.print(f"[green]OK[/] {repository.count()} chunks, {report.summary()}")
                return
            if not consistent:
                # A separate failure from drift, with a separate remedy: these
                # postings have no chunk row left to subtract them, so
                # reindexing does not heal them (AP-20 -- two faults, two
                # messages, never one summary line).
                console.print(
                    "[red]CORRUPT[/] the search index disagrees with the chunk table"
                    " — run `dynaflows index --rebuild`"
                )
            if not report.clean:
                console.print(f"[red]DRIFT[/] {report.summary()}")
            raise typer.Exit(code=1)

        total = index_corpus(connection, settings.playbook_root)
        if rebuild:
            store.rebuild_fts(connection)
            console.print("[dim]search index reconstructed from the chunk table[/]")
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


def _render_plan_gate(payload: dict[str, Any], out: Console | None = None) -> None:
    """The task list, in full. ADR-005: this gate exists because approving a
    well-worded brief tells you nothing about the twelve workers about to run
    on the wrong twelve files -- so it shows the tasks, not a count of them."""
    from rich.table import Table

    c = out or console

    tasks = payload.get("tasks") or []
    c.print()
    c.print(f"[dim]{payload.get('rationale', '')}[/]")
    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("#", width=3)
    table.add_column("task", no_wrap=True)
    table.add_column("does", style="bold")
    table.add_column("objective", overflow="fold")
    table.add_column("playbook", no_wrap=True)
    table.add_column("context", no_wrap=True, justify="right")
    budget = int(payload.get("context_budget") or 0)
    over = set(payload.get("over_budget") or [])
    for index, task in enumerate(tasks, start=1):
        tokens = int(task.get("context_tokens") or 0)
        # Measured, not estimated. The column exists because this gate used to
        # authorise the whole fan-out against a per-task constant that had no
        # relationship to what the workers would be given (ADR-021).
        size = f"{tokens:,}" + (" ⚠" if task.get("task_id") in over else "")
        table.add_row(
            str(index),
            Text(task.get("task_id", "")),
            Text(task.get("capability", "")),
            Text(task.get("objective", "")),
            Text(" ".join(task.get("anchors") or []) or "—"),
            Text(size),
        )
    c.print(table)
    c.print(
        f"[yellow]{len(tasks)} parallel worker(s)[/], "
        f"[dim]{payload.get('estimated_tokens', 0):,} context tokens measured · "
        f"plan {payload.get('plan_hash', '?')}[/]"
    )
    if over:
        c.print(
            f"[yellow]⚠ {len(over)} task(s) name more than the {budget:,}-token worker "
            f"budget:[/] {', '.join(sorted(over))}"
        )
        c.print(
            "[dim]  Those workers will be shown what fits and told what they did not get. "
            "Edit the plan to split them or name fewer files.[/]"
        )


def _render_gate(payload: dict[str, Any], out: Console | None = None) -> None:
    """Show the human what they are approving, and what it cost them nothing to see.

    Writes to `out` rather than to the module console: `brief` reserves stdout
    for the brief itself, so the gate has to be able to go elsewhere.
    """
    from rich.panel import Panel

    c = out or console

    if payload.get("gate") == "plan":
        _render_plan_gate(payload, out=c)
        return

    c.print()
    c.print(
        Panel(
            Text(payload.get("original", "")),
            title="[dim]you asked[/]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    c.print(
        Panel(
            Text(payload.get("enhanced", "")),
            title="[yellow]the workflow will run this[/]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    paths = payload.get("relevant_paths") or []
    if paths:
        # What it thinks the request is ABOUT, checked against the catalogue.
        # This is the cheapest possible way to catch a rewrite that has
        # understood the words and missed the subject.
        c.print("[dim]it reads this as being about:[/]")
        for item in paths:
            c.print(Text(f"  · {item}"), markup=False)
    invented = payload.get("invented_paths") or []
    if invented:
        c.print(f"[yellow]  {len(invented)} path(s) it named do not exist and were dropped:[/]")
        c.print(Text("  " + ", ".join(invented)), markup=False)
    intent = payload.get("intent")
    if intent:
        # Shown because it decides which capability the whole run uses, and a
        # question silently read as work produces a diff nobody asked for.
        claimed = payload.get("intent_claimed")
        if claimed and claimed != intent:
            # A correction is louder than a reading. It says the model got
            # this wrong, which is the most useful thing on the gate when it
            # happens -- silently fixing it would hide a bad enhancer.
            c.print(f"[yellow]read as a {intent}[/] [dim](the model said {claimed})[/]")
        else:
            c.print(f"[dim]read as a {intent}[/]")
    model = payload.get("model")
    if model:
        # At the gate, because it is the single best predictor of whether this
        # brief is any good, and the human is about to judge it.
        c.print(f"[dim]written by {model}[/]")
    assumptions = payload.get("assumptions") or []
    if assumptions:
        c.print("[dim]assumed on your behalf:[/]")
        for item in assumptions:
            c.print(Text(f"  · {item}"))
    else:
        c.print("[dim]no assumptions declared[/]")


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
        # stderr: the brief owns stdout (see `_err`).
        _err.print(f"[dim]opening {editor[0]}…[/]")
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


def _ask(c: Console, question: str, default: str) -> str:
    """Ask on `c`, read from stdin, and put NOTHING on stdout.

    `typer.prompt(..., err=True)` cannot do this. Click writes all but the
    last character of the prompt to the stream `err` selects and then passes
    that last character to `input()`, which always writes to stdout -- a
    deliberate readline workaround (`typer/_click/termui.py`, `prompt_func`):

        echo(text[:-1], nl=False, err=err)
        return f(text[-1:])

    So every interactive gate leaked exactly one byte, and
    `dynaflows brief "..." | pbcopy` pasted a leading space. One byte is not
    the point; a command that reserves stdout for its output and then writes
    something else there is wrong at any size, and the size is why it went
    unnoticed. `input()` with no argument prompts nowhere.
    """
    # escape(): Rich reads square brackets as markup tags, so "[a]pprove
    # [e]dit [r]eject" rendered as "pprove dit eject" -- it swallowed the
    # three letters that tell the user what to type. Introduced by replacing
    # `typer.prompt` to stop the one-byte stdout leak: that function printed
    # plain text and this one does not, and taking over a library call means
    # taking over everything it was doing, not only the part being fixed.
    c.print(f"\n{escape(question)} [dim]\\[{default}][/]: ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt) as exc:
        # No answer is not approval. The gate exists to make a human say yes.
        raise typer.Abort from exc
    return answer or default


def _ask_gate(payload: dict[str, Any], out: Console | None = None) -> dict[str, Any]:
    """approve / edit / reject.

    `edit` opens the text in $EDITOR pre-filled, because a rewritten brief is
    a paragraph and a one-line prompt would make editing it worse than
    rejecting. If no editor is available it falls back to a prompt rather than
    failing.
    """
    c = out or console
    choice = _ask(c, "[a]pprove  [e]dit  [r]eject", default="a")[:1]
    if choice == "r":
        return {"decision": "reject"}
    if choice == "e":
        original = payload.get("enhanced", "")
        edited = _edit_text(original)
        if edited == original:
            c.print("[dim]unchanged — treating as approve[/]")
            return {"decision": "approve"}
        return {"decision": "edit", "replacement": edited}
    return {"decision": "approve"}


async def _drive(
    graph: Any, cfg: dict[str, Any], first_input: Any, ui: Console | None = None
) -> dict[str, Any]:
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
        # `ui`, not `out`: the loop already binds `out` to the graph's result,
        # and naming the console the same thing made it a dict one line later.
        _render_gate(gate_payload, out=ui)
        payload_in = Command(resume=_ask_gate(gate_payload, out=ui))


def _fail(exc: DynaflowsError) -> None:
    """A typed error is a message, not a stack trace.

    Playbook §1.4: errors are typed precisely so a caller can act on them. A
    200-line LangChain traceback for "you are out of credits" throws that away
    and makes the user rediscover what the response already said.
    """
    envelope = exc.envelope
    # stderr, always. This wrote to stdout, so `dynaflows brief "..." | pbcopy`
    # on a failed call copied "MODEL_UNAVAILABLE ... Connection error" into the
    # clipboard -- the exact defect the brief command's stdout/stderr split
    # exists to prevent, in the code that reports failures. An error is never
    # a command's output.
    _err.print(f"\n[red]{envelope.code}[/] {envelope.message}")
    remedy = getattr(exc, "remedy", None)
    if remedy:
        _err.print(f"[yellow]→ {remedy}[/]")
    if envelope.attempts > 1:
        _err.print(f"[dim]after {envelope.attempts} attempt(s)[/]")
    raise typer.Exit(code=3)


def _graph_config(
    settings: Any, thread_id: str, *, auto_approve: list[str], root: Path | None = None
) -> dict[str, Any]:
    """Every dependency a node can ask for, built in ONE place.

    It was built in two, and that is how `run` shipped without a gateway while
    204 tests passed: a mechanical edit matched one copy and not the other. The
    node contract only grows -- step 1.6 added two more keys -- so the number
    of places that must know about it stays at one.
    """
    from dynaflows.gateway.client import get_gateway  # noqa: PLC0415
    from dynaflows.playbook import get_playbook_repository  # noqa: PLC0415

    return {
        "configurable": {
            "thread_id": thread_id,
            # Dependencies ride in `configurable` -- the documented place for
            # them, and the reason a test injects a fake by passing a config
            # rather than patching an import.
            "gateway": get_gateway(settings=settings),
            "playbook": get_playbook_repository(),
            # ADR-008: worker output goes to disk, state carries the reference.
            "store": get_run_store(settings.home),
            # ADR-017: the one directory input resolution may read from, and
            # ADR-018: the one the planner's catalogue is built from. It
            # defaults to the project root but is NOT fixed to it -- a tool
            # that can only ever audit its own repository is a demo.
            "source_root": root or settings.project_root,
            "auto_approve": auto_approve,
        }
    }


def _money(ledger: Any) -> str:
    """The cost line, which must never imply a price it does not have.

    A run that printed "$0.0000 spent" after a real planner call raised no
    error and failed no test; the only thing wrong with it was the number. So
    an unpriced call is named here rather than added in as zero -- the reader
    can see that the total is a floor, not a figure.
    """
    # "on this thread", not "spent". `cost` is a summing reducer and the
    # checkpoint outlives the run, so a revisited thread reports every run it
    # has ever held. Saying "spent" made that figure a claim about the command
    # the user just typed, which it is not.
    line = (
        f"${ledger.usd_spent:.4f} on this thread, ${ledger.usd_avoided:.4f} avoided by cache, "
        f"{ledger.calls_made} call(s)"
    )
    if ledger.calls_cached:
        # Without this, a run that re-used a cached enhancer answer reported
        # one fewer call than the graph made, and the only way to find out why
        # was to reproduce the whole thing. A cache hit is a fact about the
        # run, not an absence.
        line += f" + {ledger.calls_cached} from cache"
    if ledger.calls_attempted:
        # The chain positions walked past before one answered. Silent before,
        # and it mattered: brief quality tracked WHICH model answered, and
        # nothing said that three had already failed.
        line += f" after {ledger.calls_attempted} that returned nothing usable"
    if ledger.calls_trimmed:
        # Not bookkeeping. A planner that had to write inside a smaller output
        # allowance may have written a SHORTER plan, and a run that quietly
        # produced fewer tasks than it wanted is the kind of thing this
        # project has shipped four times as a silent zero.
        line += (
            f" -- {ledger.calls_trimmed} call(s) ran with a REDUCED output allowance"
            " to fit the remaining balance; the result may be shorter than intended"
        )
    if ledger.calls_unpriced:
        line += (
            f" -- {ledger.calls_unpriced} unpriced, so the total is a LOWER BOUND."
            " No provider cost was returned and config/models.toml has no price"
            " for that model."
        )
    return line


def _report(values: dict[str, Any]) -> None:
    """How a finished run is described. Shared, so `run` and `resume` cannot
    drift into describing the same state differently."""
    halted = values.get("halted")
    if halted:
        console.print(f"[red]Stopped.[/] {halted}")
        ledger = values.get("cost")
        if ledger is not None:
            console.print(f"[dim]before stopping: {_money(ledger)}[/]")
        raise typer.Exit(code=2)

    console.print("[green]Completed.[/]")
    report = values.get("evaluation")
    if report is not None:
        # report.render(), not a format string here. The old line could say
        # only ok and failed, so a run of five degraded results printed
        # "0 ok, 0 failed" -- a sentence describing nothing that happened.
        console.print(f"[dim]{report.render()}[/]")
        for reason in report.reasons:
            console.print(f"[yellow]  - {reason}[/]")
    synthesis = values.get("synthesis")
    if synthesis is not None:
        # The point of the run. Printing counters and not saying where the
        # report is makes the user go looking for their own deliverable.
        console.print("[bold]report[/] ", end="")
        console.print(Text(synthesis.path), markup=False)
        if synthesis.preview:
            console.print(Text(synthesis.preview.splitlines()[0][:120]), markup=False)
    ledger = values.get("cost")
    if ledger is not None:
        console.print(f"[dim]{_money(ledger)}[/]")


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
    root: Annotated[
        Path | None,
        typer.Option(help="Directory to analyse. Defaults to the project root."),
    ] = None,
) -> None:
    """Execute the workflow graph.

    `--root` is the tree the planner is shown (ADR-018) and the only tree a
    worker's inputs may resolve within (ADR-017). It defaulted to the project
    root and was not overridable, which meant dynaflows could only ever
    analyse its own repository.
    """
    import asyncio
    import uuid

    from dynaflows.contracts.state import initial_state
    from dynaflows.gateway.telemetry import configure_tracing
    from dynaflows.graph import build_graph, open_checkpointer

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
            cfg = _graph_config(
                settings,
                thread_id,
                auto_approve=(["prompt"] if yes_prompt else []) + (["plan"] if yes_plan else []),
                root=root,
            )
            state = initial_state(
                uuid.uuid4().hex[:8],
                thread_id,
                prompt,
                source_root=str(cfg["configurable"]["source_root"]),
            )
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
    root: Annotated[
        Path | None,
        typer.Option(help="Directory to analyse. Must match the original run."),
    ] = None,
) -> None:
    """Continue a halted or crashed run from its last checkpoint (ADR-008).

    `--root` has to be given again when the original run used one: the
    checkpoint stores the plan, not the tree it was planned against, so a
    resume without it resolves inputs somewhere else and the workers quietly
    analyse the wrong files. Recording the root in state is the better fix and
    is owed (§7).
    """
    import asyncio

    from dynaflows.gateway.telemetry import configure_tracing
    from dynaflows.graph import build_graph, open_checkpointer

    settings = get_settings()
    configure_tracing(settings)  # ADR-011; see the note in `run`.

    async def _go() -> dict[str, Any]:
        async with open_checkpointer(settings.state_db) as saver:
            graph = build_graph(saver)
            # Read state FIRST, with a config carrying only the thread id, so
            # the root this run was actually planned against can be recovered
            # rather than guessed. Resuming against a different tree analyses
            # different files and says nothing about it.
            probe_cfg: dict[str, Any] = {"configurable": {"thread_id": thread}}
            before = await graph.aget_state(probe_cfg)
            recorded = before.values.get("source_root") if before.values else None
            target = root or (Path(recorded) if recorded else None)
            cfg = _graph_config(settings, thread, auto_approve=[], root=target)
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
def diagnose(
    node: Annotated[str, typer.Argument(help="Which node's call to reproduce: enhance | plan")],
    model: Annotated[str | None, typer.Option(help="Override the model to call.")] = None,
    brief: Annotated[
        str, typer.Option(help="The user brief to send.")
    ] = "Review the error handling in the gateway package.",
    max_tokens: Annotated[int, typer.Option(help="Output allowance to request.")] = 4096,
) -> None:
    """Send a node's EXACT call twice: once bare, once through the real stack.

    `probe` proved the model works. It did not prove that THIS call works,
    because it sent a two-field schema and no system prompt while the planner
    sends a nested schema behind a four-thousand-token catalogue. A diagnostic
    that differs from the failing call in three ways cannot tell you which one
    matters.

    So: build the node's real system prompt and real schema, ask LangChain for
    the response_format it would have sent, and fire that at the provider with
    no SDK in the way. Then fire the same CallRequest through the production
    path. Exactly one of four things is then true, and each names its own fix.
    """
    import json as json_module

    from dynaflows.contracts.calls import CallRequest
    from dynaflows.contracts.state import MAX_FANOUT
    from dynaflows.gateway.invoker import langchain_response_format, raw_completion
    from dynaflows.gateway.probe import call_through_gateway
    from dynaflows.graph.capabilities import render_capabilities
    from dynaflows.graph.prompts import (
        ENHANCER_SYSTEM,
        PLANNER_SYSTEM,
        EnhancedPrompt,
        PlanDraft,
    )
    from dynaflows.playbook.repository import get_playbook_repository

    settings = get_settings()

    if node == "enhance":
        system = ENHANCER_SYSTEM
        schema: Any = EnhancedPrompt
        tier = Tier.SMALL
    elif node == "plan":
        repository = get_playbook_repository()
        system = PLANNER_SYSTEM.format(
            max_fanout=MAX_FANOUT,
            capabilities=render_capabilities(),
            section_map=repository.section_map(),
        )
        schema = PlanDraft
        tier = Tier.FRONTIER
    else:
        console.print(f"[red]Unknown node {node!r}. Expected 'enhance' or 'plan'.[/]")
        raise typer.Exit(code=2)

    from dynaflows.gateway.registry import load_registry

    registry = load_registry(settings.models_config)
    model_id = model or registry.tiers[tier].chain[0]

    response_format = langchain_response_format(schema)
    console.print(f"[bold]node[/] {node}   [bold]tier[/] {tier.value}   [bold]model[/] ", end="")
    console.print(Text(model_id), markup=False)
    console.print(f"[dim]system prompt[/] {len(system)} chars")
    console.print("[dim]response_format LangChain would send:[/]")
    console.print(Text(json_module.dumps(response_format, indent=2)[:2500]), markup=False)

    console.print("\n[bold]1. bare POST, same payload, no SDK[/]")
    try:
        status, body = raw_completion(
            settings,
            model_id,
            system=system,
            prompt=brief,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    except Exception as exc:  # noqa: BLE001 -- a diagnostic reports, never traces
        # A transport failure here is a finding, not a crash. Printing a
        # hundred-line traceback for "no route to the provider" buries the one
        # line that matters and makes the second half of the diagnostic
        # unreachable.
        console.print(f"[red]transport failed[/] {type(exc).__name__}: {exc}")
        status, body = 0, {}
    if status == 0:
        # No response to read. Saying "choices is missing" here would report a
        # provider fault for a network one.
        pass
    elif isinstance(body, str):
        console.print(f"[dim]HTTP[/] {status}")
        console.print(Text(body[:2000]), markup=False)
    else:
        console.print(f"[dim]HTTP[/] {status}")
        choices = body.get("choices")
        if choices:
            first = choices[0]
            console.print(f"[dim]finish_reason[/] {first.get('finish_reason')}")
            content = (first.get("message") or {}).get("content")
            console.print(Text(str(content)[:1500]), markup=False)
        else:
            console.print("[red]choices is missing or null -- this is the failure[/]")
        if usage := body.get("usage"):
            console.print(f"[dim]usage[/] {usage}")
        if error := body.get("error"):
            console.print("[red]error[/] ", end="")
            console.print(Text(json_module.dumps(error, indent=2)[:1500]), markup=False)

    console.print("\n[bold]2. same request through the production path[/]")
    attempt = call_through_gateway(
        settings,
        model_id,
        CallRequest(
            tier=tier,
            system=system,
            prompt=brief,
            schema=schema,
            max_tokens=max_tokens,
            label=f"diagnose.{node}",
            metadata=(("diagnostic", "true"),),
        ),
    )
    if attempt.ok:
        console.print(
            f"[green]OK[/] {attempt.detail} ({attempt.tokens_in} in / {attempt.tokens_out} out)"
        )
    else:
        console.print("[red]FAILED[/] ", end="")
        console.print(Text(attempt.detail[:2000]), markup=False)

    console.print("\n[dim]How to read this:[/]")
    console.print("[dim]  both fail  -> the payload; the error body above names which part.[/]")
    console.print("[dim]  1 ok, 2 fails -> the SDK path, not the provider.[/]")
    console.print("[dim]  both ok    -> the failure is upstream of the call.[/]")


@app.command()
def sources(
    root: Annotated[
        Path | None, typer.Option(help="Directory to catalogue. Defaults to the project root.")
    ] = None,
    show: Annotated[
        bool, typer.Option("--show/--summary", help="Print the whole listing.")
    ] = False,
) -> None:
    """Print the source catalogue the planner will be shown (ADR-018).

    The planner's view of the code is now the thing that decides whether a plan
    names real files, and it was invisible: run `w1` produced five tasks with
    no inputs and a fabricated audit, and nothing in the CLI could have shown
    why. `dynaflows context` does this for the playbook; this is its other half.
    """
    from dynaflows.playbook.tokens import estimate_tokens
    from dynaflows.store.source_map import build_source_map

    settings = get_settings()
    target = root or settings.project_root
    catalogue = build_source_map(target)

    console.print(f"[dim]root[/] {target}")
    console.print(
        f"[dim]listed[/] {catalogue.listed} of {catalogue.total} file(s), "
        f"~{estimate_tokens(catalogue.render())} tokens"
    )
    if catalogue.truncated:
        console.print(
            "[yellow]TRUNCATED[/] the planner will be told so, but it cannot name what it "
            "cannot see. Narrow --root or raise the budget."
        )
    if not catalogue.total:
        console.print("[red]Nothing to show.[/] Every task will be planned blind.")
        raise typer.Exit(code=1)
    if show:
        console.print(Text(catalogue.render()), markup=False)
    else:
        head = "\n".join(catalogue.text.splitlines()[:15])
        console.print(Text(head), markup=False)
        if catalogue.listed > 15:
            console.print(f"[dim]... {catalogue.listed - 15} more; --show for all[/]")


@app.command()
def calibrate(
    model: Annotated[
        str | None, typer.Option(help="Score one model instead of the tier's.")
    ] = None,
    sweep: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated models to score side by side. 'tiers' uses each chain's head."
        ),
    ] = None,
    runs: Annotated[
        int, typer.Option(help="Repeat and report the spread. One run is a sample.")
    ] = 1,
    lenses: Annotated[
        bool,
        typer.Option("--lenses", help="Fan out over the same file with three lenses, and union."),
    ] = False,
    show: Annotated[bool, typer.Option("--show/--quiet", help="Print every finding.")] = False,
) -> None:
    """Run a worker against a fixture whose defects are known. ADR-022.

    "No findings" is unfalsifiable on its own: a clean codebase and a worker
    that cannot find anything produce identical output, and run `s2` produced
    exactly that against the gateway package.

    `--sweep` scores several models on the same fixture, which is how OQ-01
    stops being an opinion. ADR-006 assigned the cheapest paid model to the
    workers on a cost-asymmetry argument that never checked whether that model
    could do the work.

    It measures the worker, not the truth. Five planted defects say nothing
    about the defects nobody planted, and full recall here is a floor rather
    than a ceiling.
    """
    from dynaflows.calibration import fixture_paths, load_manifest

    settings = get_settings()
    source, manifest_path = fixture_paths()
    defects = load_manifest(manifest_path)

    if sweep:
        models = _sweep_models(settings, sweep)
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        table.add_column("model", no_wrap=True)
        table.add_column("recall", justify="right")
        table.add_column("claimed", justify="right")
        table.add_column("discarded", justify="right")
        table.add_column("$", justify="right")
        table.add_column("missed", overflow="fold")
        for candidate in models:
            card, result = _calibrate_once(settings, source, defects, candidate)
            if card is None:
                table.add_row(Text(candidate), Text("error"), "", "", "", Text(str(result)))
                continue
            table.add_row(
                Text(candidate),
                Text(f"{len(card.found)}/{card.planted}"),
                Text(str(card.reported)),
                Text(f"{card.discarded} ({card.discard_rate:.0%})"),
                Text(f"{result.cost_usd or 0:.4f}"),
                Text(", ".join(d.id for d in card.missed) or "—"),
            )
        console.print(table)
        console.print(
            "[dim]Five planted defects in one file. A floor, not a ceiling, and a score that "
            "rises after a prompt change may be overfitting to it (ADR-022).[/]"
        )
        return

    from dynaflows.calibration import aggregate, score

    if lenses:
        # The measurement the fan-out pattern has never had: does splitting one
        # file across three workers find more than one worker does? Every
        # configuration so far produced exactly three findings regardless of
        # schema, so if the cap is per-worker rather than per-file, three
        # workers should clear it.
        found: list[Any] = []
        claimed = discarded = 0
        for name, objective in _LENSES:
            card, result = _calibrate_once(settings, source, defects, model, objective)
            if card is None:
                console.print(f"[yellow]{name}[/] failed: ", end="")
                console.print(Text(str(result)[:200]), markup=False)
                continue
            claimed += card.reported
            discarded += card.discarded
            found.extend(card.findings)
            console.print(
                f"[dim]{name:<16}[/] {len(card.found)}/{card.planted}  ({card.reported} claimed)"
            )
        union = score(found, defects, reported=claimed, discarded=discarded)
        console.print(
            f"[bold]union[/] {len(union.found)}/{union.planted} "
            f"({union.recall:.0%}) from {claimed} claim(s) across {len(_LENSES)} workers"
        )
        console.print(f"[red]  still missed[/] {', '.join(d.id for d in union.missed) or '—'}")
        return

    cards = []
    for _ in range(max(runs, 1)):
        card, result = _calibrate_once(settings, source, defects, model)
        if card is None:
            console.print(f"[red]calibration failed[/] {result}")
            raise typer.Exit(code=2)
        cards.append(card)
    card = cards[-1]
    rollup = aggregate(cards, defects)

    console.print(f"[dim]model[/] {result.model_id}   [dim]status[/] {result.status}")
    console.print(f"[bold]recall[/] {rollup.summary()}")
    if rollup.runs > 1:
        # The spread IS the finding. A defect caught every time and one caught
        # a third of the time say different things about a model, and a mean
        # alone hides which is which. A tier decision was already made on one
        # sample before this existed.
        console.print(f"[green]  always found[/] {', '.join(rollup.always) or '—'}")
        console.print(f"[yellow]  sometimes[/]    {', '.join(rollup.sometimes) or '—'}")
        console.print(f"[red]  never[/]        {', '.join(rollup.never) or '—'}")
    console.print(
        f"[bold]citations[/] {card.reported} claimed, {card.discarded} discarded as "
        f"ungrounded ({card.discard_rate:.0%})"
    )
    if rollup.runs == 1:
        for defect in card.missed:
            console.print(
                f"[yellow]  missed[/] {defect.id} (lines {defect.start}-{defect.end}): ", end=""
            )
            console.print(Text(defect.kind), markup=False)
    if card.unplanted:
        console.print(
            f"[dim]  {len(card.unplanted)} finding(s) outside any planted range -- "
            "possibly real, possibly noise[/]"
        )
    if show:
        for finding in card.findings:
            console.print(
                Text(f"  [{finding.severity}] {finding.file}:{finding.lines} {finding.claim}"),
                markup=False,
            )
            console.print(Text(f"      evidence: {finding.quoted_lines[:160]!r}"), markup=False)
        if card.discarded:
            console.print(
                f"[dim]  {card.discarded} claim(s) were discarded before scoring; the "
                "worker's report in .dynaflows/runs/ quotes each one.[/]"
            )
    if not any(c.found for c in cards):
        console.print(
            "\n[red]Zero recall.[/] This worker cannot find a bare `except: pass`, so "
            "'no findings' from a real run means nothing. The prompt or the tier is the "
            "subject, not the codebase."
        )
        raise typer.Exit(code=1)


# PLACEHOLDER (playbook 4.5): enough to ride out a shared-pool overload, not
# enough to turn a measurement into a long wait.
_CALIBRATION_ATTEMPTS = 3
_CALIBRATION_BACKOFF = 8


def _pin_worker(registry: Any, model_id: str) -> Any:
    """Score THIS model, whatever the MID chain says."""
    from dataclasses import replace

    from dynaflows.contracts.tiers import Tier
    from dynaflows.gateway.registry import TierChain

    return replace(
        registry,
        tiers={**registry.tiers, Tier.MID: TierChain(Tier.MID, "calibration", (model_id,))},
    )


def _sweep_models(settings: Any, sweep: str) -> list[str]:
    """Which models to score. 'tiers' takes the head of each configured chain."""
    if sweep != "tiers":
        return [m.strip() for m in sweep.split(",") if m.strip()]
    from dynaflows.gateway.registry import load_registry

    registry = load_registry(settings.models_config)
    heads = [chain.chain[0] for chain in registry.tiers.values() if chain.chain]
    return list(dict.fromkeys(heads))


# Three lenses over the same file, none of which names a planted defect. They
# are the split a planner would plausibly produce for "audit error handling",
# and they exist to test whether findings scale with WORKER COUNT rather than
# with worker quality -- which is the premise of the whole fan-out pattern and
# has never been measured.
_LENSES = (
    (
        "exception-flow",
        "Audit exception handling and control flow. Report every place an error is caught "
        "and not re-raised, retried when retrying cannot succeed, or allowed to change the "
        "meaning of a return value.",
    ),
    (
        "caller-contract",
        "Audit what a caller can learn when something goes wrong. Report every place the "
        "cause of a failure is discarded, replaced, or flattened into a value the caller "
        "cannot distinguish from success.",
    ),
    (
        "disclosure",
        "Audit logging, diagnostics and anything written out on a failure path. Report "
        "every place a secret, credential or sensitive value can reach a log, and every "
        "place a failure leaves no trace at all.",
    ),
)


def _calibrate_once(
    settings: Any, source: Path, defects: list[Any], model: str | None, objective: str | None = None
) -> tuple[Any, Any]:
    """One worker run over the fixture, through the production node.

    Returns (scorecard, result) or (None, error). A model that cannot be
    reached is a row in the table, not the end of the sweep: the comparison is
    the point and one unavailable endpoint should not cost the other four.
    """
    import asyncio

    from dynaflows.calibration import score
    from dynaflows.contracts.state import PlanTask, initial_state
    from dynaflows.gateway.client import get_gateway
    from dynaflows.graph import nodes
    from dynaflows.graph.prompts import Finding
    from dynaflows.playbook import get_playbook_repository
    from dynaflows.store.run_store import RunStore

    store = RunStore(settings.home)
    gateway = get_gateway(settings=settings)
    if model:
        gateway.registry = _pin_worker(gateway.registry, model)

    task = PlanTask(
        task_id="calibration",
        capability="analyse",
        objective=objective
        or (
            "Audit error handling in this file. Report every place an error is swallowed, "
            "retried when it cannot succeed, stripped of its cause, or exposed in a log."
        ),
        inputs=[source.name],
        playbook_anchors=["AP-09", "§4.1"],
    )
    config: dict[str, Any] = {
        "configurable": {
            "gateway": gateway,
            "playbook": get_playbook_repository(),
            "store": store,
            "source_root": source.parent,
        }
    }
    # Pinning a model for measurement removes the fallback chain, so a
    # transient upstream 429 ends the whole run -- which is what happened the
    # first time a provider was overloaded mid-calibration. Falling back to
    # another model would be worse than failing: it would silently measure a
    # different model and label the number with this one. So the same model is
    # retried, with a backoff, and only a persistent failure is reported.

    transient = {ErrorCode.RATE_LIMIT, ErrorCode.TIMEOUT, ErrorCode.MODEL_UNAVAILABLE}
    state = {**initial_state("calib", "calib", task.objective), "task": task}
    result = None
    for attempt in range(_CALIBRATION_ATTEMPTS):
        out = asyncio.run(nodes.worker(state, config))  # type: ignore[arg-type]
        result = out["results"][0]
        if result.status != "failed":
            break
        code = result.error.code if result.error else None
        if code not in transient:
            break
        if attempt + 1 < _CALIBRATION_ATTEMPTS:
            wait = _CALIBRATION_BACKOFF * (2**attempt)
            console.print(f"[dim]  {code} from the provider; retrying in {wait}s[/]")
            time.sleep(wait)
    assert result is not None
    if result.status == "failed":
        return None, result.error.message if result.error else "failed"

    findings: list[Any] = []
    if result.findings_ref is not None:
        raw = store.read_data(result.findings_ref)
        findings = [Finding.model_validate(f) for f in raw] if isinstance(raw, list) else []

    card = score(
        findings,
        defects,
        reported=result.findings_reported,
        discarded=result.findings_reported - result.findings_grounded,
    )
    return card, result


async def _guard_thread(saver: Any, cfg: dict[str, Any], prompt: str) -> None:
    """Refuse to start a second run on a thread that already has one.

    LangGraph appends: invoking a graph with a fresh input on a thread that
    already holds state starts ANOTHER run over the same checkpoint, re-runs
    nodes, and -- because `cost` is a reducer that sums -- accumulates one
    ledger across both. The cost line then reports the lifetime of the thread
    while saying "spent", which is how `brief --thread p2` reported $0.0066
    for what its author believed was one free call.

    Fifth wrong number in this project, and the same shape as the other four:
    structurally valid, semantically false, asserted nowhere.

    Same prompt on the same thread is a revisit and is allowed -- that is what
    `--thread` is for. A different prompt is a new run wearing an old name.
    """
    tup = await saver.aget_tuple(cfg)
    if tup is None:
        return
    previous = str((tup.checkpoint.get("channel_values") or {}).get("raw_prompt") or "")
    if previous.strip() == prompt.strip():
        return
    raise DynaflowsError.of(
        ErrorCode.CONFIG_INVALID,
        f"thread {cfg['configurable']['thread_id']!r} already holds a run of a different"
        " prompt. Re-using it would start a second run over the same state and bill both"
        " to one ledger. Choose another --thread, or omit it for a fresh one.",
    )


def _save_brief(
    settings: Any,
    values: dict[str, Any],
    thread_id: str,
    enhanced: str,
    ledger: Any,
    *,
    rejected: bool,
) -> None:
    """Keep the run, and say where it went.

    Under the PROJECT root, never under `--root`. A brief can be about
    another repository; writing this project's bookkeeping into someone
    else's tree is exactly what ADR-016 forbids, and `--root` is the option
    that makes it possible to do by accident.

    Failing to save is reported and does not fail the command: the brief is
    already on stdout, and losing a record is not worth losing the output the
    user actually asked for.
    """
    from dynaflows.store.briefs import BRIEF_DIR_NAME, BriefRecord, save_brief

    record = BriefRecord(
        thread_id=thread_id,
        run_id=str(values.get("run_id") or ""),
        raw_prompt=str(values.get("raw_prompt") or ""),
        brief=enhanced,
        decision="reject" if rejected else "approve",
        source_root=str(values.get("source_root") or ""),
        cost=_money(ledger) if ledger is not None else "not recorded",
        model=str(values.get("enhancer_model") or ""),
        intent=str(values.get("enhancer_intent") or ""),
        relevant_paths=list(values.get("relevant_paths") or []),
        invented_paths=list(values.get("invented_paths") or []),
        assumptions=list(values.get("enhancer_assumptions") or []),
    )
    try:
        _, record_path = save_brief(settings.project_root / BRIEF_DIR_NAME, record)
    except OSError as exc:
        _err.print(f"[yellow]Could not save the brief:[/] {exc}")
        return
    _err.print(f"[dim]saved {record_path.parent.name}/{record_path.stem}.{{brief.txt,md}}[/]")


@app.command()
def brief(
    prompt: Annotated[str, typer.Argument(help="What you want done, in your own words.")],
    root: Annotated[
        Path | None,
        typer.Option(help="Repository the brief is about. Defaults to the project root."),
    ] = None,
    thread: Annotated[str, typer.Option(help="Thread id. Reuse it to revisit.")] = "",
    yes: Annotated[bool, typer.Option("--yes", help="Skip the gate. For scripting.")] = False,
    save: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Keep the brief and its gate in .ai/. On by default.",
        ),
    ] = True,
) -> None:
    """Sharpen a request against this codebase and print the brief, nothing else.

    The smallest useful version of the whole tool: enhance, show you what it
    understood, and hand you plain text to paste into whatever executes it.

    The brief goes to STDOUT and everything else -- the gate, the paths, the
    assumptions -- goes to stderr, so `dynaflows brief "..." | pbcopy` copies
    the brief and not the furniture. That split is the entire reason this
    command exists rather than being a flag on `run`.
    """
    import asyncio
    import uuid

    from dynaflows.contracts.state import initial_state
    from dynaflows.gateway.telemetry import configure_tracing
    from dynaflows.graph import build_graph, open_checkpointer

    settings = get_settings()
    configure_tracing(settings)  # ADR-011; see the note in `run`.
    thread_id = thread or f"brief-{uuid.uuid4().hex[:8]}"

    async def _go() -> dict[str, Any]:
        async with open_checkpointer(settings.state_db) as saver:
            # Halt before planning: this command's whole job ends at G1.
            graph = build_graph(saver, interrupt_before=("plan",))
            cfg = _graph_config(
                settings, thread_id, auto_approve=["prompt"] if yes else [], root=root
            )
            await _guard_thread(saver, cfg, prompt)
            state = initial_state(
                uuid.uuid4().hex[:8],
                thread_id,
                prompt,
                source_root=str(cfg["configurable"]["source_root"]),
            )
            await _drive(graph, cfg, state, ui=_err)
            snapshot = await graph.aget_state(cfg)
            return dict(snapshot.values)

    try:
        values = asyncio.run(_go())
    except DynaflowsError as exc:
        _fail(exc)

    from dynaflows.contracts.state import GateDecision  # noqa: PLC0415
    from dynaflows.graph.prompts import compose_brief  # noqa: PLC0415

    gate = values.get("prompt_gate")
    rejected = gate is not None and gate.decision is GateDecision.REJECT

    sharpened = values.get("enhanced_prompt") or ""
    if not sharpened.strip():
        _err.print("[red]The enhancer returned nothing.[/]")
        raise typer.Exit(code=2)
    # Composed HERE, at the boundary. This output is pasted into something
    # with a shell; the same text inside the graph reaches workers that have
    # no tools, and telling those to run commands makes them invent output.
    enhanced = compose_brief(sharpened)

    ledger = values.get("cost")
    if save:
        # A rejection is saved too. It is the most informative run there is --
        # the model being wrong with a human's verdict attached -- and keeping
        # only the approvals keeps the wrong half.
        _save_brief(settings, values, thread_id, enhanced, ledger, rejected=rejected)

    if rejected:
        _err.print("[red]Rejected.[/] Nothing written to stdout.")
        raise typer.Exit(code=1)

    # The paths were printed at the gate, with the header that says what they
    # mean. Printing them again here -- bare, after the prompt -- was left over
    # from before `_render_gate` learned to show them, and read as a second,
    # different list.
    if ledger is not None:
        _err.print(f"[dim]{_money(ledger)}  ·  thread {thread_id}[/]")

    # stdout, plain, no markup, no panel. This is the deliverable.
    print(enhanced)


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
