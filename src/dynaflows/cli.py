"""Terminal entry point.

Phase 0 ships two commands and no workflow: `doctor` proves the environment,
`models` answers OQ-01 with data instead of opinion.
"""

from __future__ import annotations

from typing import Annotated

import typer
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
