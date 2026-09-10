"""Terminal entry point.

Phase 0 ships two commands and no workflow: `doctor` proves the environment,
`models` answers OQ-01 with data instead of opinion.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

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
        table.add_row(_GLYPH[check.status], check.name, check.detail)
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
    from dynaflows.gateway.probe import catalogue

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
                model.id,
                f"{model.prompt_usd_per_mtok:.2f}",
                f"{model.completion_usd_per_mtok:.2f}",
                f"{model.context_length // 1000}k",
            )
        console.print(table)
        return

    # --suggest is a starting point, not a recommendation. ADR-006 assigns
    # tiers by fan-out multiplier and requires verifier family diversity;
    # cost order alone cannot express either. 4.5: measure before you trust.
    cheapest = by_cost[:limit]
    dearest = sorted(by_cost, key=lambda m: -m.prompt_usd_per_mtok)[:limit]
    longest = sorted(available, key=lambda m: -m.context_length)[:limit]

    console.print("[dim]# Paste into config/models.toml and bump `version`.[/]")
    console.print("[dim]# Cost order is a starting point only -- ADR-006 assigns tiers by[/]")
    console.print("[dim]# fan-out multiplier, and mid_high must not share a family with mid.[/]")
    for tier, candidates in (
        (Tier.SMALL, cheapest),
        (Tier.MID, cheapest),
        (Tier.MID_HIGH, longest),
        (Tier.FRONTIER, dearest),
    ):
        console.print(f"\n[bold][tiers.{tier.value}][/]")
        console.print("chain = [")
        for model in candidates:
            console.print(
                f'    "{model.id}",'
                f"  [dim]# in ${model.prompt_usd_per_mtok:.2f} "
                f"out ${model.completion_usd_per_mtok:.2f} "
                f"ctx {model.context_length // 1000}k[/]"
            )
        console.print("]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
