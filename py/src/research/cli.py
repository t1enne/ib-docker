"""`ibkr research` top-level command — cross-sectional statistics engine."""

from __future__ import annotations

from typing import Optional

import click

from src.research.cmds.scan import run_scan


@click.command(name="research")
@click.option(
    "--universe",
    "-U",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Universe .json file PATH (e.g. 'universes/biotech.json').",
)
@click.option(
    "--bench",
    required=True,
    help="Benchmark ETF to beta against (e.g. XBI for biotech, QQQ for nsdq).",
)
@click.option("--from", "-f", "from_date", help="Narrow panel start (YYYY-MM-DD).")
@click.option("--to", "-t", "to_date", help="Narrow panel end (YYYY-MM-DD).")
def research_cmd(
    universe: str,
    bench: str,
    from_date: Optional[str],
    to_date: Optional[str],
) -> None:
    """Scan a universe of stocks against a benchmark for strategy-buildable stats.

    Reads the local candle DB, builds a rectangular daily panel of full-history
    members, then reports per-edge-family statistics: residual dispersion,
    cross-sectional momentum/reversal forecast, vol clustering & spike response,
    intraday catalyst drift-vs-fade, and benchmark-regime stability.
    """
    try:
        report = run_scan(universe, bench, from_date, to_date)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(report)


def register(owner: click.Group) -> None:
    """Attach the research command onto a CLI group (used by main)."""
    owner.add_command(research_cmd)
