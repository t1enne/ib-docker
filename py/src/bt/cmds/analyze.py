"""`bt analyze` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import _json_default


@click.command(name="analyze")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="json",
    help="Output format: json (full metrics/trades, default) or text (summary).",
)
def analyze(strategy_file: str, fmt: str):
    """Analyze a backtest result from a strategy config.

    Runs the backtest once, then reports the actual PortfolioResult metrics -
    never re-parsed from rendered text.

    Output: JSON with metrics, trades, and equity curve (default), or the
    human-readable summary (text).
    """
    from src.bt import (
        load_strategy,
        run_backtest_results,
        get_backtest_results_analysis,
    )
    from src.bt.output import render_result_json

    config = load_strategy(strategy_file)
    results = run_backtest_results(config)

    if fmt == "text":
        click.echo(
            get_backtest_results_analysis(
                results.pf, benchmark_curves=results.benchmark_curves
            )
        )
    else:
        click.echo(
            json.dumps(render_result_json(results), indent=2, default=_json_default)
        )


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(analyze)
