"""`bt run` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import _json_default


@click.command(name="run")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["jsonl", "json", "text"]),
    default="text",
    help="Output format: jsonl (equity curve lines), json (full result), "
    "or text (summary)",
)
def run(strategy_file: str, fmt: str):
    """Run a backtest from a strategy JSON config file.

    STRATEGY_FILE: JSON config with symbols, dates, strategy params.

    Output:
      text  — human-readable summary table.
      json  — full structured result as one JSON document (metrics, trades,
              equity curve, benchmarks).
      jsonl — equity curve points as JSON lines, then a final metrics+trades
              record.
    """
    from src.bt import load_strategy, run_backtest_results
    from src.bt import get_backtest_results_analysis
    from src.bt.output import render_result_json, render_result_jsonl

    config = load_strategy(strategy_file)
    results = run_backtest_results(config)

    if fmt == "json":
        click.echo(
            json.dumps(render_result_json(results), indent=2, default=_json_default)
        )
    elif fmt == "jsonl":
        for line in render_result_jsonl(results):
            click.echo(json.dumps(line, default=_json_default))
    else:
        click.echo(
            get_backtest_results_analysis(
                results.pf, benchmark_curves=results.benchmark_curves
            )
        )


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(run)
