"""Backtest CLI — run walk-forward backtests from strategy configs.

Usage:
    py bt run breakout.json
    py data query AAPL --from 2024-01-01 | py ind ema --span 20 | py bt run strategy.json
"""

from __future__ import annotations

import asyncio
import json

import click


@click.group(name="bt")
def bt_group():
    """Backtesting engine for trading strategies."""


@bt_group.command(name="run")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["jsonl", "text"]),
    default="text",
    help="Output format: jsonl (equity curve + trades) or text (summary)",
)
def bt_run(strategy_file: str, fmt: str):
    """Run a backtest from a strategy JSON config file.

    STRATEGY_FILE: JSON config with symbols, dates, strategy params.

    Output (text mode): human-readable summary table.
    Output (jsonl mode): equity curve + trades as JSON lines.
    """
    from src.bt import load_strategy, backtest_async

    config = load_strategy(strategy_file)
    output = asyncio.run(backtest_async(config))

    if fmt == "jsonl":
        # Parse the text output and produce JSON
        _output_jsonl_from_text(output)
    else:
        click.echo(output)


@bt_group.command(name="analyze")
@click.argument("strategy_file", type=click.Path(exists=True))
def bt_analyze(strategy_file: str):
    """Analyze a backtest result (load from last run or config).

    Output: JSON with detailed metrics.
    """
    from src.bt import load_strategy, backtest_async

    config = load_strategy(strategy_file)
    output = asyncio.run(backtest_async(config))

    # Parse metrics from text output

    metrics: dict = {}
    for line in output.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip().rstrip("%")
            try:
                metrics[key] = float(val.replace(",", ""))
            except ValueError:
                metrics[key] = val

    click.echo(json.dumps(metrics, indent=2))


def _output_jsonl_from_text(text: str) -> None:
    """Parse text backtest output and emit JSON lines.

    This is a bridge until the backtest engine natively outputs structured data.
    """
    lines = text.split("\n")
    metrics: dict = {}
    trades: list[dict] = []

    in_trades = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "TRADE LOG" in line or "TRADE" in line.upper() and "---" in line:
            in_trades = True
            continue
        if in_trades and ("---" in line or "===" in line):
            in_trades = False
            continue

        if in_trades:
            # Parse trade row
            parts = line.split()
            if len(parts) >= 5:
                try:
                    trades.append(
                        {
                            "symbol": parts[0],
                            "entry": parts[1],
                            "exit": parts[2],
                            "pnl": parts[3],
                            "reason": " ".join(parts[4:]) if len(parts) > 4 else "",
                        }
                    )
                except ValueError, IndexError:
                    pass
        else:
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip().rstrip("%")
                try:
                    metrics[key] = float(val.replace(",", ""))
                except ValueError:
                    metrics[key] = val

    click.echo(json.dumps({"metrics": metrics, "trades": trades}, indent=2))
