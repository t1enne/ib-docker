"""Screen module CLI — stock screener.

Usage:
    py screen run breakout_screen universe.yml --param fast=50 --param slow=200
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Optional

import click


@click.group(name="screen")
def screen_group():
    """Finviz-like stock screener."""


@screen_group.command(name="run")
@click.argument("screen_name")
@click.argument("universe", default="universe.yml")
@click.option("-p", "--param", "params", multiple=True, help="Screen parameter as key=value")
@click.option("--format", "-F", "fmt", type=click.Choice(["jsonl", "text"]), default="text")
def screen_run(
    screen_name: str,
    universe: str,
    params: tuple[str, ...],
    fmt: str,
):
    """Run a stock screen against a universe.

    SCREEN_NAME: Module name in screens/ directory (e.g., breakout_screen).
    UNIVERSE: Path to universe YAML file.

    Example:
        py screen run breakout_screen universe.yml --param fast=50
    """
    from src.syncm import load_universe_config
    from src.screen import make_screen, run_screen, print_screen_output

    parsed_params: dict[str, str | float | int | bool] = {}
    for p in params:
        if "=" not in p:
            raise click.UsageError(f"Invalid param: '{p}'. Use key=value.")
        key, val = p.split("=", 1)
        try:
            if "." in val:
                parsed_params[key] = float(val)
            else:
                parsed_params[key] = int(val)
        except ValueError:
            if val.lower() in ("true", "false"):
                parsed_params[key] = val.lower() == "true"
            else:
                parsed_params[key] = val

    universe_data = load_universe_config(universe)
    assert universe_data.from_date is not None, "Missing from_date in universe config"

    from_date = universe_data.from_date
    to_date = universe_data.to_date or date.today()

    click.echo(f"Screening {len(universe_data.symbols)} symbols with '{screen_name}'...",
               err=True)

    output = asyncio.run(
        run_screen(
            name=screen_name,
            symbols=universe_data.symbols,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            bar=universe_data.bar,
            params=parsed_params,
        )
    )

    if fmt == "jsonl":
        results = []
        for r in output.results:
            results.append({
                "symbol": r.symbol,
                "signal": r.signal,
                "score": r.score,
                "price": r.price,
                "metadata": r.metadata,
            })
        click.echo(json.dumps({
            "screen": output.screen_name,
            "params": output.params,
            "results": results,
        }, indent=2))
    else:
        click.echo(print_screen_output(output))
