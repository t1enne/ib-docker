"""`bt screen` command — score a universe for manual trading signals."""

from __future__ import annotations

from datetime import datetime

import click
import pandas as pd

from src.bt.cmds._shared import cli_ts, parse_param_grid
from src.bt.table import render_from_dicts


@click.command(name="screen")
@click.argument("screen_name")
@click.option("--symbols", "-s", multiple=True, help="Symbols to score")
@click.option(
    "--universe",
    "-U",
    help="Universe file path (e.g. 'universes/nsdq.json'). Overrides --symbols.",
)
@click.option(
    "--interval",
    "-i",
    multiple=True,
    default=("1d",),
    help="Bar interval to score (1h, 4h, 1d, ...); repeatable for multiple TFs",
)
@click.option("--from", "from_dt", type=click.DateTime(), help="Start date")
@click.option("--to", "to_dt", type=click.DateTime(), help="End date")
@click.option(
    "--params",
    "-p",
    default="{}",
    help="Screen params as JSON, e.g. '{fast: 20, slow: 50}'",
)
@click.option("--top", type=int, default=None, help="Limit to top-N ranked rows")
def screen(
    screen_name: str,
    symbols: tuple[str, ...],
    universe: str | None,
    interval: tuple[str, ...],
    from_dt: datetime | None,
    to_dt: datetime | None,
    params: str,
    top: int | None,
):
    """Score a universe and print a ranked table of signals.

    Output is a rank only, sorted by score desc. A high score means the entry
    condition fired — it is NOT a profit expectation (pre-cost by design).
    """
    parsed_params = parse_param_grid(params) if isinstance(params, str) else {}
    from_ts: pd.Timestamp = (
        cli_ts(from_dt)
        if from_dt
        else cli_ts(pd.Timestamp.now() - pd.Timedelta(days=365))
    )
    to_ts: pd.Timestamp = cli_ts(to_dt) if to_dt else cli_ts(pd.Timestamp.now())

    from src.bt.screen.screens import init_screen
    from src.bt.screen import run_screen
    from src.bt.screen.adapter import state_per_interval
    from src.bt.screen.runner import DivergenceParams, rank_divergence

    # Fail fast on an unknown screen before touching the feed.
    init_screen(screen_name)

    from src.data import load_universe_config

    syms: list[str]
    if universe:
        syms = list(load_universe_config(universe).symbols)
    elif symbols:
        syms = list(symbols)
    else:
        raise click.UsageError("provide --symbols or --universe/-U")

    # Multi-interval: merge per-TF states into a cross-timeframe consensus rank
    # so each symbol appears ONCE, ranked by trend alignment (TF divergence).
    # Single-interval: fall back to the per-state screen rank.
    states = state_per_interval(syms, from_ts, to_ts, sorted(set(interval)))
    if len(states) > 1:
        merged = rank_divergence(
            states, DivergenceParams(alignment_threshold=0.5), top=top
        )
        rows = [
            {
                "interval": ", ".join(sorted(states)),
                "symbol": r.symbol,
                "action": r.action,
                "score": f"{r.score:.3f}",
                "signals": ", ".join(r.signals),
                "timestamp": str(r.timestamp),
            }
            for r in merged
        ]
    else:
        rows = []
        for iv, state in states.items():
            for r in run_screen(state, screen_name, parsed_params):
                rows.append(
                    {
                        "interval": iv,
                        "symbol": r.symbol,
                        "action": r.action,
                        "score": f"{r.score:.3f}",
                        "signals": ", ".join(r.signals),
                        "timestamp": str(r.timestamp),
                    }
                )
        if top is not None:
            rows = sorted(rows, key=lambda x: float(x["score"]), reverse=True)[:top]

    if not rows:
        click.echo("No signals.")
        return

    for line in render_from_dicts(
        ["interval", "symbol", "action", "score", "signals", "timestamp"],
        rows,
        align="<",
    ):
        click.echo(line)


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(screen)
