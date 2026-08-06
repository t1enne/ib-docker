"""Backtest CLI — run walk-forward backtests from strategy configs.

Usage:
    py bt run breakout.json
    py data query AAPL --from 2024-01-01 | py ind ema --span 20 | py bt run strategy.json
"""

from __future__ import annotations

import click

from src.bt.cmds.analyze import register as register_analyze
from src.bt.cmds.optimize import register as register_optimize
from src.bt.cmds.run import register as register_run
from src.bt.cmds.split import register as register_split
from src.bt.cmds.sweep import register as register_sweep


@click.group(name="bt")
def bt_group():
    """Backtesting engine for trading strategies."""


register_run(bt_group)
register_split(bt_group)
register_sweep(bt_group)
register_optimize(bt_group)
register_analyze(bt_group)
