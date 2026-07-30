"""IBKR Trading Library — agent-friendly CLI.

Usage:
    py bt run strategy.json
    py bt analyze strategy.json
    py data query AAPL --from 2026-01-01
    py data dl AAPL MSFT --from 2026-01-01

Pipe composition:
    py data query AAPL | py bt run strategy.json
"""

import click

from src.data.cli import data_group
from src.bt.cli import bt_group


@click.group()
def main():
    """IBKR — composable CLI for market data and backtesting."""


main.add_command(data_group)
main.add_command(bt_group)


if __name__ == "__main__":
    main()
