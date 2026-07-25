"""IBKR Trading Library — agent-friendly CLI.

Usage:
    py bt run strategy.yml
    py bt analyze strategy.yml
    py data query AAPL --from 2026-01-01
    py data dl AAPL MSFT --from 2026-01-01
    py kalman run AAPL --q 1e-5
    py kalman pairs AAPL MSFT
    py hmm fit AAPL --n-regimes 3
    py ind ema --span 20 --symbol AAPL --from 2026-01-01

Pipe composition:
    py data query AAPL | py ind ema --span 20 | py bt run strategy.yml
"""

import click

from src.data.cli import data_group
from src.kalman.cli import kalman_group
from src.hmm.cli import hmm_group
from src.indicators.cli import ind_group
from src.bt.cli import bt_group


@click.group()
def main():
    """IBKR — composable CLI for market data, indicators, models, and backtesting."""


main.add_command(data_group)
main.add_command(kalman_group)
main.add_command(hmm_group)
main.add_command(ind_group)
main.add_command(bt_group)


if __name__ == "__main__":
    main()
