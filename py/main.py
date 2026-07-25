"""IBKR Trading Library — agent-friendly CLI.

Usage:
    py data query AAPL --from 2026-01-01
    py data dl AAPL MSFT --from 2026-01-01
    py kalman run AAPL --q 1e-5
    py kalman pairs AAPL MSFT
    py hmm fit AAPL --n-regimes 3
    py hmm predict AAPL --model hmm_models/AAPL.pkl
    py ind ema --span 20 --symbol AAPL --from 2026-01-01
    py ind rsi --window 14 --symbol AAPL
    py spread analyze AAPL MSFT --from 2024-01-01
    py mx matrix AAPL MSFT GOOGL --from 2024-01-01
    py bt run strategy.yml
    py screen run breakout_screen universe.yml

Pipe composition:
    py data query AAPL --from 2026-01-01 | py kalman run --stdin
    py data query AAPL | py ind ema --span 20 | py ind rsi --window 14
    py data dl AAPL MSFT --from 2026-01-01 | py kalman pairs --stdin | py ind ema --span 50
"""

import click

from src.data.cli import data_group
from src.kalman.cli import kalman_group
from src.hmm.cli import hmm_group
from src.indicators.cli import ind_group
from src.spread.cli import spread_group
from src.mx.cli import mx_group
from src.bt.cli import bt_group
from src.screen.cli import screen_group


@click.group()
def main():
    """IBKR — composable CLI for market data, indicators, models, and backtesting."""


main.add_command(data_group)
main.add_command(kalman_group)
main.add_command(hmm_group)
main.add_command(ind_group)
main.add_command(spread_group)
main.add_command(mx_group)
main.add_command(bt_group)
main.add_command(screen_group)


if __name__ == "__main__":
    main()
