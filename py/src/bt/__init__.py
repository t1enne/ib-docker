from datetime import date
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from dataclasses import dataclass
from enum import Enum
import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import logging
from zipline import run_algorithm
from zipline.data.bundles import ingest


from src.utils import (
    calculate_zscore_spread,
    get_ols_fit_model,
    get_returns,
    read_candles,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    z_score: float
    symbol: str
    position: str  # "long" or "short"
    pnl: float = 0.0
    status: str = "open"  # "open", "closed", "stopped"
    close_reason: Optional[str] = None


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    trades: List[Trade]
    equity_curve: pd.Series


@dataclass
class Strategy:
    name: str
    strategy_type: str
    symbols: list[str]
    ma_period: int
    entry_z: float
    stop_loss: float
    take_profit: float
    initial_capital: float
    position_size: float
    commission: float
    training_start: str
    training_end: str
    trading_start: str
    trading_end: str
    retrain_interval_months: int
    plot: bool


def load_strategy(path: str) -> Strategy:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        for key in ["training_start", "training_end", "trading_start", "trading_end"]:
            if key in data and isinstance(data[key], date):
                data[key] = data[key].isoformat()

    return Strategy(**data)


class StrategyType(Enum):
    PND = "pnd"
    SPREAD = "spread"


def backtest(strategy: Strategy):
    """
    Backtest a trading strategy using Zipline.
    """
    try:
        # Validate inputs
        if (
            strategy.strategy_type
            in [StrategyType.PND.value, StrategyType.SPREAD.value]
            and len(strategy.symbols) != 2
        ):
            raise click.BadParameter(
                f"{strategy.strategy_type.upper()} strategy requires exactly 2 symbols"
            )

        # Ingest data into Zipline bundle
        ingest("ibkr_bundle", bundle_kwargs={"symbols": strategy.symbols})

        # Import here to avoid import issues
        from src.bt.algorithm import PairsTradingAlgorithm

        # Run the algorithm
        perf = run_algorithm(
            start=pd.Timestamp(strategy.trading_start),
            end=pd.Timestamp(strategy.trading_end),
            initialize=PairsTradingAlgorithm,
            initialize_kwargs={"strategy": strategy},
            bundle="ibkr_bundle",
            capital_base=strategy.initial_capital,
        )

        # Display results
        display_zipline_results(perf, strategy.symbols, strategy.strategy_type)

    except Exception as e:
        print(e)
        click.echo(f"Error in backtest: {str(e)}", err=True)
        raise click.Abort()


def display_zipline_results(perf: pd.DataFrame, symbols: list[str], strategy: str):
    """Display Zipline backtest results"""
    click.echo("\n" + "=" * 50)
    click.echo(f"BACKTEST RESULTS - {strategy.upper()} Strategy")
    click.echo(f"Symbols: {', '.join(symbols)}")
    click.echo("=" * 50)

    total_return = (perf.portfolio_value.iloc[-1] / perf.portfolio_value.iloc[0]) - 1
    returns = perf.returns.dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    max_dd = (perf.portfolio_value / perf.portfolio_value.expanding().max() - 1).min()

    click.echo(f"Total Return: {total_return:.2%}")
    click.echo(f"Sharpe Ratio: {sharpe:.2f}")
    click.echo(f"Max Drawdown: {max_dd:.2%}")

    # Trades from perf.transactions
    transactions = perf.transactions.dropna()
    if not transactions.empty:
        click.echo(f"Total Trades: {len(transactions)}")
    else:
        click.echo("No trades executed")


__all__ = ["backtest", "load_strategy", "Strategy"]
