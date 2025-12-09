import asyncio
from datetime import date
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from dataclasses import dataclass
from enum import Enum
import click
import yaml
import logging

from src.bt.engine.backtest_engine import BacktestEngine, DataFeed
from src.bt.engine.walk_forward_engine import WalkForwardEngine
from src.bt.algos.pairs_trading import PairsTradingStrategy
from src.bt.portfolio.portfolio import Portfolio
from src.bt.plotting.plotting import plot_results


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
    Backtest a trading strategy using walk-forward analysis.

    Walk-forward analysis provides more realistic backtesting by:
    - Training on historical data (in-sample)
    - Testing on future data (out-of-sample)
    - Rolling forward through time with retraining at specified intervals
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

        # Use walk-forward analysis as the default backtesting method
        # The strategy defines the walk-forward parameters:
        # - training_start: initial training period start
        # - training_end: initial training period end
        # - trading_end: end of the entire walk-forward period
        # - retrain_interval_months: how often to retrain (step size)

        # Calculate training window size from initial training period
        train_start = pd.Timestamp(strategy.training_start)
        train_end = pd.Timestamp(strategy.training_end)
        train_window_months = int((train_end - train_start).days / 30)  # Approximate months

        wf_engine = WalkForwardEngine(
            strategy_class=PairsTradingStrategy,
            symbols=strategy.symbols,
            initial_train_start=strategy.training_start,
            initial_train_end=strategy.training_end,
            walk_forward_end=strategy.trading_end,
            train_window_months=train_window_months,
            test_window_months=strategy.retrain_interval_months,  # Test window = retrain interval
            step_months=strategy.retrain_interval_months,  # Retrain at specified intervals
            # Strategy parameters
            entry_z=strategy.entry_z,
            stop_loss=strategy.stop_loss,
            take_profit=strategy.take_profit,
            retrain_interval_months=strategy.retrain_interval_months,
            # Portfolio parameters
            initial_capital=strategy.initial_capital,
            position_size=strategy.position_size,
            commission=strategy.commission,
        )

        # Run walk-forward analysis
        asyncio.run(wf_engine.run_walk_forward())

        # Display and plot results
        aggregate_results = wf_engine.get_aggregate_results()
        display_walk_forward_results(aggregate_results, strategy.symbols, strategy.strategy_type)
        if strategy.plot:
            plot_results(aggregate_results, is_walk_forward=True)

    except Exception as e:
        print(e)
        click.echo(f"Error in backtest: {str(e)}", err=True)
        raise click.Abort()


def display_results(results: Dict, symbols: list[str], strategy: str):
    """Display backtest results"""
    click.echo("\n" + "=" * 50)
    click.echo(f"BACKTEST RESULTS - {strategy.upper()} Strategy")
    click.echo(f"Symbols: {', '.join(symbols)}")
    click.echo("=" * 50)

    click.echo(f"Total Return: {results['total_return']:.2%}")
    click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    click.echo(f"Total Trades: {len(results['trades'])}")


def display_walk_forward_results(results: Dict, symbols: list[str], strategy: str):
    """Display walk-forward analysis results"""
    click.echo("\n" + "=" * 60)
    click.echo(f"WALK-FORWARD ANALYSIS - {strategy.upper()} Strategy")
    click.echo(f"Symbols: {', '.join(symbols)}")
    click.echo("=" * 60)

    click.echo(f"Total Return: {results['total_return']:.2%}")
    click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    click.echo(f"Max Drawdown: {results['max_drawdown']:.2%}")
    click.echo(f"Total Trades: {results['total_trades']}")
    click.echo(f"Number of Windows: {results['num_windows']}")
    click.echo(f"Average Window Return: {results['avg_window_return']:.2%}")
    click.echo(f"Window Return Std Dev: {results['std_window_return']:.2%}")
    click.echo(f"Average Window Sharpe: {results['avg_window_sharpe']:.2f}")
    click.echo("=" * 60)


__all__ = ["backtest", "load_strategy", "Strategy"]
