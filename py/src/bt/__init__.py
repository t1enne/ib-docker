from src.bt.types import PortfolioResult
from datetime import date
from dataclasses import dataclass
from enum import Enum
import click
import yaml
import logging

from src.bt.engine.walk_forward_engine import WalkForwardEngine
from src.bt.algos.pairs_trading import PairsTradingStrategy
from src.bt.metrics import print_results_analysis


logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    name: str
    strategy_type: str
    symbols: list[str]
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
    rolling_window_size: int
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


async def backtest(strategy: Strategy):
    """
    Backtest a trading strategy using walk-forward analysis.

    Walk-forward analysis provides more realistic backtesting by:
    - Training on historical data (in-sample)
    - Testing on future data (out-of-sample)
    - Rolling forward through time with retraining at specified intervals
    """
    # Validate inputs
    if (
        strategy.strategy_type in [StrategyType.PND.value, StrategyType.SPREAD.value]
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

    wf_engine = WalkForwardEngine(
        strategy_class=PairsTradingStrategy,
        symbols=strategy.symbols,
        initial_train_start=strategy.training_start,
        initial_train_end=strategy.training_end,
        trading_start=strategy.trading_start,
        trading_end=strategy.trading_end,
        # Strategy parameters
        entry_z=strategy.entry_z,
        stop_loss=strategy.stop_loss,
        take_profit=strategy.take_profit,
        rolling_window_size=strategy.rolling_window_size,
        # Portfolio parameters
        initial_capital=strategy.initial_capital,
        position_size=strategy.position_size,
        commission=strategy.commission,
        plot=strategy.plot,
    )
    # Run walk-forward analysis
    results = await wf_engine.run()
    print_results_analysis(results)
    # display_walk_forward_results(results, strategy.symbols, "pairsstrat")


def display_walk_forward_results(
    results: PortfolioResult, symbols: list[str], strategy: str
):
    """Display walk-forward analysis results"""
    separator = "=" * 60
    output = f"""
{separator}
WALK-FORWARD ANALYSIS - {strategy.upper()} Strategy
Symbols: {", ".join([s.upper() for s in symbols])}
{separator}
Total Return: {results.total_return:.2%}
Sharpe Ratio: {results.sharpe_ratio:.2f}
Max Drawdown: {results.max_drawdown:.2%}
Total Trades: {results.total_return}
{separator}
"""
    click.echo(output.strip())


__all__ = ["backtest", "load_strategy", "Strategy"]
