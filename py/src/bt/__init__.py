from src.bt.plotting.plotting import plot_backtest_results
from datetime import date
import logging

import click
import yaml

from src.bt.engine.functional_engine import FunctionalBacktestEngine
from src.bt.metrics import print_results_analysis
from src.bt.types import StrategyConfig, StrategyType
from src.bt.types import PortfolioResult


logger = logging.getLogger(__name__)


def load_strategy(path: str) -> StrategyConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        for key in ["training_start", "training_end", "trading_start", "trading_end"]:
            if key in data and isinstance(data[key], date):
                data[key] = data[key].isoformat()

    return StrategyConfig(**data)


async def backtest(
    strategy_conf: StrategyConfig, return_output: bool = False, plot: bool = True
):
    """
    Backtest a trading strategy using walk-forward analysis.

    Walk-forward analysis provides more realistic backtesting by:
    - Training on historical data (in-sample)
    - Testing on future data (out-of-sample)
    - Rolling forward through time with retraining at specified intervals

    Args:
        strategy_conf: Strategy configuration
        return_output: If True, return the output string instead of printing
        plot: If True, generate plot (default True)
    """
    # Validate inputs
    if (
        strategy_conf.strategy_type
        in [StrategyType.PND.value, StrategyType.SPREAD.value]
        and len(strategy_conf.symbols) != 2
    ):
        raise click.BadParameter(
            f"{strategy_conf.strategy_type.upper()} strategy requires exactly 2 symbols"
        )

    # Use functional engine
    engine = FunctionalBacktestEngine(strategy_conf)
    results = await engine.run()
    output = print_results_analysis(results.pf, return_output=return_output)

    if plot:
        plot_backtest_results(strategy_conf, results)

    if return_output:
        return output
    return None
