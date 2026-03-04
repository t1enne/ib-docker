# beartype workaround
from src.bt.algos import init_strat

from src.bt.plotting.plotting import plot_backtest_results
from datetime import date
import logging

import click
import yaml

from src.bt.engine.backtest import Backtest, ticks_generator, run_backtest, run
from src.bt.engine.handlers import (
    ExecutionHandler,
    RiskHandler,
    default_execution_handler,
    default_risk_handler,
)
from src.bt.metrics import get_backtest_results_analysis
from src.bt.types import StrategyConfig, StrategyType
from src.bt.types import PortfolioResult


def load_strategy(path: str) -> StrategyConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        for key in ["training_start", "training_end", "trading_start", "trading_end"]:
            if key in data and isinstance(data[key], date):
                data[key] = data[key].isoformat()

        if "hft" in data and "htf" not in data:
            data["htf"] = data.pop("hft")

        if data.get("htf") is None:
            data["htf"] = []

    return StrategyConfig(**data)


async def backtest_async(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (async version).

    Args:
        strategy_conf: Strategy configuration
    """
    from src.bt.data_feed import load_candles

    bt = Backtest(strategy_conf)
    df = load_candles(
        strategy_conf.symbols,
        bt.window.train_start,
        bt.window.test_end,
        strategy_conf.bar,
    )
    strat_mod = init_strat(strategy_conf.strategy_type)
    results = run(bt, df, strat_mod=strat_mod)
    output = get_backtest_results_analysis(results.pf)

    if strategy_conf.plot:
        plot_backtest_results(strategy_conf, results)

    return output


def backtest(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (sync version).

    Args:
        strategy_conf: Strategy configuration
    """
    import asyncio

    return asyncio.run(backtest_async(strategy_conf))
