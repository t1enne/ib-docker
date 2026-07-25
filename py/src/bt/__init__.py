"""BT module — backtesting engine."""

from src.bt.cli import bt_group

from src.bt.strategies import init_strat
from src.bt.engine.backtest import Backtest, candle_generator, run_backtest, run
from src.bt.engine.handlers import (
    ExecutionHandler,
    RiskHandler,
    default_execution_handler,
    default_risk_handler,
)
from src.bt.metrics import get_backtest_results_analysis
from src.bt.types import StrategyConfig, StrategyType, PortfolioResult

import yaml
from datetime import date
import asyncio


def load_strategy(path: str) -> StrategyConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        for key in ["training_start", "training_end", "trading_start", "trading_end"]:
            if key in data and isinstance(data[key], date):
                data[key] = data[key].isoformat()
        if data.get("htf") is None:
            data["htf"] = []
    return StrategyConfig(**data)


async def backtest_async(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (async version). Returns text report."""
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
    return get_backtest_results_analysis(results.pf)


def backtest(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (sync version)."""
    return asyncio.run(backtest_async(strategy_conf))


__all__ = [
    "bt_group",
    "Backtest",
    "candle_generator",
    "run_backtest",
    "run",
    "load_strategy",
    "backtest",
    "backtest_async",
    "get_backtest_results_analysis",
    "StrategyConfig",
    "StrategyType",
    "PortfolioResult",
    "init_strat",
]
