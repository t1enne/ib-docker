"""BT module — backtesting engine."""

from src.bt.cli import bt_group

from src.bt.strategies import init_strat
from src.bt.engine.backtest import Backtest, candle_generator, run_backtest, run
from src.bt.engine.handlers import (  # noqa: F401
    ExecutionHandler,
    RiskHandler,
    default_execution_handler,
    default_risk_handler,
)
from src.bt.metrics import get_backtest_results_analysis, build_symbol_attribution
from src.bt.types import StrategyConfig, PortfolioResult

import json
import asyncio


def load_strategy(path: str) -> StrategyConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return StrategyConfig(**data)


def run_backtest_results(strategy_conf: StrategyConfig):
    """Run a strategy backtest and return the structured BacktestResults.

    Returns the full ``BacktestResults`` (metrics, trades, equity curve,
    benchmark curves) — not a rendered string. The CLI renders text from
    this; structured output uses it directly with no text round-trip.
    """
    from src.bt.data_feed import load_candles

    bt = Backtest(strategy_conf)
    df = load_candles(
        strategy_conf.symbols,
        bt.window.train_start,
        bt.window.test_end,
        strategy_conf.bars[0],
    )
    strat_mod = init_strat(strategy_conf.strategy_type)
    return run(bt, df, strat_mod=strat_mod)


async def backtest_async(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (async version). Returns text report."""
    results = run_backtest_results(strategy_conf)
    return get_backtest_results_analysis(
        results.pf, benchmark_curves=results.benchmark_curves
    )


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
    "run_backtest_results",
    "get_backtest_results_analysis",
    "build_symbol_attribution",
    "StrategyConfig",
    "PortfolioResult",
    "init_strat",
]
