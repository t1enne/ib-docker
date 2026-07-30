"""Tests for backtest engine — critical paths only."""

import pandas as pd
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.types import StrategyConfig


def _make_multi_idx_df(symbols: list[str], n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    data = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_candle_generator_multi_symbol():
    df = _make_multi_idx_df(["AAPL", "GOOGL"])
    candles = list(candle_generator(df, ["AAPL", "GOOGL"]))
    assert len([c for c in candles if c.symbol == "AAPL"]) == 5
    assert len([c for c in candles if c.symbol == "GOOGL"]) == 5


def test_run_backtest_no_crash():
    cfg = StrategyConfig(
        name="test",
        strategy_type="momentum_regime",
        symbols=["AAPL"],
        stop_loss=0.05,
        take_profit=0.1,
        initial_capital=10000.0,
        position_size=0.2,
        commission=0.5,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1h"],
        strategy_params={},
        model_params={},
    )
    bt = Backtest(cfg)
    df = _make_multi_idx_df(["AAPL"])
    gen = candle_generator(df, ["AAPL"])
    results, state = run_backtest(
        bt, gen, default_execution_handler(), default_risk_handler()
    )
    assert results is not None
    assert state is not None
    assert state.portfolio is not None
