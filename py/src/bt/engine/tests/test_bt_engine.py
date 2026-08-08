"""Tests for backtest engine — critical paths only."""

import pandas as pd
import pytest
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp


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
        initial_capital=10000.0,
        commission=0.5,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1h"],
        strategy_params={"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
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


def test_build_benchmark_curves_slices_and_normalizes():
    from src.bt.engine.backtest import build_benchmark_curves

    # MultiIndex (symbol, field) with rising close prices
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    data = {
        ("SPY", "close"): [100.0, 110.0, 120.0, 130.0, 140.0],
        ("SPY", "open"): [100.0] * 5,
    }
    bm_df = pd.DataFrame(data, index=idx)
    bm_df.columns = pd.MultiIndex.from_tuples(bm_df.columns)

    cfg = StrategyConfig(
        name="t",
        strategy_type="momentum_regime",
        symbols=["AAPL"],
        initial_capital=1000.0,
        commission=0.5,
        training_start="2025-01-01",
        training_end="2025-01-01",
        trading_start="2025-01-01",
        trading_end="2025-01-05",
        bars=["1d"],
        strategy_params={"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
        model_params={},
        benchmark_symbols=["SPY"],
    )
    curves = build_benchmark_curves(
        bm_df, cfg, parse_timestamp("2025-01-02"), parse_timestamp("2025-01-04")
    )
    assert "SPY" in curves
    # window-sliced to 3 points (2025-01-02..04) but normalized against the
    # in-window first close, so the first sampled point = initial_capital
    ser = curves["SPY"]
    assert len(ser) == 3
    assert ser.iloc[0] == pytest.approx(cfg.initial_capital)


def test_build_benchmark_curves_empty_when_fewer_than_two_points():
    from src.bt.engine.backtest import build_benchmark_curves

    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    data = {("SPY", "close"): [100.0, 101.0, 102.0], ("SPY", "open"): [100.0] * 3}
    bm_df = pd.DataFrame(data, index=idx)
    bm_df.columns = pd.MultiIndex.from_tuples(bm_df.columns)
    cfg = StrategyConfig(
        name="t",
        strategy_type="momentum_regime",
        symbols=["AAPL"],
        initial_capital=1000.0,
        commission=0.5,
        training_start="2025-01-01",
        training_end="2025-01-01",
        trading_start="2025-01-01",
        trading_end="2025-01-03",
        bars=["1d"],
        strategy_params={"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
        model_params={},
        benchmark_symbols=["SPY"],
    )
    # window has a single point -> <2 -> excluded
    curves = build_benchmark_curves(
        bm_df, cfg, parse_timestamp("2025-01-03"), parse_timestamp("2025-01-03")
    )
    assert "SPY" not in curves
