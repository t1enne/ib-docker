"""Tests for regime model updater — critical paths only."""

from __future__ import annotations

import pandas as pd

from src.bt.regime.detectors import create_sma_detector
from src.bt.regime.model_updater import create_regime_model_updater
from src.bt.regime.types import TREND_INT_TO_LABEL
from src.bt.state import Candle, create_initial_backtest_state
from src.bt.engine.utils import merge_bt_state


def _make_tick(
    ts: pd.Timestamp, symbol: str, price: float, interval: str = "1d"
) -> Candle:
    return Candle(
        timestamp=ts,
        symbol=symbol,
        open=price * 0.999,
        high=price * 1.002,
        low=price * 0.998,
        close=price,
        volume=1_000_000,
        interval=interval,
    )


def _make_candles_df(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": 1_000_000,
        },
        index=prices.index,
    )


def test_bull_market_sets_bull():
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = pd.Series(100.0 * (1.0005 ** pd.Series(range(300), index=idx)))
    state = create_initial_backtest_state(
        symbols=["AAPL"], initial_capital=10_000, start_timestamp=idx[0]
    )
    state = merge_bt_state(state, dict(candles={("AAPL", "1d"): _make_candles_df(rng)}))
    update = create_regime_model_updater(
        create_sma_detector(fast_window=20, slow_window=50)
    )
    result = update(state, _make_tick(idx[-1], "AAPL", float(rng.iloc[-1])))
    assert result.model_state.current_regime is not None
    assert TREND_INT_TO_LABEL.get(result.model_state.current_regime) == "BULL"


def test_bear_market_sets_bear():
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = pd.Series(100.0 * (0.999 ** pd.Series(range(300), index=idx)))
    state = create_initial_backtest_state(
        symbols=["AAPL"], initial_capital=10_000, start_timestamp=idx[0]
    )
    state = merge_bt_state(state, dict(candles={("AAPL", "1d"): _make_candles_df(rng)}))
    update = create_regime_model_updater(
        create_sma_detector(fast_window=20, slow_window=50)
    )
    result = update(state, _make_tick(idx[-1], "AAPL", float(rng.iloc[-1])))
    assert TREND_INT_TO_LABEL.get(result.model_state.current_regime) == "BEAR"


def test_short_candles_no_update():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    state = create_initial_backtest_state(
        symbols=["AAPL"], initial_capital=10_000, start_timestamp=idx[0]
    )
    state = merge_bt_state(
        state,
        dict(candles={"AAPL": _make_candles_df(pd.Series([100.0] * 10, index=idx))}),
    )
    update = create_regime_model_updater(create_sma_detector(slow_window=50))
    result = update(state, _make_tick(idx[-1], "AAPL", 100.0))
    assert result.model_state.current_regime is None
