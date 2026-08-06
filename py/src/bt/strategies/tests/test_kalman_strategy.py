"""Tests for kalman_pairs strategy — strategy-owned model path."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.bt.strategies.kalman_pairs as strat
from src.bt.strategies.kalman_pairs import Params
from src.bt.state import ActionType, Candle, create_initial_backtest_state
from src.bt.engine.utils import merge_bt_state


def _mk_df(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": 1_000_000,
        },
        index=closes.index,
    )


def _candle(ts: pd.Timestamp, sym: str, price: float, interval="1d") -> Candle:
    return Candle(
        timestamp=ts,
        symbol=sym,
        open=price * 0.999,
        high=price * 1.002,
        low=price * 0.998,
        close=price,
        volume=1_000_000,
        interval=interval,
    )


def _pair_series(n: int = 300):
    """Return aligned (a, b) close Series. logB = 2·logA + small noise."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    a = pd.Series(100.0 * (1.0003 ** np.arange(n)), index=idx)
    la = np.log(a.values)
    b = pd.Series(np.exp(2.0 * la + 0.001 * np.sin(np.linspace(0, 30, n))), index=idx)
    return a, b


def _sliced_state(i: int):
    a, b = _pair_series()
    a = a.iloc[:i]
    b = b.iloc[:i]
    state = create_initial_backtest_state(
        symbols=["A", "B"], initial_capital=100_000, start_timestamp=a.index[0]
    )
    return merge_bt_state(
        state, dict(candles={("A", "1d"): _mk_df(a), ("B", "1d"): _mk_df(b)})
    )


def test_strategy_model_path_signals_without_model_updater():
    """With use_strategy_model=True the strategy drives its own OnlinePairs and
    does not need the engine's kalman_* fields (which are None here)."""
    strat.reset_global()
    params = Params(pair=("A", "B"), z_entry=0.3, z_exit=0.5, use_strategy_model=True)

    a, _ = _pair_series()
    signals = []
    for i in range(100, 300):
        s = _sliced_state(i)
        sig = strat.on_candle(s, _candle(s.timestamp, "B", 120, "1d"), params)
        signals.extend(sig)

    # The strategy-owned filter should have produced at least one signal over
    # 200 bars of a mean-reverting pair (entry or exit).
    assert signals, "expected the strategy-owned Kalman path to emit signals"
    assert any(s.action == ActionType.long for s in signals) or any(
        s.action == ActionType.short for s in signals
    )


def test_model_updater_path_still_working_default():
    """Default (use_strategy_model=False) requires the engine model_updater."""
    strat.reset_global()
    params = Params(pair=("A", "B"), use_strategy_model=False)
    s = _sliced_state(250)
    # kalman_z_score is None (no model_updater) → no signals, no crash.
    assert strat.on_candle(s, _candle(s.timestamp, "B", 120, "1d"), params) == []
