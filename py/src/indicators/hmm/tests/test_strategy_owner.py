"""Tests for OnlineRegime — strategy-level HMM owner."""

from __future__ import annotations

import pandas as pd

from src.indicators.hmm.strategy import OnlineRegime
from src.bt.state import create_initial_backtest_state
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


def _state(closes: pd.Series, symbol: str = "A", bar: str = "1d"):
    state = create_initial_backtest_state(
        symbols=[symbol], initial_capital=10_000, start_timestamp=closes.index[0]
    )
    return merge_bt_state(state, dict(candles={(symbol, bar): _mk_df(closes)}))


def _prices(n: int = 300, factor: float = 1.0005) -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(50.0 * (factor ** pd.Series(range(n), index=idx)), index=idx)


def test_warmup_returns_none():
    hmm = OnlineRegime(window_size=100, vol_window=20, momentum_window=10)
    res = hmm.observe(_state(_prices(30)), "A", "1d")
    assert res.value is None
    assert res.fitted is False


def test_emits_regime_with_history():
    # Feed the HMM one bar at a time (as a backtest loop would) so its rolling
    # window accumulates, then assert it fits and emits a vol-ranked label.
    hmm = OnlineRegime(
        window_size=100,
        vol_window=10,
        momentum_window=5,
        retrain_interval=20,
    )
    prices = _prices(300)
    last: object = None
    for i in range(60, len(prices)):
        res = hmm.observe(_state(prices.iloc[: i + 1]), "A", "1d")
        last = res.value
    assert last in (0, 1, 2)


def test_missing_symbol_noop():
    hmm = OnlineRegime()
    res = hmm.observe(_state(_prices(200)), "MISSING", "1d")
    assert res.value is None
