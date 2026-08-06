"""Tests for OnlinePairs — strategy-level Kalman owner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.strategy import OnlinePairs
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


def _pair_state(n: int = 300):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    a = pd.Series(100.0 * (1.0003 ** pd.Series(range(n), index=idx)), index=idx)
    # b co-moves but noisier around beta=2.0: log b = 2*log a + small wiggles
    la = np.log(a.values)
    lb = 2.0 * la + 0.001 * np.sin(np.linspace(0, 30, n))
    b = pd.Series(np.exp(lb), index=idx)
    state = create_initial_backtest_state(
        symbols=["A", "B"], initial_capital=10_000, start_timestamp=idx[0]
    )
    return merge_bt_state(
        state,
        dict(candles={("A", "1d"): _mk_df(a), ("B", "1d"): _mk_df(b)}),
    )


def test_warmup_returns_not_ready():
    kf = OnlinePairs(warmup_bars=150)
    res = kf.observe(_pair_state(100), "A", "B", "1d")
    assert res.ready is False
    assert res.z_score is None


def _pair_state_sliced(n: int):
    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    a_full = pd.Series(100.0 * (1.0003 ** pd.Series(range(300), index=idx)), index=idx)
    # b is built so logB = 2·logA. The filter convention is logA = α + β·logB,
    # so the true β is the inverse slope = 0.5.
    la = np.log(a_full.values)
    lb = 2.0 * la + 0.001 * np.sin(np.linspace(0, 30, 300))
    b_full = pd.Series(np.exp(lb), index=idx)
    a = a_full.iloc[:n]
    b = b_full.iloc[:n]
    state = create_initial_backtest_state(
        symbols=["A", "B"], initial_capital=10_000, start_timestamp=a.index[0]
    )
    return merge_bt_state(
        state, dict(candles={("A", "1d"): _mk_df(a), ("B", "1d"): _mk_df(b)})
    )


def test_becomes_ready_with_beta_converges():
    # Feed sequentially (as a backtest loop would) so the filter sees history
    # and beta converges toward the true 2.0 used to generate b.
    kf = OnlinePairs(ols_warmup=50, warmup_bars=50, z_window=20)
    final = None
    for i in range(50, 300):
        final = kf.observe(_pair_state_sliced(i + 1), "A", "B", "1d")
    assert final is not None and final.ready is True
    assert final.beta is not None
    assert 0.3 < final.beta < 0.7  # converged near the true 0.5
    assert isinstance(final.z_score, float)


def test_missing_symbol_not_ready():
    kf = OnlinePairs()
    st = _pair_state_sliced(200)
    res = kf.observe(st, "A", "NOPE", "1d")
    assert res.ready is False


def test_n_steps_grows_across_observe():
    kf = OnlinePairs(warmup_bars=50)
    st = _pair_state_sliced(200)
    r1 = kf.observe(st, "A", "B", "1d")
    r2 = kf.observe(st, "A", "B", "1d")
    assert r2.n_steps > r1.n_steps
