"""Tests for cup_handle trend-aware exit helpers & TP logic."""

import numpy as np
import pandas as pd

from src.bt.strategies.cup_handle import is_uptrend, trend_strength


def _synthetic(n: int, start: float, step: float, noise: float = 0.0) -> tuple:
    """Build a monotonically increasing (or drifting) OHLC series."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    base = start + step * np.arange(n)
    closes = base.copy()
    highs = base + noise + 0.02
    lows = base - noise - 0.02
    return (
        pd.Series(highs, index=idx),
        pd.Series(lows, index=idx),
        pd.Series(closes, index=idx),
    )


def test_trend_strength_strong_trend():
    # Consistent daily advances with minimal noise => high ADX.
    hi, lo, cl = _synthetic(120, start=100.0, step=1.0, noise=0.01)
    val = trend_strength(hi, lo, cl, adx_window=14)
    assert val >= 25.0


def test_trend_strength_flat_chops_is_low():
    # Oscillating between fixed bounds => ADX near zero.
    n = 120
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = pd.Series(100.0 + 2.0 * np.sin(np.arange(n) / 3.0), index=idx)
    highs = closes + 0.5
    lows = closes - 0.5
    val = trend_strength(highs, lows, closes, adx_window=14)
    assert val < 25.0


def test_is_uptrend_true_in_rising_trend():
    hi, lo, cl = _synthetic(120, start=100.0, step=1.0, noise=0.01)
    assert is_uptrend(hi, lo, cl, threshold=25.0) is True


def test_is_uptrend_false_in_declining_series():
    # Close below the slow MA (declining throughout) => not an uptrend even
    # though ADX may be high, because bias is bearish.
    n = 120
    closes = np.linspace(150.0, 90.0, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    hi = pd.Series(closes + 0.5, index=idx)
    lo = pd.Series(closes - 0.5, index=idx)
    cl = pd.Series(closes, index=idx)
    assert is_uptrend(hi, lo, cl, threshold=0.0) is False


def test_is_uptrend_false_on_short_history():
    hi, lo, cl = _synthetic(30, start=100.0, step=1.0, noise=0.01)
    assert is_uptrend(hi, lo, cl) is False  # < ma_span+1 bars
