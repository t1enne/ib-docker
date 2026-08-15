"""Tests for the screen feed adapter — multi-interval resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.screen.adapter import (
    frames_by_interval,
    frames_from_feed,
    state_per_interval,
)


def _ts(v) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _hourly_df(symbols: list[str], n: int = 120) -> pd.DataFrame:
    """Hourly MultiIndex-column feed: 5 hour bars/day over ~n hours."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    cols = pd.MultiIndex.from_product(
        [symbols, ["open", "high", "low", "close", "volume"]]
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for s in symbols:
        closes = 100.0 * (1.0 + np.linspace(0, 0.4, n))  # steady uptrend
        for f, mul in [("open", 1.0), ("high", 1.002), ("low", 0.998), ("close", 1.0)]:
            df[(s, f)] = closes * mul
        df[(s, "volume")] = 1_000_000.0
    df.columns.names = ["symbol", "field"]
    return df


def test_frames_by_interval_resamples_hourly(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _hourly_df(["A"]))
    frames = frames_by_interval(
        ["A"], _ts("2024-01-01"), _ts("2024-01-06"), ["1h", "4h", "1d"]
    )

    assert set(frames.keys()) == {"1h", "4h", "1d"}
    n_hour = len(frames["1h"][0][1])
    # 4h buckets = n_hour/4; 1d (5 trading hours/day convention in resample)
    # Resample drops partial last bucket, so counts are floors.
    assert 0 < len(frames["4h"][0][1]) < n_hour
    assert 0 < len(frames["1d"][0][1]) < len(frames["4h"][0][1])


def test_state_per_interval_builds_state(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _hourly_df(["A"]))
    states = state_per_interval(
        ["A"], _ts("2024-01-01"), _ts("2024-01-06"), ["1h", "1d"]
    )
    assert set(states.keys()) == {"1h", "1d"}
    for iv, st in states.items():
        frame = st.frame("A")
        assert frame is not None
        assert st.ts == frame["close"].index[-1]


def test_frames_by_interval_missing_symbol_skipped(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _hourly_df(["A"]))
    frames = frames_by_interval(
        ["A", "NA"], _ts("2024-01-01"), _ts("2024-01-06"), ["1h"]
    )
    # "NA" absent from feed -> skipped, only A remains in every interval map.
    for iv, fr in frames.items():
        assert [s for s, _ in fr] == ["A"]


def _gappy_df() -> pd.DataFrame:
    """Two symbols where B carries a >96h hole in its hourly bars."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    cols = pd.MultiIndex.from_product(
        [["A", "B"], ["open", "high", "low", "close", "volume"]]
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    df.columns.names = ["symbol", "field"]
    for s in ["A", "B"]:
        closes = 100.0 * (1.0 + np.linspace(0, 0.4, n))
        for f, mul in [("open", 1.0), ("high", 1.002), ("low", 0.998), ("close", 1.0)]:
            df[(s, f)] = closes * mul
        df[(s, "volume")] = 1_000_000.0
    # Carve a 109-hour hole out of B (keeps rows 0..90 and 200..299), so B has
    # a real >96h discontinuity while A stays contiguous.
    for f in ["open", "high", "low", "close", "volume"]:
        df.loc[idx[91:199], ("B", f)] = float("nan")
    return df


def test_gappy_symbol_dropped_single_interval(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _gappy_df())
    frames = frames_from_feed(["A", "B"], _ts("2024-01-01"), _ts("2024-01-06"), "1h")
    # B's row gaps exceed DEFAULT_MAX_GAP -> dropped, A survives.
    assert [s for s, _ in frames] == ["A"]


def test_gappy_symbol_dropped_multi_interval(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _gappy_df())
    frames = frames_by_interval(
        ["A", "B"], _ts("2024-01-01"), _ts("2024-01-06"), ["1h", "1d"]
    )
    for iv, fr in frames.items():
        assert [s for s, _ in fr] == ["A"]


def test_state_per_interval_skips_gappy_symbol(monkeypatch):
    import src.bt.screen.adapter as adapter

    monkeypatch.setattr(adapter, "load_candles", lambda *a, **k: _gappy_df())
    states = state_per_interval(
        ["A", "B"], _ts("2024-01-01"), _ts("2024-01-06"), ["1h"]
    )
    assert set(states.keys()) == {"1h"}
    st = states["1h"]
    assert st.frame("A") is not None
    assert st.frame("B") is None
