"""Tests for the online (accumulating) Volume Profile.

The core invariant: feeding candles one-by-one through ``observe`` yields the
*exact* same POC / VAH / VAL / histogram as the batch ``volume_profile`` over
the corresponding rolling window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.volume_profile.online import OnlineVolumeProfile
from src.indicators.volume_profile.pure import volume_profile


def _synthetic(n: int, run: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(run)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = mid + rng.uniform(0.2, 1.0, n)
    low = mid - rng.uniform(0.2, 1.0, n)
    vol = rng.uniform(100, 1000, n)
    return high, low, vol


def _batch_over(high, low, vol, num_bins=50, va=0.70):
    """Batch volume_profile over the full arrays."""
    return volume_profile(
        pd.Series(high),
        pd.Series(low),
        pd.Series(vol),
        num_bins=num_bins,
        value_area_pct=va,
    )


@pytest.mark.parametrize("window", [None, 200])
def test_incremental_matches_batch_full_history(window):
    high, low, vol = _synthetic(120)
    num_bins, va = 40, 0.70
    acc = OnlineVolumeProfile(
        num_bins=num_bins, value_area_pct=va, window=window, warmup_bars=5
    )
    for h, lo, v in zip(high, low, vol):
        acc.observe(float(lo), float(h), float(v))

    assert acc.ready
    batch = _batch_over(high, low, vol, num_bins, va)
    snap = acc.snapshot()
    np.testing.assert_allclose(snap.volume, batch.volume, atol=1e-6)
    assert snap.poc == pytest.approx(batch.poc, rel=1e-6)
    assert snap.vah == pytest.approx(batch.vah, rel=1e-6)
    assert snap.val == pytest.approx(batch.val, rel=1e-6)
    assert snap.total_volume == pytest.approx(batch.total_volume, rel=1e-9)


def test_incremental_matches_batch_rolling_window():
    """With a rolling window, only the last `window` candles are profiled."""
    high, low, vol = _synthetic(300)
    window, num_bins, va = 150, 40, 0.70
    acc = OnlineVolumeProfile(
        num_bins=num_bins, value_area_pct=va, window=window, warmup_bars=5
    )
    for h, lo, v in zip(high, low, vol):
        acc.observe(float(lo), float(h), float(v))

    batch = _batch_over(high[-window:], low[-window:], vol[-window:], num_bins, va)
    snap = acc.snapshot()
    np.testing.assert_allclose(snap.volume, batch.volume, atol=1e-6)
    assert snap.poc == pytest.approx(batch.poc, rel=1e-6)
    assert snap.vah == pytest.approx(batch.vah, rel=1e-6)
    assert snap.val == pytest.approx(batch.val, rel=1e-6)
    assert snap.n_candles == window


def test_rolling_window_evicts_oldest_exactly():
    """After the window fills, each candle aligns with the batch over the tail."""
    high, low, vol = _synthetic(80)
    window, num_bins, va = 30, 30, 0.75
    acc = OnlineVolumeProfile(
        num_bins=num_bins, value_area_pct=va, window=window, warmup_bars=3
    )
    for i, (h, lo, v) in enumerate(zip(high, low, vol)):
        acc.observe(float(lo), float(h), float(v))
        k = acc.n_candles
        # At step i the accumulator holds candles [i-k+1 .. i], which for a
        # full window is the last `window` candles observed so far.
        start = i + 1 - k
        batch = _batch_over(
            high[start : i + 1], low[start : i + 1], vol[start : i + 1], num_bins, va
        )
        snap = acc.snapshot()
        np.testing.assert_allclose(snap.volume, batch.volume, atol=1e-5)
        if snap.ready:
            assert snap.poc == pytest.approx(batch.poc, rel=1e-6)


def test_warmup_controls_ready():
    acc = OnlineVolumeProfile(window=None, warmup_bars=4)
    high, low, vol = _synthetic(10)
    statuses = []
    for h, lo, v in zip(high, low, vol):
        snap = acc.observe(float(lo), float(h), float(v))
        statuses.append(snap.ready)
    # First three observations not ready; from the 4th onward ready.
    assert statuses[:3] == [False, False, False]
    assert all(statuses[3:])


def test_invalid_candles_skipped_but_counted():
    acc = OnlineVolumeProfile(window=None, warmup_bars=1)
    high, low, vol = _synthetic(5)
    # Feed: valid, NaN low, high<low, zero vol, valid
    for i in range(5):
        if i == 1:
            acc.observe(float("nan"), float(high[1]), float(vol[1]))
        elif i == 2:
            acc.observe(float(high[2] + 1.0), float(high[2]), float(vol[2]))
        elif i == 3:
            acc.observe(float(low[3]), float(high[3]), 0.0)
        else:
            acc.observe(float(low[i]), float(high[i]), float(vol[i]))
    snap = acc.snapshot()
    # Two valid candles counted in the profile.
    assert snap.n_candles == 2
    assert acc.n_observed == 5
    assert snap.ready


def test_reset_clears_state():
    high, low, vol = _synthetic(20)
    acc = OnlineVolumeProfile(window=None, warmup_bars=2)
    for h, lo, v in zip(high, low, vol):
        acc.observe(float(lo), float(h), float(v))
    assert acc.n_observed == 20
    acc.reset()
    assert acc.n_observed == 0
    assert acc.n_candles == 0
    assert acc.range is None


def test_snapshot_does_not_mutate():
    high, low, vol = _synthetic(30)
    acc = OnlineVolumeProfile(window=None, warmup_bars=2)
    for h, lo, v in zip(high, low, vol):
        acc.observe(float(lo), float(h), float(v))
    before = acc.snapshot().poc
    after = acc.snapshot().poc
    assert before == after
    assert acc.n_observed == 30  # snapshot appended nothing


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        OnlineVolumeProfile(num_bins=0)
    with pytest.raises(ValueError):
        OnlineVolumeProfile(value_area_pct=0.0)
    with pytest.raises(ValueError):
        OnlineVolumeProfile(value_area_pct=1.5)
    with pytest.raises(ValueError):
        OnlineVolumeProfile(window=0)


def test_flat_price_level_single_bin():
    """All candles at one price -> one degenerate bin, no crash."""
    acc = OnlineVolumeProfile(window=None, warmup_bars=1)
    for _ in range(5):
        snap = acc.observe(10.0, 10.0, 100.0)
    assert snap.n_candles == 5
    assert snap.poc == pytest.approx(10.0, abs=1e-3)
    assert snap.total_volume == pytest.approx(500.0)
