"""Tests for univariate Kalman filter — critical behaviours only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.pure import run_filter, compute_stats
from src.indicators.kalman.types import KalmanConfig


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_filtered_tracks_price():
    prices = _series([100.0 + i * 0.1 for i in range(100)])
    config = KalmanConfig(process_noise=1e-6, measurement_noise=1e-2)
    result = run_filter(prices, config)
    rmse = float(np.sqrt(((prices - result.filtered).dropna() ** 2).mean()))
    assert rmse < 5.0


def test_velocity_sign():
    up = _series([100.0 + i * 0.5 for i in range(200)])
    assert float(run_filter(up, KalmanConfig()).velocity.mean()) > 0

    down = _series([200.0 - i * 0.5 for i in range(200)])
    assert float(run_filter(down, KalmanConfig()).velocity.mean()) < 0


def test_deterministic():
    rng = np.random.default_rng(99)
    prices = pd.Series(100 + rng.standard_normal(100).cumsum())
    a = run_filter(prices, KalmanConfig())
    b = run_filter(prices, KalmanConfig())
    assert a.filtered.equals(b.filtered)
    assert a.kalman_gains.equals(b.kalman_gains)


def test_compute_stats():
    rng = np.random.default_rng(42)
    prices = pd.Series(100 + rng.standard_normal(100).cumsum())
    result = run_filter(prices, KalmanConfig())
    stats = compute_stats(prices, result)
    assert 0 <= stats.rmse < 1e6
    assert 0 <= stats.mae < 1e6
    assert 0.0 <= stats.coverage_95 <= 1.0
    assert stats.n_observations == 100
