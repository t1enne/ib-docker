"""Kalman filter module — price smoothing, trend estimation, pairs trading.

Exports
-------
Univariate filter (batch):
    KalmanConfig, KalmanStats, FilterResult
    run_filter, compute_stats     — from src.indicators.kalman.pure

Pairs-trading filter (batch):
    PairsKalmanConfig, PairsKalmanResult
    run_pairs_kalman              — from src.indicators.kalman.pure

Pairs-trading filter (online / backtest hot path):
    PairsKalmanOnline             — from src.indicators.kalman.online
"""

from __future__ import annotations

from src.indicators.kalman.pure import run_filter, compute_stats, run_pairs_kalman
from src.indicators.kalman.online import PairsKalmanOnline
from src.indicators.kalman.types import (
    KalmanConfig,
    KalmanStats,
    FilterResult,
    PairsKalmanConfig,
    PairsKalmanResult,
)

__all__ = [
    "run_filter",
    "compute_stats",
    "run_pairs_kalman",
    "PairsKalmanOnline",
    "KalmanConfig",
    "KalmanStats",
    "FilterResult",
    "PairsKalmanConfig",
    "PairsKalmanResult",
]
