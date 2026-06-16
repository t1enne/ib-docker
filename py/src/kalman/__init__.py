"""Kalman filter module — price smoothing, trend estimation, pairs trading.

Exports
-------
Univariate filter (batch):
    KalmanConfig, KalmanStats, FilterResult
    run_filter, compute_stats     — from src.kalman.pure

Pairs-trading filter (batch):
    PairsKalmanConfig, PairsKalmanResult
    run_pairs_kalman              — from src.kalman.pure

Pairs-trading filter (online / backtest hot path):
    PairsKalmanOnline             — from src.kalman.online

CLI entry-point:
    kalman                        — from src.kalman.cli
"""

from __future__ import annotations

from src.kalman.pure import run_filter, compute_stats, run_pairs_kalman
from src.kalman.online import PairsKalmanOnline
from src.kalman.types import (
    KalmanConfig,
    KalmanStats,
    FilterResult,
    PairsKalmanConfig,
    PairsKalmanResult,
)
from src.kalman.cli import kalman

__all__ = [
    "run_filter",
    "compute_stats",
    "run_pairs_kalman",
    "PairsKalmanOnline",
    "kalman",
    "KalmanConfig",
    "KalmanStats",
    "FilterResult",
    "PairsKalmanConfig",
    "PairsKalmanResult",
]
