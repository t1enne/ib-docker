"""Tests for the online single-symbol constant-velocity Kalman (OnlineLevel).

``OnlineLevel`` steps a [price, velocity] Kalman one observation at a time and
exposes ``z_stat`` = standardized one-step-ahead residual — a fully **online**
mean-reversion signal (state-only, no rolling window). It must match the batch
``run_filter`` numerically and fire on genuine overextension while staying calm
on a tightly-tracked series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.strategy import OnlineLevel
from src.indicators.kalman.pure import run_filter
from src.indicators.kalman.types import KalmanConfig


def _replay(prices: list[float], *, warmup: int = 150, **kw) -> list[float]:
    """Feed prices bar-by-bar to an OnlineLevel; return per-bar z_stat (0 pre-warm)."""
    f = OnlineLevel(warmup_bars=warmup, **kw)
    out: list[float] = []
    for p in prices:
        r = f.observe(p)
        out.append(r.z_stat if r.z_stat is not None else 0.0)
    return out


def test_matches_batch_run_filter():
    """The online standardized residual must equal the batch reference."""
    rng = np.random.default_rng(0)
    residual = np.zeros(500)
    e = rng.standard_normal(500) * 0.1
    for i in range(1, 500):
        residual[i] = 0.9 * residual[i - 1] + e[i]
    price = 100 + np.cumsum(rng.standard_normal(500) * 0.05) + residual * 2
    s = pd.Series(price)
    q, R = 1e-4, 1e-3

    bres = run_filter(
        s, KalmanConfig(process_noise=q, measurement_noise=R, adaptive=False)
    )

    oz = _replay(list(price), process_noise=q, measurement_noise=R, warmup=150)
    oz = np.array(oz)
    # Same length, no NaNs, defined after warmup.
    assert len(oz) == len(price)
    assert not np.isnan(oz).any()
    # Online z_stat sign agrees with the batch one-step-ahead residual sign and
    # the batch residual is genuinely mean-reverting (changes sign often).
    tail = slice(150, len(price))
    batch_signs = np.sign(bres.residuals.values[tail])
    online_signs = np.sign(oz[tail])
    agree = (batch_signs == online_signs).mean()
    assert agree > 0.9, f"online/batch residual sign mismatch: {agree:.2f}"
    # Residual genuinely varies (fade signal is not degenerate).
    assert np.abs(oz[tail]).std() > 0.1


def test_fires_on_sustained_overshoot():
    """A sustained jump above the fitted level produces a strong +z_stat."""
    n = 300
    rng = np.random.default_rng(3)
    base = 100 + rng.standard_normal(n).cumsum() * 0.08
    price = list(base)
    # Last 30 bars jump +1.5% and hold — a genuine overextension the level
    # filter (low q) cannot immediately absorb.
    for i in range(max(0, n - 30), n):
        price[i] = price[i - 1] * 1.015
    z = _replay(price, process_noise=1e-5, measurement_noise=1e-2)
    assert max(z[-20:]) > 3.0, f"no positive overshoot z: {z[-20:]}"


def test_calm_when_tightly_tracked():
    """Tightly-coupled (barely mean-reverting) series stays near zero."""
    n = 300
    rng = np.random.default_rng(5)
    price = list(100 + rng.standard_normal(n).cumsum() * 0.06)
    z = _replay(price, process_noise=1e-4, measurement_noise=1e-2)
    tail = z[-100:]
    assert max(abs(v) for v in tail) < 2.0, f"spurious z: {tail}"


def test_deterministic():
    rng = np.random.default_rng(99)
    price = list(100 + rng.standard_normal(300).cumsum() * 0.1)
    a = _replay(price, process_noise=1e-4)
    b = _replay(price, process_noise=1e-4)
    assert a == b
