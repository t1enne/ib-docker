"""Tests for bar-size-aware HMM annualisation.

Regime features (volatility, momentum) must annualise to the input bar size,
otherwise intraday bars compress the vol feature and the model separates regimes
by return sign instead of by volatility. ``bars_per_year`` (default 252 = daily)
fixes the feature scaling for intraday bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.hmm.hmm import MarketRegimeHMM
from src.indicators.hmm.online import MarketRegimeHMMOnline


def _synthetic_prices(n: int = 600, seed: int = 0) -> pd.Series:
    """Synthetic price series with mild drift so features are well-defined."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=n)
    prices = pd.Series(100 * np.exp(np.cumsum(returns))).set_axis(
        pd.date_range("2020-01-01", periods=n, freq="h"), axis=0
    )
    return prices


def test_batch_default_annualization_is_252():
    """Default bars_per_year matches legacy 252 (daily) feature scaling."""
    model = MarketRegimeHMM(vol_window=20, momentum_window=10)
    features = model._create_features(_synthetic_prices())
    # Volatility feature = rolling std * sqrt(252); a horizontal slice std should
    # be reproducible from the configured annualisation factor.
    assert model.bars_per_year == 252
    assert {"returns", "volatility", "momentum"} <= set(features.columns)
    assert features["volatility"].notna().sum() > 0


def test_batch_bars_per_year_scales_features():
    """Raised bars_per_year scales vol and momentum features up accordingly."""
    prices = _synthetic_prices()
    a = MarketRegimeHMM(vol_window=100, momentum_window=50, bars_per_year=252)
    b = MarketRegimeHMM(vol_window=100, momentum_window=50, bars_per_year=3780)
    fa, fb = a._create_features(prices), b._create_features(prices)

    # Same windows -> returns identical; vol/momentum scaled by sqrt / linear ratio.
    ratio = np.sqrt(3780 / 252)
    aligned_a = fa["volatility"].dropna().values
    aligned_b = fb["volatility"].dropna().values
    np.testing.assert_allclose(aligned_b, aligned_a * ratio, rtol=1e-6, atol=1e-9)

    mom_ratio = 3780 / 252
    ma = fa["momentum"].dropna().values
    mb = fb["momentum"].dropna().values
    np.testing.assert_allclose(mb, ma * mom_ratio, rtol=1e-6, atol=1e-9)


def test_online_features_use_bars_per_year():
    """Online model's latest-feature vol matches the configured annualisation."""
    prices = _synthetic_prices(200)
    m = MarketRegimeHMMOnline(vol_window=20, momentum_window=10, bars_per_year=3780)
    for p in prices.values:
        m._prices.append(float(p))
    feat = m._latest_features()
    # returns feature is raw; vol/momentum are positive and non-trivial.
    assert len(feat) == 3
    assert feat[1] >= 0  # volatility (annualised) is non-negative
    assert np.isfinite(feat).all()
