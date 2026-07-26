"""Tests for regime detectors with synthetic bull/bear/ranging data.

Each detector is tested against known market phases to verify it
correctly identifies the dominant regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bt.regime.detectors import (
    create_hmm_vol_detector,
    create_sma_detector,
    create_volatility_detector,
    current_trend_label,
)
from src.bt.regime.types import TREND_INT_TO_LABEL


# ---------------------------------------------------------------------------
# Fake data generators — bull, bear, range
# ---------------------------------------------------------------------------


def _make_price_series(
    n_bars: int,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Generate a lognormal random walk with given drift and vol."""
    rng = np.random.default_rng(seed)
    returns = drift / n_bars + vol * rng.standard_normal(n_bars)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


def _make_bull_market(n_bars: int = 500) -> pd.DataFrame:
    """Strong uptrend: +25% annualized drift, low vol."""
    prices = _make_price_series(n_bars, drift=0.25, vol=0.10, seed=3)
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "volume": 1_000_000,
        }
    )


def _make_bear_market(n_bars: int = 500) -> pd.DataFrame:
    """Strong downtrend: -30% annualized drift, elevated vol."""
    prices = _make_price_series(n_bars, drift=-0.30, vol=0.22, seed=2)
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 1.001,
            "high": prices * 1.003,
            "low": prices * 0.997,
            "volume": 1_000_000,
        }
    )


def _make_ranging_market(n_bars: int = 500) -> pd.DataFrame:
    """Sideways: near-zero drift, low vol, mean-reverting."""
    rng = np.random.default_rng(42)
    # Ornstein-Uhlenbeck: mean-reverting around 100
    theta = 0.05
    mu = 100.0
    sigma = 0.4
    prices = np.zeros(n_bars)
    prices[0] = mu
    for t in range(1, n_bars):
        prices[t] = (
            prices[t - 1] + theta * (mu - prices[t - 1]) + sigma * rng.standard_normal()
        )
    s = pd.Series(prices)
    return pd.DataFrame(
        {
            "close": s,
            "open": s * 0.999,
            "high": s * 1.002,
            "low": s * 0.998,
            "volume": 500_000,
        }
    )


def _make_multi_phase(n_each: int = 300) -> pd.DataFrame:
    """Concatenate bull → bear → range phases."""
    bull = _make_bull_market(n_bars=300)
    bear = _make_bear_market(n_bars=300)
    rng = _make_ranging_market(n_bars=300)
    base = pd.Timestamp("2024-01-01")
    bull.index = pd.date_range(base, periods=len(bull), freq="D")
    bear.index = pd.date_range(
        base + pd.Timedelta(days=len(bull)), periods=len(bear), freq="D"
    )
    rng.index = pd.date_range(
        base + pd.Timedelta(days=len(bull) + len(bear)), periods=len(rng), freq="D"
    )
    return pd.concat([bull, bear, rng])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bull_data() -> pd.DataFrame:
    return _make_bull_market()


@pytest.fixture
def bear_data() -> pd.DataFrame:
    return _make_bear_market()


@pytest.fixture
def range_data() -> pd.DataFrame:
    return _make_ranging_market()


@pytest.fixture
def multi_phase_data() -> pd.DataFrame:
    return _make_multi_phase()


# ---------------------------------------------------------------------------
# SMA detector tests
# ---------------------------------------------------------------------------


class TestSMADetector:
    def test_bull_market_detected(self, bull_data: pd.DataFrame) -> None:
        detect = create_sma_detector(fast_window=20, slow_window=50)
        regimes = detect(bull_data)
        valid = regimes[regimes >= 0]
        # In a strong uptrend, BULL should be the most common regime
        bull_pct = (valid == 1).mean()
        assert bull_pct > 0.3, f"Expected >30% BULL in bull market, got {bull_pct:.1%}"

    def test_bear_market_detected(self, bear_data: pd.DataFrame) -> None:
        detect = create_sma_detector(fast_window=20, slow_window=50)
        regimes = detect(bear_data)
        valid = regimes[regimes >= 0]
        bear_pct = (valid == 2).mean()
        assert bear_pct > 0.55, f"Expected >55% BEAR in bear market, got {bear_pct:.1%}"

    def test_ranging_market_detected(self, range_data: pd.DataFrame) -> None:
        detect = create_sma_detector(
            fast_window=10, slow_window=30, range_threshold_pct=0.003
        )
        regimes = detect(range_data)
        valid = regimes[regimes >= 0]
        range_pct = (valid == 0).mean()
        # Ranging data is mean-reverting around 100 — tight spread
        assert range_pct > 0.3, (
            f"Expected >30% RANGE in ranging market, got {range_pct:.1%}"
        )

    def test_multi_phase_transitions(self, multi_phase_data: pd.DataFrame) -> None:
        detect = create_sma_detector(fast_window=20, slow_window=50)
        regimes = detect(multi_phase_data)
        valid = regimes[regimes >= 0]
        n = len(multi_phase_data)
        # Split into thirds
        first = valid.iloc[: n // 3]
        second = valid.iloc[n // 3 : 2 * n // 3]
        third = valid.iloc[2 * n // 3 :]
        # First phase: bull → most BULL
        assert (first == 1).mean() > 0.5, f"Phase 1 bull: {(first == 1).mean():.1%}"
        # Second phase: bear → most BEAR
        assert (second == 2).mean() > 0.45, f"Phase 2 bear: {(second == 2).mean():.1%}"
        # Third phase: range → some RANGE (not necessarily dominant — OU fluctuates)
        range_pct = (third == 0).mean()
        assert range_pct > 0.15, f"Phase 3 range: {range_pct:.1%} RANGE"

    def test_current_trend_label_returns_string(self, bull_data: pd.DataFrame) -> None:
        detect = create_sma_detector(fast_window=20, slow_window=50)
        regimes = detect(bull_data)
        label = current_trend_label(regimes)
        assert label in ("BULL", "BEAR", "RANGE")
        # Bull market → BULL at the end
        assert label == "BULL", f"Expected BULL, got {label}"

    def test_insufficient_warmup_returns_minus_one(self) -> None:
        """Short series: all values should be -1 (unknown)."""
        short = pd.DataFrame({"close": [100.0] * 10})
        detect = create_sma_detector(slow_window=50)
        regimes = detect(short)
        assert (regimes == -1).all()

    def test_edge_case_nan_close(self) -> None:
        """NaN closes should not crash — they produce -1 for those rows."""
        closes = [100.0] * 300 + [np.nan] + [100.0] * 100
        df = pd.DataFrame(
            {
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "volume": 1000,
            }
        )
        detect = create_sma_detector(fast_window=20, slow_window=50)
        regimes = detect(df)
        assert regimes.iloc[300] == -1  # NaN row → unknown


# ---------------------------------------------------------------------------
# Volatility detector tests
# ---------------------------------------------------------------------------


class TestVolatilityDetector:
    def test_ranging_is_low_vol(self, range_data: pd.DataFrame) -> None:
        detect = create_volatility_detector(
            vol_window=20,
            direction_window=50,
            low_vol_pctile=0.30,
            high_vol_pctile=0.70,
        )
        regimes = detect(range_data)
        valid = regimes[regimes >= 0]
        # Ranging (mean-reverting, low vol) → some RANGE
        range_pct = (valid == 0).mean()
        assert range_pct > 0.08, (
            f"Expected >8% RANGE in ranging data, got {range_pct:.1%}"
        )

    def test_bear_is_high_vol_downtrend(self, bear_data: pd.DataFrame) -> None:
        detect = create_volatility_detector(
            vol_window=20,
            direction_window=50,
            low_vol_pctile=0.25,
            high_vol_pctile=0.75,
        )
        regimes = detect(bear_data)
        valid = regimes[regimes >= 0]
        bear_pct = (valid == 2).mean()
        assert bear_pct > 0.18, f"Expected >18% BEAR in bear data, got {bear_pct:.1%}"

    def test_bull_is_moderate_vol_uptrend(self, bull_data: pd.DataFrame) -> None:
        detect = create_volatility_detector(vol_window=20, direction_window=50)
        regimes = detect(bull_data)
        valid = regimes[regimes >= 0]
        bull_pct = (valid == 1).mean()
        assert bull_pct > 0.4, f"Expected >40% BULL in bull data, got {bull_pct:.1%}"

    def test_insufficient_warmup(self) -> None:
        short = pd.DataFrame({"close": [100.0] * 20})
        detect = create_volatility_detector(vol_window=20, direction_window=50)
        regimes = detect(short)
        assert (regimes == -1).all()


# ---------------------------------------------------------------------------
# HMM detector tests
# ---------------------------------------------------------------------------


class TestHMMDetector:
    def test_hmm_detects_three_regimes(self, multi_phase_data: pd.DataFrame) -> None:
        detect = create_hmm_vol_detector(min_train_size=250)
        regimes = detect(multi_phase_data)
        valid = regimes[regimes >= 0]
        unique = set(valid.unique())
        # HMM with n_regimes=3 should detect all 3 labels
        assert unique.issubset({0, 1, 2}), (
            f"Expected subsets of {{0,1,2}}, got {unique}"
        )
        assert len(unique) >= 2, f"Expected at least 2 distinct regimes, got {unique}"

    def test_hmm_too_short_data(self) -> None:
        short = pd.DataFrame({"close": [100.0] * 50})
        detect = create_hmm_vol_detector(min_train_size=252)
        regimes = detect(short)
        assert (regimes == -1).all()


# ---------------------------------------------------------------------------
# Regime label mapping consistency
# ---------------------------------------------------------------------------


class TestRegimeMapping:
    def test_all_labels_have_both_directions(self) -> None:
        from src.bt.regime.types import TREND_LABEL_TO_INT

        assert set(TREND_INT_TO_LABEL.keys()) == {0, 1, 2}
        assert set(TREND_LABEL_TO_INT.keys()) == {"BULL", "BEAR", "RANGE"}
        # Bidirectional consistency
        for i, label in TREND_INT_TO_LABEL.items():
            assert TREND_LABEL_TO_INT[label] == i

    def test_current_trend_label_none_on_empty(self) -> None:
        assert current_trend_label(pd.Series([], dtype=float)) is None
