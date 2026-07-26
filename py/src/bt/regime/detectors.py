"""Built-in regime detectors.

Each factory returns a RegimeDetector callable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.bt.regime.types import (
    REGIME_INT_TO_LABEL,
    REGIME_LABEL_TO_INT,
    RegimeDetector,
    RegimeLabel,
)


# ---------------------------------------------------------------------------
# HMM-based detector (wraps existing MarketRegimeHMM)
# ---------------------------------------------------------------------------


def create_hmm_detector(
    price_col: str = "close",
    n_regimes: int = 3,
    vol_window: int = 20,
    momentum_window: int = 10,
    min_train_size: int = 252,
    retrain_interval: int = 50,
) -> RegimeDetector:
    """Create an HMM-based regime detector.

    Uses src.indicators.hmm.MarketRegimeHMM under the hood.
    Caches the fitted model and only refits every retrain_interval bars
    to avoid expensive HMM fitting on every tick.

    HMM states are reordered by volatility: 0=lowest vol, N-1=highest vol.
    These map to the canonical convention: 0=RANGE, 1=BULL, 2=BEAR.
    """
    from src.indicators.hmm import MarketRegimeHMM

    _model: MarketRegimeHMM | None = None
    _last_len: int = 0

    def detect(prices: pd.DataFrame) -> pd.Series:
        nonlocal _model, _last_len

        if price_col not in prices.columns or len(prices) < min_train_size:
            return pd.Series(np.full(len(prices), -1), index=prices.index)

        n = len(prices)
        # Only refit every retrain_interval bars or on first call
        if _model is None or n - _last_len >= retrain_interval:
            _model = MarketRegimeHMM(
                n_regimes=n_regimes,
                vol_window=vol_window,
                momentum_window=momentum_window,
                min_train_size=min_train_size,
                random_state=42,
            )
            _model.fit(prices[price_col])
            _last_len = n

        regimes = _model.predict(prices[price_col])
        return regimes.fillna(-1).astype(int).rename("regime")

    return detect


# ---------------------------------------------------------------------------
# SMA-based directional detector
# ---------------------------------------------------------------------------


def create_sma_detector(
    price_col: str = "close",
    fast_window: int = 50,
    slow_window: int = 200,
    range_threshold_pct: float = 0.005,
) -> RegimeDetector:
    """Detect regime via SMA cross + flatness check.

    - BULL: fast SMA > slow SMA and spread > threshold
    - BEAR: fast SMA < slow SMA and spread > threshold
    - RANGE: spread within threshold band (sideways)
    """

    def detect(prices: pd.DataFrame) -> pd.Series:
        closes = prices[price_col]
        if len(closes) < slow_window:
            return pd.Series(np.full(len(closes), -1), index=closes.index)

        fast_sma = closes.rolling(fast_window).mean()
        slow_sma = closes.rolling(slow_window).mean()
        spread_pct = (fast_sma - slow_sma).abs() / slow_sma

        regime_int = np.full(len(closes), -1)
        for i in range(slow_window - 1, len(closes)):
            if pd.isna(spread_pct.iloc[i]):
                continue
            if spread_pct.iloc[i] <= range_threshold_pct:
                regime_int[i] = REGIME_LABEL_TO_INT["RANGE"]
            elif fast_sma.iloc[i] > slow_sma.iloc[i]:
                regime_int[i] = REGIME_LABEL_TO_INT["BULL"]
            else:
                regime_int[i] = REGIME_LABEL_TO_INT["BEAR"]

        return pd.Series(regime_int, index=closes.index, name="regime")

    return detect


# ---------------------------------------------------------------------------
# Volatility-band detector
# ---------------------------------------------------------------------------


def create_volatility_detector(
    price_col: str = "close",
    vol_window: int = 20,
    low_vol_pctile: float = 0.25,
    high_vol_pctile: float = 0.75,
    direction_window: int = 50,
) -> RegimeDetector:
    """Detect regime via rolling volatility percentiles + trend.

    - RANGE: volatility in bottom 25% of historical range
    - BULL: rising trend (price > SMA) with moderate vol
    - BEAR: falling trend (price < SMA) with high vol
    """

    def detect(prices: pd.DataFrame) -> pd.Series:
        closes = prices[price_col]
        if len(closes) < max(vol_window, direction_window):
            return pd.Series(np.full(len(closes), -1), index=closes.index)

        returns = closes.pct_change()
        rolling_vol = returns.rolling(vol_window).std()

        # Expanding percentiles to avoid lookahead
        vol_low = rolling_vol.expanding().quantile(low_vol_pctile)
        vol_high = rolling_vol.expanding().quantile(high_vol_pctile)

        sma = closes.rolling(direction_window).mean()
        trend_up = closes > sma

        regime_int = np.full(len(closes), -1)
        for i in range(1, len(closes)):
            v = rolling_vol.iloc[i]
            vl = vol_low.iloc[i]
            vh = vol_high.iloc[i]
            if pd.isna(v) or pd.isna(vl) or pd.isna(vh):
                continue

            if v <= vl:
                regime_int[i] = REGIME_LABEL_TO_INT["RANGE"]
            elif v > vh and not trend_up.iloc[i]:
                regime_int[i] = REGIME_LABEL_TO_INT["BEAR"]
            else:
                regime_int[i] = REGIME_LABEL_TO_INT["BULL"]

        return pd.Series(regime_int, index=closes.index, name="regime")

    return detect


# ---------------------------------------------------------------------------
# Utility: get current regime for a single point
# ---------------------------------------------------------------------------


def current_regime_label(
    regime_series: pd.Series, idx: int = -1
) -> Optional[RegimeLabel]:
    """Extract the current regime label from a regime series.

    Args:
        regime_series: Output of a RegimeDetector
        idx: Index position to query (default: -1, last)

    Returns:
        "BULL", "BEAR", "RANGE", or None if unknown
    """
    if len(regime_series) == 0:
        return None
    val = regime_series.iloc[idx]
    if pd.isna(val) or val == -1:
        return None
    return REGIME_INT_TO_LABEL.get(int(val))
