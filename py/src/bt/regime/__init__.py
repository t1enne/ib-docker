"""Pluggable regime detection for strategies.

Two regime dimensions:

  Trend  — market direction (BULL/BEAR/RANGE) via SMA cross
  Vol    — volatility level (LOW_VOL/MED_VOL/HIGH_VOL) via HMM

Protocol
--------
RegimeDetector  — (prices: pd.DataFrame) -> pd.Series[int]

Built-in detectors
------------------
create_sma_detector          — price vs SMA cross (trend)
create_volatility_detector   — rolling vol percentile (trend)
create_hmm_vol_detector      — HMM vol-ranked (vol)

Gates (strategy-facing)
-----------------------
TrendGate            — typed BULL/BEAR/RANGE with allows_long/short + hostile_to
sma_trend            — SMA fast/slow crossover → TrendGate
above_sma            — last close vs N-SMA (bool)
series_above_sma     — stateless close series vs N-SMA (bool)
weekly_above_sma     — weekly close vs weekly N-SMA (structural trend, bool)

State ownership
---------------
Regimes are strategy-owned. Instead of an engine ``model_updater`` writing
``ModelState`` fields, a strategy holds its own model object in
``ctx.shared`` and reads its signal inline — e.g.
``src.indicators.hmm.strategy.OnlineRegime`` for a per-symbol online HMM, or
the stateless ``sma_trend``/``weekly_above_sma`` gates computed from
``state.candles``.
"""

from src.bt.regime.types import (
    RegimeDetector,
    TREND_INT_TO_LABEL,
    TREND_LABEL_TO_INT,
    VOL_INT_TO_LABEL,
    VOL_LABEL_TO_INT,
    TrendRegime,
    VolRegime,
)
from src.bt.regime.detectors import (
    create_hmm_vol_detector,
    create_sma_detector,
    create_volatility_detector,
    current_trend_label,
    current_vol_label,
)
from src.bt.regime.gates import (
    TrendGate,
    sma_trend,
    above_sma,
    series_above_sma,
    weekly_above_sma,
)

__all__ = [
    # Types
    "RegimeDetector",
    "TrendRegime",
    "VolRegime",
    "TREND_INT_TO_LABEL",
    "TREND_LABEL_TO_INT",
    "VOL_INT_TO_LABEL",
    "VOL_LABEL_TO_INT",
    # Detectors
    "create_hmm_vol_detector",
    "create_sma_detector",
    "create_volatility_detector",
    "current_trend_label",
    "current_vol_label",
    # Gates
    "TrendGate",
    "sma_trend",
    "above_sma",
    "series_above_sma",
    "weekly_above_sma",
]
