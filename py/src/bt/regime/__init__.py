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

Model updaters
--------------
create_regime_model_updater   — single detector → current_regime
create_dual_online_updater    — SMA trend + online HMM vol → current_trend + current_vol
create_hmm_online_updater     — online HMM → current_regime
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
from src.bt.regime.model_updater import (
    create_regime_model_updater,
    create_hmm_online_updater,
    create_dual_online_updater,
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
    # Updaters
    "create_regime_model_updater",
    "create_hmm_online_updater",
    "create_dual_online_updater",
]
