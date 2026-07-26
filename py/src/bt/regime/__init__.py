"""Pluggable regime detection for strategies.

Protocol
--------
RegimeDetector  — (prices: pd.DataFrame) -> pd.Series[int]
RegimeLabel     — Literal["BULL", "BEAR", "RANGE"]

Built-in detectors
------------------
create_hmm_detector     — HMM on returns/vol/momentum
create_sma_detector     — price vs SMA cross
create_volatility_detector — rolling vol percentile bands
"""

from src.bt.regime.types import RegimeDetector, RegimeLabel
from src.bt.regime.detectors import (
    create_hmm_detector,
    create_sma_detector,
    create_volatility_detector,
)

from src.bt.regime.model_updater import (
    create_regime_model_updater,
    create_hmm_online_updater,
)

__all__ = [
    "RegimeDetector",
    "RegimeLabel",
    "create_hmm_detector",
    "create_sma_detector",
    "create_volatility_detector",
    "create_regime_model_updater",
    "create_hmm_online_updater",
]
