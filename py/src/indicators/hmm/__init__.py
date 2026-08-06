"""HMM module — Hidden Markov Model for market regime detection.

Exports
-------
MarketRegimeHMM         — batch model class
MarketRegimeHMMOnline   — online (step-by-step) model for backtest hot path
RegimeStats             — result types
create_regime_features  — feature builder

Strategy-level owner:
    OnlineRegime         — from src.indicators.hmm.strategy
"""

from src.indicators.hmm.hmm import MarketRegimeHMM, create_regime_features
from src.indicators.hmm.online import MarketRegimeHMMOnline
from src.indicators.hmm.strategy import OnlineRegime, OnlineRegimeResult
from src.indicators.hmm.types import RegimeStats

__all__ = [
    "MarketRegimeHMM",
    "MarketRegimeHMMOnline",
    "RegimeStats",
    "create_regime_features",
    "OnlineRegime",
    "OnlineRegimeResult",
]
