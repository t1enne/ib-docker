"""HMM module — Hidden Markov Model for market regime detection.

Exports
-------
MarketRegimeHMM     — model class (from src.indicators.hmm.hmm)
RegimeStats         — result types (from src.indicators.hmm.types)
create_regime_features — feature builder (from src.indicators.hmm.hmm)
"""

from src.indicators.hmm.hmm import MarketRegimeHMM, create_regime_features
from src.indicators.hmm.types import RegimeStats

__all__ = ["MarketRegimeHMM", "RegimeStats", "create_regime_features"]
