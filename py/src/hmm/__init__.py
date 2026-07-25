"""HMM module — Hidden Markov Model for market regime detection.

Exports
-------
MarketRegimeHMM     — model class (from src.hmm.hmm)
RegimeStats         — result types (from src.hmm.types)
create_regime_features — feature builder (from src.hmm.hmm)
"""

from src.hmm.hmm import MarketRegimeHMM, create_regime_features
from src.hmm.types import RegimeStats

__all__ = ["MarketRegimeHMM", "RegimeStats", "create_regime_features"]
