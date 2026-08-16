"""Indicators — technical analysis, Kalman filters, HMM regime detection, and macro assets."""

from src.indicators.ta import (
    ema,
    sma,
    rsi,
    atr,
    bollinger_bands,
    macd,
    stochastic,
    momentum,
    volatility,
    vwma,
    obv,
    mfi,
    lsma,
    plus_di,
    minus_di,
    adx,
)

from src.indicators.volume_profile import (
    volume_profile,
    VolumeProfile,
    OnlineVolumeProfile,
    VolumeProfileSnapshot,
    OnlineVP,
)

from src.indicators.kalman.pure import run_filter, compute_stats, run_pairs_kalman
from src.indicators.kalman.online import PairsKalmanOnline
from src.indicators.kalman.strategy import OnlinePairs, OnlinePairsResult
from src.indicators.kalman.types import (
    KalmanConfig,
    KalmanStats,
    FilterResult,
    PairsKalmanConfig,
    PairsKalmanResult,
)

from src.indicators.hmm.hmm import MarketRegimeHMM, create_regime_features
from src.indicators.hmm.strategy import OnlineRegime, OnlineRegimeResult
from src.indicators.hmm.types import RegimeStats

from src.indicators.macro import deflated_log_prices, init_macro_indicator

from src.indicators.adaptive_entropy import (
    adaptive_entropy,
    OnlineAdaptiveEntropy,
    AdaptiveEntropyConfig,
    AdaptiveEntropyResult,
)

__all__ = [
    # TA
    "ema",
    "sma",
    "rsi",
    "atr",
    "bollinger_bands",
    "macd",
    "stochastic",
    "momentum",
    "volatility",
    "vwma",
    "obv",
    "mfi",
    "lsma",
    "plus_di",
    "minus_di",
    "adx",
    # Volume Profile
    "volume_profile",
    "VolumeProfile",
    "OnlineVolumeProfile",
    "VolumeProfileSnapshot",
    "OnlineVP",
    # Kalman
    "run_filter",
    "compute_stats",
    "run_pairs_kalman",
    "PairsKalmanOnline",
    "OnlinePairs",
    "OnlinePairsResult",
    "KalmanConfig",
    "KalmanStats",
    "FilterResult",
    "PairsKalmanConfig",
    "PairsKalmanResult",
    # HMM
    "MarketRegimeHMM",
    "RegimeStats",
    "create_regime_features",
    "OnlineRegime",
    "OnlineRegimeResult",
    # Macro assets
    "init_macro_indicator",
    "deflated_log_prices",
    # Adaptive entropy trend
    "adaptive_entropy",
    "OnlineAdaptiveEntropy",
    "AdaptiveEntropyConfig",
    "AdaptiveEntropyResult",
]
