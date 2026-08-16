"""Type definitions for the adaptive entropy trend indicator.

Frozen dataclasses follow the repo convention of immutable, fully-annotated
state.  Bands are the primary trading surface (mirroring the Pine Script
$close/band crossings); ``trend`` is the quantised direction used for
bar-colouring and regime gating.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveEntropyConfig:
    """Configuration for the adaptive entropy trend.

    Matches the Pine ``adaptive_entropy.pine`` settings.
    """

    lookback: int = 25
    """Lookback used to sample log returns for Shannon entropy + ATR (>= 5)."""
    num_bins: int = 10
    """Number of histogram bins for the log-return distribution."""
    fast_multiplier: float = 1.8
    """ATR multiplier for the inner (trend-trigger) band."""
    slow_multiplier: float = 3.5
    """ATR multiplier for the outer (envelope) band."""

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError(f"lookback must be >= 5, got {self.lookback}")
        if self.num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {self.num_bins}")
        if self.fast_multiplier <= 0:
            raise ValueError(f"fast_multiplier must be > 0, got {self.fast_multiplier}")
        if self.slow_multiplier <= 0:
            raise ValueError(f"slow_multiplier must be > 0, got {self.slow_multiplier}")


@dataclass(frozen=True)
class AdaptiveEntropyResult:
    """Row of adaptive entropy indicator outputs for one bar.

    ``entropy`` and ``trend_strength`` expose the model's raw view of market
    structure (low entropy = directional/trending, high entropy = choppy),
    while the bands quantise that into a trading signal on close crosses.
    """

    close: float
    """Bar close (mirrors the input)."""
    entropy: float
    """Normalized Shannon entropy of the log-return distribution in [0, 1]."""
    normalized_entropy: float
    """Alias of ``entropy`` kept for parity with the Pine naming."""
    trend_strength: float
    """``1 - entropy``, the (soft) directional conviction in [0, 1]."""
    adaptive_ema: float
    """Entropy-adaptive exponential moving average of close."""
    atr: float
    """Average true range (Wilder) over the lookback window."""
    fast_band_width: float
    """``atr * fast_multiplier * (0.5 + trend_strength)``."""
    slow_band_width: float
    """``atr * slow_multiplier * (0.5 + trend_strength)``."""
    inner_upper: float
    """``adaptive_ema + fast_band_width`` — bullish trend trigger."""
    inner_lower: float
    """``adaptive_ema - fast_band_width`` — bearish trend trigger."""
    outer_upper: float
    """``adaptive_ema + slow_band_width`` — high-conviction bullish envelope."""
    outer_lower: float
    """``adaptive_ema - slow_band_width`` — high-conviction bearish envelope."""
    trend: int
    """Quantised direction: 1 (bullish), -1 (bearish), else 0."""
