"""Volume Profile indicator — volume-at-price distribution and levels.

Exports
-------
    volume_profile      — compute a fixed-range Volume Profile (pure function)
    VolumeProfile       — result dataclass (POC, Value Area, per-bin histogram)

Usage from a strategy::

    from src.indicators import volume_profile

    def on_candle(state, candle, params):
        df = state.candles.get((candle.symbol, candle.interval or "1h"))
        if df is None or len(df) < params.vp_lookback:
            return []
        window = df.iloc[-params.vp_lookback:]
        vp = volume_profile(window["high"], window["low"], window["volume"])
        close = window["close"].iloc[-1]
        if close > vp.vah:
            ...  # breakout above the value area
"""

from __future__ import annotations

from src.indicators.volume_profile.online import OnlineVolumeProfile
from src.indicators.volume_profile.pure import volume_profile
from src.indicators.volume_profile.strategy import OnlineVP
from src.indicators.volume_profile.types import VolumeProfile, VolumeProfileSnapshot

__all__ = [
    "volume_profile",
    "VolumeProfile",
    "OnlineVolumeProfile",
    "VolumeProfileSnapshot",
    "OnlineVP",
]
