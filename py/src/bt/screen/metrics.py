"""Shared technical metrics for every screen's ``model_features``.

Every screen merges these into each ``ScreenResult.model_features`` so output is
uniform across screens for the same symbol/bar:

  ema_50   EMA(50) of close   (price)
  ema_100  EMA(100) of close  (price)
  atr_14   ATR(14)            (price)
  rsi_14   RSI(14)            (0..100)
  hi_52w   trailing 52-week high of close (``NaT``-safe calendar rolling)
  lo_52w   trailing 52-week low  of close

Missing/inf window values are returned as ``float('nan')`` (never omitted) so
downstream code can rely on a fixed key set. These are diagnostics/context, not
scoring inputs — the per-screen score/action is unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_METRIC_KEYS = ("ema_50", "ema_100", "atr_14", "rsi_14", "hi_52w", "lo_52w")


def _ta_ema(closes: pd.Series, span: int) -> float:
    """Lazy ``ta.ema`` — avoid pulling ``src.indicators`` at bt bootstrap."""
    from src.indicators.ta import ema

    try:
        v = float(ema(closes, span).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ta_atr(frame: pd.DataFrame) -> float:
    """Lazy ``ta.atr`` over the frame's OHLC (falls back to a manual TR when
    a high/low column is missing, e.g. a close-only frame)."""
    closes = frame["close"]
    high = frame["high"] if "high" in frame.columns else closes
    low = frame["low"] if "low" in frame.columns else closes
    if len(high) < 1:
        return float("nan")
    from src.indicators.ta import atr

    try:
        v = float(atr(high, low, closes, window=14).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ta_rsi(closes: pd.Series) -> float:
    """Lazy ``ta.rsi`` (window 14)."""
    from src.indicators.ta import rsi

    try:
        v = float(rsi(closes, window=14).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def common_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute the common metric set for a symbol frame.

    Returns a flat dict keyed by ``COMMON_METRIC_KEYS``; each value is a float
    (``nan`` on missing data). ``hi_52w``/``lo_52w`` use calendar-aware rolling
    (365 days) over the timestamp index, correct on any bar interval.
    """
    closes = frame["close"] if "close" in frame.columns else None
    if closes is None or len(closes) == 0:
        return {k: float("nan") for k in COMMON_METRIC_KEYS}

    ema_50 = _ta_ema(closes, 50)
    ema_100 = _ta_ema(closes, 100)
    atr_14 = _ta_atr(frame)
    rsi_14 = _ta_rsi(closes)

    hi_52w = lo_52w = float("nan")
    if isinstance(closes.index, pd.DatetimeIndex) and len(closes) > 0:
        hi = closes.rolling("365D", min_periods=1).max()
        lo = closes.rolling("365D", min_periods=1).min()
        hi_52w = float(hi.iloc[-1]) if np.isfinite(hi.iloc[-1]) else float("nan")
        lo_52w = float(lo.iloc[-1]) if np.isfinite(lo.iloc[-1]) else float("nan")

    return {
        "ema_50": ema_50,
        "ema_100": ema_100,
        "atr_14": atr_14,
        "rsi_14": rsi_14,
        "hi_52w": hi_52w,
        "lo_52w": lo_52w,
    }


def with_common_metrics(
    features: dict[str, float], frame: pd.DataFrame
) -> dict[str, float]:
    """Merge the common metric set into a screen's feature dict.

    Screen-specific features win on key collisions; the common keys are always
    present (possibly ``nan``) so output stays uniform across screens.
    """
    out: dict[str, float] = dict(common_metrics(frame))
    out.update(features)
    return out
