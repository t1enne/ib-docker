"""Shared risk-composite wave helpers (cursor-safe) for the regime strategies.

Discovery skips leading-underscore modules, so this is pure importable plumbing,
not a strategy. Mirrors scripts/streamlit_cycle.py constants + composite logic
(the post-SPY/TLT, 5-leg trimmed set) so every ``risk_wave_*`` strategy trades
off the exact dashboard verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Mirrored from scripts/streamlit_cycle.py (post SPY/TLT drop: 5 legs).
_COMPOSITE_LEGS: tuple[tuple[str, str, str], ...] = (
    ("HYG / TLT", "HYG", "TLT"),
    ("IEF / TLT", "IEF", "TLT"),
    ("IWM / SPY", "IWM", "SPY"),
    ("CPER / GLD", "CPER", "GLD"),
    ("XLF / XLU", "XLF", "XLU"),
)

_MIN_HISTORY_DAYS = 90
_STALE_TRADING_DAYS = 7
_HEALTH_Z_WINDOW = 126
_MIN_QUORUM_FRACTION = 0.30
_EWMA_SPAN = 7
_EWMA_MIN_PERIODS = 3


def closes_for_legs(ctx: Any) -> dict[str, np.ndarray]:
    """Cursor-safe close array for every ticker touched by the composite legs."""
    need = sorted({s for _l, n, d in _COMPOSITE_LEGS for s in (n, d)})
    out: dict[str, np.ndarray] = {}
    for sym in need:
        try:
            out[sym] = ctx.ohlcv(sym).close.to_array()
        except Exception:
            out[sym] = np.array([], dtype=np.float64)
    return out


def composite_smooth(close_map: dict[str, np.ndarray]) -> float | None:
    """Current EWMA-smoothed cross-leg mean-z (all inputs <= cursor => safe).

    Reproduces ``_leg_z_series`` -> ``composite_wave`` from the screen: each leg
    normalises on its own trailing HEALTH_Z_WINDOW mean/std (NaN head until
    MIN_HISTORY_DAYS and where dispersion ~ 0), is re-seated on the union row
    axis and forward-filled to the staleness limit, then the cross-leg mean is
    taken with quorum gating and EWMA(span 7) smoothed. None when no leg can
    speak or quorum is unmet (no regime verdict that bar).
    """
    df = pd.DataFrame({sym: pd.Series(arr) for sym, arr in close_map.items()})
    leg_z: dict[str, pd.Series] = {}
    for _label, num, den in _COMPOSITE_LEGS:
        if num not in df or den not in df:
            continue
        pair = df[[num, den]].dropna()
        if pair.shape[0] < _MIN_HISTORY_DAYS:
            continue
        ratio = (pair.iloc[:, 0] / pair.iloc[:, 1]).replace(0.0, np.nan)
        agg = ratio.rolling(_HEALTH_Z_WINDOW, min_periods=_MIN_HISTORY_DAYS).agg(
            ["mean", "std"]
        )
        z = (ratio - agg["mean"]) / agg["std"].replace(0.0, np.nan)
        leg_z[_label] = z.reindex(df.index).ffill(limit=_STALE_TRADING_DAYS)

    if not leg_z:
        return None
    stack = pd.DataFrame(leg_z)
    alive = stack.notna().sum(axis=1)
    mean_z = stack.mean(axis=1, skipna=True)
    mean_z = mean_z.where(alive / len(_COMPOSITE_LEGS) > _MIN_QUORUM_FRACTION)
    smooth = mean_z.ewm(
        span=_EWMA_SPAN, adjust=False, min_periods=_EWMA_MIN_PERIODS
    ).mean()
    lst = smooth.dropna()
    return float(lst.iloc[-1]) if len(lst) else None
