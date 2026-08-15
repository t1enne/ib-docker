"""MFI-divergence screen — score a universe for price-MFI divergences.

Scores symbols on a fresh price-Money Flow Index (MFI) divergence at the latest
bar. MFI is RSI weighted by *volume* (a money-flow oscillator on 0..100), so it
trades like the RSI-divergence screen but imports a volume dimension: an
overbought/oversold extreme that volume is not confirming. A **bullish**
divergence prints when the most recent swing low makes a *lower* price low while
MFI makes a *higher* low and sits genuinely oversold (below ``mfi_floor``); a
**bearish** divergence prints when the most recent swing high makes a *higher*
price high while MFI makes a *lower* high and sits genuinely overbought (above
``mfi_ceiling``). Otherwise flat.

Direction is decided purely by which divergence fired. The score is a monotonic
function of MFI depth (how far past the oversold/overbought gate) and the size
of the MFI higher-low / lower-high gap, same shape as the RSI screen — but the
underlying axis is money-flow, so a divergence only scores when *volume* backed
the move, filtering weak, low-participation reversals that a pure RSI screen
would flag.

Manual-trading *screening* layer: returns a 0..1 score + reasons, never a
``TradeSignal``. Reuses the pure fractal-swing detector from the RSI divergence
strategies (same pivot math, same swing indices).

Scoring knobs (``Params``): ``mfi_period``, ``pivot_lookback``, ``mfi_floor``
(bullish oversold gate), ``mfi_ceiling`` (bearish overbought gate),
``min_lookback`` warmup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bt.screen.metrics import with_common_metrics
from src.bt.screen.types import ScreenParams, ScreenResult, ScreenState
from src.bt.strategies.rsi_divergence_dsl import find_swing_lows
from src.bt.strategies.rsi_bearish_divergence_dsl import find_swing_highs

SCREEN_TYPE = "mfi_divergence"

#: MFI oversold/overbought anchors used to scale the depth term (these bound how
#: far past the gate the depth term saturates, mirroring the RSI screen).
_OVERSOLD = 20.0
_OVERBOUGHT = 80.0
#: MFI-value gap (0..100) above which the divergence-gap term saturates at 1.0.
_GAP_SATURATE = 6.0


@dataclass(frozen=True)
class Params(ScreenParams):
    mfi_period: int = 14
    pivot_lookback: int = 5
    mfi_floor: float = 35.0
    mfi_ceiling: float = 65.0
    min_lookback: int = 100


# ---------------------------------------------------------------------------
# pure scoring helpers (vectorized windows; no module state)
# ---------------------------------------------------------------------------


def _ta_mfi(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Lazy entry to ``ta.mfi`` returning a float ndarray (0..100)."""
    from src.indicators.ta import mfi

    mfi_arr = mfi(frame["high"], frame["low"], frame["close"], frame["volume"], window)
    return mfi_arr.to_numpy(dtype=float)


def _bullish_divergence(
    lows: np.ndarray, mfi: np.ndarray, lookback: int, mfi_floor: float
) -> tuple[int, int] | None:
    """Lower price low while MFI prints a higher low below ``mfi_floor``."""
    n = len(lows)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_lows(lows, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    m_cur, m_prev = mfi[cur], mfi[prev]
    if not (np.isfinite(m_cur) and np.isfinite(m_prev)):
        return None
    if not (lows[cur] < lows[prev]):  # price lower low
        return None
    if not (m_prev < m_cur):  # MFI higher low -> divergence
        return None
    if not (m_cur < mfi_floor):  # genuinely oversold (money flow)
        return None
    return (cur, prev)


def _bearish_divergence(
    highs: np.ndarray, mfi: np.ndarray, lookback: int, mfi_ceiling: float
) -> tuple[int, int] | None:
    """Higher price high while MFI prints a lower high above ``mfi_ceiling``."""
    n = len(highs)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_highs(highs, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    m_cur, m_prev = mfi[cur], mfi[prev]
    if not (np.isfinite(m_cur) and np.isfinite(m_prev)):
        return None
    if not (highs[cur] > highs[prev]):  # price higher high
        return None
    if not (m_prev > m_cur):  # MFI lower high -> divergence
        return None
    if not (m_cur > mfi_ceiling):  # genuinely overbought (money flow)
        return None
    return (cur, prev)


def _bullish_score(cur_mfi: float, prev_mfi: float, mfi_floor: float) -> float:
    """0..1 bullish-divergence strength from depth + divergence gap.

    ``depth_term`` is how far ``cur_mfi`` sits below ``mfi_floor`` (the OVERSOLD
    money-flow depth), scaled from ``mfi_floor`` down to ``_OVERSOLD``.
    ``gap_term`` is the size of the higher-MFI-low gap, scaled to
    ``_GAP_SATURATE``. Blend, clamp to [0, 1].
    """
    depth = (mfi_floor - cur_mfi) / max(1e-9, mfi_floor - _OVERSOLD)
    gap = (cur_mfi - prev_mfi) / _GAP_SATURATE
    return float(min(1.0, max(0.0, 0.6 * depth + 0.4 * gap)))


def _bearish_score(cur_mfi: float, prev_mfi: float, mfi_ceiling: float) -> float:
    """0..1 bearish-divergence strength from depth + divergence gap."""
    depth = (cur_mfi - mfi_ceiling) / max(1e-9, _OVERBOUGHT - mfi_ceiling)
    gap = (prev_mfi - cur_mfi) / _GAP_SATURATE
    return float(min(1.0, max(0.0, 0.6 * depth + 0.4 * gap)))


def _flat(ts: pd.Timestamp, symbol: str, frame: pd.DataFrame) -> ScreenResult:
    return ScreenResult(
        symbol=symbol,
        timestamp=ts,
        score=0.0,
        action="flat",
        signals=("flat",),
        model_features=with_common_metrics({}, frame),
    )


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------


def on_state(state: ScreenState, params: Params) -> tuple[ScreenResult, ...]:
    """Score every symbol's latest bar for a fresh price-MFI divergence."""
    results: list[ScreenResult] = []
    for symbol, df in state.frames:
        if df is None or df.empty or "close" not in df.columns:
            continue
        if len(df) < params.min_lookback:
            results.append(_flat(state.ts, symbol, df))
            continue

        lows = df["low"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        if "volume" not in df.columns:
            results.append(_flat(state.ts, symbol, df))
            continue
        mfi_arr = _ta_mfi(df, params.mfi_period)
        if len(mfi_arr) != len(lows):
            results.append(_flat(state.ts, symbol, df))
            continue

        bull = _bullish_divergence(
            lows, mfi_arr, params.pivot_lookback, params.mfi_floor
        )
        bear = _bearish_divergence(
            highs, mfi_arr, params.pivot_lookback, params.mfi_ceiling
        )

        if bull is None and bear is None:
            results.append(_flat(state.ts, symbol, df))
            continue

        bull_score = (
            _bullish_score(mfi_arr[bull[0]], mfi_arr[bull[1]], params.mfi_floor)
            if bull
            else 0.0
        )
        bear_score = (
            _bearish_score(mfi_arr[bear[0]], mfi_arr[bear[1]], params.mfi_ceiling)
            if bear
            else 0.0
        )

        if bull_score >= bear_score:
            assert bull is not None
            action, score = "long", bull_score
            cur, prev = bull
            gap = mfi_arr[cur] - mfi_arr[prev]
            signals = ("bullish mfi divergence",)
        else:
            assert bear is not None
            action, score = "short", bear_score
            cur, prev = bear
            gap = mfi_arr[prev] - mfi_arr[cur]
            signals = ("bearish mfi divergence",)

        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=score,
                action=action,
                signals=signals,
                model_features=with_common_metrics(
                    {"cur_mfi": mfi_arr[cur], "mfi_gap": gap}, df
                ),
            )
        )
    return tuple(results)
