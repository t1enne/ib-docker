"""Price-volume (OBV) divergence screen — score a universe for accumulation/
distribution warnings.

Scores symbols on a fresh divergence between price and On-Balance Volume (OBV)
at the latest bar. OBV accumulates volume on up-moves and sheds it on down-
moves; a **bullish** price-volume divergence prints when the most recent swing
low makes a *lower* price low while OBV makes a *higher* low (price sold off but
volume was not confirming the selloff — accumulation underneath). A **bearish**
divergence prints when the most recent swing high makes a *higher* price high
while OBV makes a *lower* high (price advanced but volume did not confirm —
distribution). Otherwise flat.

This is a leading, high-value companion to momentum (RSI/MACD) divergences
because it uses a *different* input dimension: it measures the money-flow
confirmation of a price move rather than the oscillator position, so it is not
correlated with RSI/MACD divergence and catches a distinct class of reversal
(the "smart money" exit / accumulation-into-the-dip) that an oscillator-only
screen cannot see. OBV is unbounded in scale, so scores are normalized by the
local cross-sectional range of OBV to stay comparable across symbols.

Manual-trading *screening* layer: returns a 0..1 score + reasons, never a
``TradeSignal``. Reuses the pure fractal-swing detector from the RSI divergence
strategies (same pivot math, same swing indices).

Scoring knobs (``Params``): ``pivot_lookback``, ``min_lookback``. The
divergence direction is decided purely by price-vs-OBV slope disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.screen.metrics import with_common_metrics
from src.bt.screen.types import ScreenParams, ScreenResult, ScreenState
from src.bt.strategies.rsi_divergence_dsl import find_swing_lows
from src.bt.strategies.rsi_bearish_divergence_dsl import find_swing_highs

SCREEN_TYPE = "obv_divergence"


@dataclass(frozen=True)
class Params(ScreenParams):
    pivot_lookback: int = 5
    min_lookback: int = 100


# ---------------------------------------------------------------------------
# pure scoring helpers (vectorized windows; no module state)
# ---------------------------------------------------------------------------


def _ta_obv(closes: pd.Series, volume: pd.Series) -> np.ndarray:
    """Lazy entry to ``ta.obv`` returning a float ndarray."""
    from src.indicators.ta import obv

    return obv(closes, volume).to_numpy(dtype=float)


def _bullish_divergence(
    lows: np.ndarray, obv: np.ndarray, lookback: int
) -> tuple[int, int] | None:
    """Lower price low while OBV prints a higher low (accumulation)."""
    n = len(lows)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_lows(lows, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    o_cur, o_prev = obv[cur], obv[prev]
    if not (np.isfinite(o_cur) and np.isfinite(o_prev)):
        return None
    if not (lows[cur] < lows[prev]):  # price lower low
        return None
    if not (o_prev < o_cur):  # OBV higher low -> accumulation
        return None
    return (cur, prev)


def _bearish_divergence(
    highs: np.ndarray, obv: np.ndarray, lookback: int
) -> tuple[int, int] | None:
    """Higher price high while OBV prints a lower high (distribution)."""
    n = len(highs)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_highs(highs, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    o_cur, o_prev = obv[cur], obv[prev]
    if not (np.isfinite(o_cur) and np.isfinite(o_prev)):
        return None
    if not (highs[cur] > highs[prev]):  # price higher high
        return None
    if not (o_prev > o_cur):  # OBV lower high -> distribution
        return None
    return (cur, prev)


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
    """Score every symbol's latest bar for a fresh price-OBV divergence."""
    results: list[ScreenResult] = []
    for symbol, df in state.frames:
        if df is None or df.empty or "close" not in df.columns:
            continue
        if len(df) < params.min_lookback:
            results.append(_flat(state.ts, symbol, df))
            continue

        closes = cast(pd.Series, df["close"])
        lows = df["low"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        if "volume" not in df.columns:
            results.append(_flat(state.ts, symbol, df))
            continue
        obv = _ta_obv(closes, cast(pd.Series, df["volume"]))
        if len(obv) != len(lows):
            results.append(_flat(state.ts, symbol, df))
            continue

        bull = _bullish_divergence(lows, obv, params.pivot_lookback)
        bear = _bearish_divergence(highs, obv, params.pivot_lookback)

        if bull is None and bear is None:
            results.append(_flat(state.ts, symbol, df))
            continue

        bull_score = _raw_gap(bull, obv) if bull else 0.0
        bear_score = _raw_gap(bear, obv) if bear else 0.0

        if bull_score >= bear_score:
            assert bull is not None
            cur, prev = bull
            gap = obv[cur] - obv[prev]
            action, signals = "long", ("bullish obv divergence",)
        else:
            assert bear is not None
            cur, prev = bear
            gap = obv[prev] - obv[cur]
            action, signals = "short", ("bearish obv divergence",)

        # Normalize the raw OBV gap by the trailing OBV range so the 0..1 score
        # is cross-sectionally comparable (OBV scale varies wildly per name).
        scale = _obv_scale(obv)
        strength = min(1.0, abs(gap) / scale) if scale > 0 else 0.0
        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=strength,
                action=action,
                signals=signals,
                model_features=with_common_metrics(
                    {"obv_gap": gap, "obv": obv[cur]}, df
                ),
            )
        )
    return tuple(results)


def _raw_gap(div: tuple[int, int], obv: np.ndarray) -> float:
    """Absolute OBV gap for a fired divergence (picks the stronger print)."""
    cur, prev = div
    return float(abs(obv[cur] - obv[prev]))


def _obv_scale(obv: np.ndarray) -> float:
    """A robust scale proxy: the full-range minus the trailing quarter, else 0.

    Uses the last ``n/4`` bars' min/max spread so the normalization reflects the
    *recent* OBV trading band rather than the cumulative all-time range.
    """
    if len(obv) < 2:
        return 0.0
    tail = max(1, len(obv) // 4)
    window = obv[-tail:]
    return float(max(np.nanmax(window) - np.nanmin(window), 0.0))
