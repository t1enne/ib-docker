"""MACD-divergence screen — score a universe for price-MACD divergences.

Scores symbols on a fresh price-MACD divergence at the latest bar, mirroring the
RSI-divergence screen but with the **MACD histogram** as the momentum axis. A
bullish divergence prints when the most recent swing low makes a *lower* price
low while the MACD histogram makes a *higher* low (and is genuinely weak, below
the ``hist_zero_floor``); a bearish divergence prints when the most recent swing
high makes a *higher* price high while the histogram makes a *lower* high (and
is genuinely strong, above ``hist_zero_ceiling``). Otherwise flat.

The MACD histogram ("oscillator") is more sensitive to trend *breaks* than RSI
because it zero-crosses on every trend reversal; divergence on the histogram
surfaces fading momentum earlier than a raw RSI low/high. Direction is decided
purely by which divergence fired; the score rewards histogram depth (how far
past the zero gate) and divergence size (the higher-low / lower-high histogram
gap between the two swing extremes), same monotonic shape as the RSI screen.

This is a manual-trading *screening* layer: returns a 0..1 score + reasons,
never a ``TradeSignal``. Reuses the pure fractal-swing detector from the
backtest divergence strategies (same pivot math, same swing indices).

Scoring knobs (``Params``): ``fast`` / ``slow`` / ``signal`` (MACD),
``bull_hist_floor`` (gate on the *trigger* swing low's histogram), ``bear_hist_ceiling``
(gate on the *trigger* swing high's histogram), ``pivot_lookback``, ``min_lookback``.
The histogram strength gates are evaluated on the recent price extreme, not the
zero-crossing itself.
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

SCREEN_TYPE = "macd_divergence"


@dataclass(frozen=True)
class Params(ScreenParams):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    pivot_lookback: int = 5
    #: Histogram gates on the trigger swing: below ``bull_hist_floor`` for a
    #: bearish histogram low (bullish divergence), above ``bear_hist_ceiling``
    #: for a strong histogram high (bearish divergence).
    bull_hist_floor: float = 0.0
    bear_hist_ceiling: float = 0.0
    min_lookback: int = 100


# ---------------------------------------------------------------------------
# pure scoring helpers (vectorized windows; no module state)
# ---------------------------------------------------------------------------


def _ta_histogram(closes: pd.Series, fast: int, slow: int, signal: int) -> np.ndarray:
    """Lazy entry to ``ta.macd`` returning the histogram as a float ndarray."""
    from src.indicators.ta import macd

    hist = macd(closes, fast, slow, signal)["histogram"]
    return hist.to_numpy(dtype=float)


def _bullish_divergence(
    lows: np.ndarray, hist: np.ndarray, lookback: int, bull_floor: float
) -> tuple[int, int] | None:
    """Lower price low while histogram prints a higher low below ``bull_floor``."""
    n = len(lows)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_lows(lows, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    h_cur, h_prev = hist[cur], hist[prev]
    if not (np.isfinite(h_cur) and np.isfinite(h_prev)):
        return None
    if not (lows[cur] < lows[prev]):  # price lower low
        return None
    if not (h_prev < h_cur):  # histogram higher low -> divergence
        return None
    if not (h_cur < bull_floor):  # genuinely weak histogram
        return None
    return (cur, prev)


def _bearish_divergence(
    highs: np.ndarray, hist: np.ndarray, lookback: int, bear_ceiling: float
) -> tuple[int, int] | None:
    """Higher price high while histogram prints a lower high above the ceiling."""
    n = len(highs)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_highs(highs, lookback)
    if len(idxs) < 2:
        return None
    cur, prev = idxs[-1], idxs[-2]
    if n - 1 - cur > lookback * 2:
        return None
    h_cur, h_prev = hist[cur], hist[prev]
    if not (np.isfinite(h_cur) and np.isfinite(h_prev)):
        return None
    if not (highs[cur] > highs[prev]):  # price higher high
        return None
    if not (h_prev > h_cur):  # histogram lower high -> divergence
        return None
    if not (h_cur > bear_ceiling):  # genuinely strong histogram
        return None
    return (cur, prev)


def _finalize(
    state: ScreenState,
    symbol: str,
    df: pd.DataFrame,
    hist: np.ndarray,
    cur: int,
    prev: int,
    strength: float,
    gap: float,
    signals: tuple[str, ...],
) -> ScreenResult:
    """Build a scored result for a fired divergence."""
    return ScreenResult(
        symbol=symbol,
        timestamp=state.ts,
        score=strength,
        action="long" if "bullish" in signals[0] else "short",
        signals=signals,
        model_features=with_common_metrics({"hist": hist[cur], "hist_gap": gap}, df),
    )


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
    """Score every symbol's latest bar for a fresh price-MACD divergence."""
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
        hist = _ta_histogram(closes, params.fast, params.slow, params.signal)
        if len(hist) != len(lows):
            results.append(_flat(state.ts, symbol, df))
            continue

        bull = _bullish_divergence(
            lows, hist, params.pivot_lookback, params.bull_hist_floor
        )
        bear = _bearish_divergence(
            highs, hist, params.pivot_lookback, params.bear_hist_ceiling
        )
        bull_score = _score(bull, hist) if bull else 0.0
        bear_score = _score(bear, hist) if bear else 0.0

        if bull is None and bear is None:
            results.append(_flat(state.ts, symbol, df))
            continue

        if bull_score >= bear_score:
            assert bull is not None
            cur, prev = bull
            gap = hist[cur] - hist[prev]
            signals = ("bullish macd divergence",)
        else:
            assert bear is not None
            cur, prev = bear
            gap = hist[prev] - hist[cur]
            signals = ("bearish macd divergence",)

        # Scaled strength: normalize histogram depth/gap against the local price
        # scale (ATR-relative) so scores are comparable across symbols and the
        # result stays a bounded 0..1 proxy for "how far histogram diverged".
        scale = _atr_scale(df)
        results.append(
            _finalize(
                state,
                symbol,
                df,
                hist,
                cur,
                prev,
                strength=min(1.0, abs(gap) / scale),
                gap=gap,
                signals=signals,
            )
        )
    return tuple(results)


def _atr_scale(frame: pd.DataFrame) -> float:
    """A short-Wilder ATR used to normalise histogram magnitudes to 0..1."""
    closes = frame["close"]
    high = frame["high"] if "high" in frame.columns else closes
    low = frame["low"] if "low" in frame.columns else closes
    from src.indicators.ta import atr

    try:
        v = float(atr(high, low, closes, window=14).iloc[-1])
    except IndexError, ValueError:
        return 1.0
    return v if np.isfinite(v) and v > 0 else 1.0


def _score(div: tuple[int, int], hist: np.ndarray) -> float:
    """Raw histogram gap magnitude for a fired divergence (used to pick which
    of a simultaneous bull/bear print is the stronger setup)."""
    cur, prev = div
    return float(abs(hist[cur] - hist[prev]))
