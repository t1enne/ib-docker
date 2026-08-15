"""RSI-divergence screen — score a universe for price-RSI divergences.

Scores symbols on a fresh price-RSI divergence at the latest bar, reusing the
pure fractal swing + divergence detectors from the backtest strategies
``rsi_divergence_dsl`` (bullish/LONG oversold reversion) and
``rsi_bearish_divergence_dsl`` (bearish/SHORT overbought reversion). This is a
manual-trading *screening* layer, not a fill instruction: it returns a 0..1
score and reasons, never a ``TradeSignal``.

A symbol is scored **long** when a fresh bullish divergence prints — the most
recent swing low makes a *lower* price low while RSI makes a *higher* low and
sits genuinely oversold (below ``rsi_floor``). It is scored **short** when a
fresh bearish divergence prints — the most recent swing high makes a *higher*
price high while RSI makes a *lower* high and sits genuinely overbought (above
``rsi_ceiling``). Otherwise the symbol is **flat**. The screen does not simulate
confirmation-bar entry, cooldowns, or position management — those live in the
backtest strategies. Reuse keeps the screen's scoring identical to the
strategy's signal detection (same swing/divergence maths, same knobs).

Scoring knobs (``Params``): ``rsi_period``, ``pivot_lookback``,
``rsi_floor`` (bullish oversold gate), ``rsi_ceiling`` (bearish overbought
gate), ``min_lookback`` warmup.

Direction is decided purely by which divergence fired and how fresh it is. The
score is a monotonic function of divergence strength and depth:

  best-of-bull/bear = score_sign * (depth_term + gap_term) clamped to 0..1

``depth_term`` rewards how far the trigger RSI is past its oversold/overbought
gate (deeper reversal), ``gap_term`` rewards the *size* of the RSI divergence
(the higher-low / lower-high gap between the last two swing RSI values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.screen.metrics import with_common_metrics
from src.bt.screen.types import ScreenParams, ScreenResult, ScreenState
from src.bt.strategies.rsi_divergence_dsl import detect_bullish_divergence
from src.bt.strategies.rsi_bearish_divergence_dsl import detect_bearish_divergence

SCREEN_TYPE = "rsi_divergence"

#: Oversold/overbought thresholds used to scale the score (not gates themselves —
#: the gates are ``rsi_floor``/``rsi_ceiling``). These bound how far RSI can be
#: past the gate before the depth term saturates at 1.0.
_OVERSOLD = 20.0
_OVERBOUGHT = 80.0
#: RSI-value gap (0..100) above which the divergence-gap term saturates at 1.0.
_GAP_SATURATE = 6.0


@dataclass(frozen=True)
class Params(ScreenParams):
    rsi_period: int = 14
    pivot_lookback: int = 5
    rsi_floor: float = 35.0
    rsi_ceiling: float = 65.0
    min_lookback: int = 100


# ---------------------------------------------------------------------------
# pure scoring helpers (vectorized windows; no module state)
# ---------------------------------------------------------------------------


def _ta_rsi(closes: pd.Series, window: int) -> np.ndarray:
    """Lazy entry to ``ta.rsi`` to avoid triggering ``src.indicators`` package
    init at ``src.bt`` bootstrap/collection time (same pattern as momentum)."""
    from src.indicators.ta import rsi

    return rsi(closes, window=window).to_numpy(dtype=float)


def _bullish_score(cur_rsi: float, prev_rsi: float, rsi_floor: float) -> float:
    """0..1 bullish-divergence strength from the trigger + divergence depth.

    ``depth_term`` is how far ``cur_rsi`` sits below ``rsi_floor`` (the OVERSOLD
    depth), scaled from ``rsi_floor`` down to ``_OVERSOLD``. ``gap_term`` is the
    size of the higher-RSI-low gap (``prev_rsi`` lower than ``cur_rsi``) scaled
    to ``_GAP_SATURATE``. Blend, clamp to [0, 1].
    """
    depth = (rsi_floor - cur_rsi) / max(1e-9, rsi_floor - _OVERSOLD)
    gap = (cur_rsi - prev_rsi) / _GAP_SATURATE
    return float(min(1.0, max(0.0, 0.6 * depth + 0.4 * gap)))


def _bearish_score(cur_rsi: float, prev_rsi: float, rsi_ceiling: float) -> float:
    """0..1 bearish-divergence strength from the trigger + divergence depth.

    ``depth_term`` is how far ``cur_rsi`` sits above ``rsi_ceiling`` (the
    OVERBOUGHT depth), scaled from ``rsi_ceiling`` up to ``_OVERBOUGHT``.
    ``gap_term`` is the size of the lower-RSI-high gap (``prev_rsi`` higher than
    ``cur_rsi``). Blend, clamp to [0, 1].
    """
    depth = (cur_rsi - rsi_ceiling) / max(1e-9, _OVERBOUGHT - rsi_ceiling)
    gap = (prev_rsi - cur_rsi) / _GAP_SATURATE
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
    """Score every symbol's latest bar for a fresh price-RSI divergence."""
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
        rsi_arr = _ta_rsi(closes, params.rsi_period)
        if len(rsi_arr) != len(lows):
            results.append(_flat(state.ts, symbol, df))
            continue

        bull = detect_bullish_divergence(
            lows, rsi_arr, params.pivot_lookback, params.rsi_floor
        )
        bear = detect_bearish_divergence(
            highs, rsi_arr, params.pivot_lookback, params.rsi_ceiling
        )

        # A symbol may carry both flags; resolve by the stronger score.
        bull_score = (
            _bullish_score(rsi_arr[bull[0]], rsi_arr[bull[1]], params.rsi_floor)
            if bull is not None
            else 0.0
        )
        bear_score = (
            _bearish_score(rsi_arr[bear[0]], rsi_arr[bear[1]], params.rsi_ceiling)
            if bear is not None
            else 0.0
        )

        if bull is None and bear is None:
            results.append(_flat(state.ts, symbol, df))
            continue

        if bull_score >= bear_score:
            assert bull is not None
            action = "long"
            score = bull_score
            cur, prev = bull
            gap = rsi_arr[cur] - rsi_arr[prev]
            signals = ("bullish rsi divergence",)
            feats: dict[str, float] = {"cur_rsi": rsi_arr[cur], "rsi_gap": gap}
        else:
            assert bear is not None
            action = "short"
            score = bear_score
            cur, prev = bear
            gap = rsi_arr[prev] - rsi_arr[cur]
            signals = ("bearish rsi divergence",)
            feats = {"cur_rsi": rsi_arr[cur], "rsi_gap": gap}

        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=score,
                action=action,
                signals=signals,
                model_features=with_common_metrics(feats, df),
            )
        )
    return tuple(results)
