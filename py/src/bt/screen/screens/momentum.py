"""Momentum screen — manual-signal scoring on EMA cross + regime trend gate.

Scores symbols on a fast/slow EMA crossover gated by an EMA-based regime trend
gate, with a momentum filter, using the ``src.indicators`` TA layer (no
hand-rolled rolling math). Returns a 0..1 score proportional to momentum
magnitude, NOT a fill instruction.

Scoring knobs (``Params``):

  entry crossover   -> ``fast`` / ``slow`` EMA cross
  trend gate        -> ``trend_fast`` / ``trend_slow`` EMA cross with a range
                       threshold
  momentum filter   -> ``momentum_lookback`` / ``momentum_threshold``
  warmup            -> ``warmup_bars`` / ``slow`` window gating

Direction (long/short/flat) is decided purely by the trend gate + crossover +
momentum. Vol regime is surfaced as a diagnostic only — it never suppresses an
entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.bt.screen.types import (
    ScreenParams,
    ScreenResult,
    ScreenState,
    TrendRegime,
    VolRegime,
)

SCREEN_TYPE = "momentum"


@dataclass(frozen=True)
class Params(ScreenParams):
    # EMA crossover (entry signal).
    fast: int = 20
    slow: int = 50

    # Momentum filter: require N-bar return beyond threshold to enter.
    momentum_lookback: int = 20
    momentum_threshold: float = 0.02  # 2% over lookback

    # Regime trend gate (EMA cross + range threshold).
    trend_fast: int = 50
    trend_slow: int = 200
    range_threshold_pct: float = 0.005

    # Vol regime sizing hint (diagnostic only — never gates direction).
    size_high_vol: float = 0.5

    # Warmup.
    warmup_bars: int = 60


# ---------------------------------------------------------------------------
# pure helpers (vectorized, built on src.indicators; no module state)
# ---------------------------------------------------------------------------


def _ta_volatility(closes: pd.Series, window: int, annualized: bool):
    """Lazy entry to ``ta.volatility`` — avoids triggering ``src.indicators``
    package init (which pulls the HMM/Kalman stack and a ``src.bt.state``
    import cycle) at ``src.bt`` bootstrap/collection time."""
    from src.indicators.ta import volatility

    return volatility(closes, window=window, annualized=annualized)


def _trend_label(
    closes: pd.Series | None, fast: int, slow: int, range_threshold_pct: float
) -> TrendRegime | None:
    """Classify ``BULL`` / ``BEAR`` / ``RANGE`` via fast/slow EMA crossover.

    ``RANGE`` when the spread between the EMAs is within the threshold band,
    else direction of the spread.
    """
    if closes is None or len(closes) < slow:
        return None
    from src.indicators.ta import ema

    fast_ema = ema(closes, fast).iloc[-1]
    slow_ema = ema(closes, slow).iloc[-1]
    if pd.isna(fast_ema) or pd.isna(slow_ema) or slow_ema <= 0:
        return None
    spread = abs(fast_ema - slow_ema) / slow_ema
    if spread <= range_threshold_pct:
        return "RANGE"
    return "BULL" if fast_ema > slow_ema else "BEAR"


def _ema_cross(closes: pd.Series, fast: int, slow: int) -> tuple[bool, bool]:
    """Two-bar fast/slow EMA cross detection.

    Returns ``(crossed_up, crossed_down)``.
    """
    if len(closes) < slow:
        return (False, False)
    from src.indicators.ta import ema

    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    crossed_up = fast_ema.shift(1).iloc[-1] <= slow_ema.shift(1).iloc[-1] and (
        fast_ema.iloc[-1] > slow_ema.iloc[-1]
    )
    crossed_down = fast_ema.shift(1).iloc[-1] >= slow_ema.shift(1).iloc[-1] and (
        fast_ema.iloc[-1] < slow_ema.iloc[-1]
    )
    return (bool(crossed_up), bool(crossed_down))


def _momentum(closes: pd.Series, lookback: int) -> float | None:
    """N-bar return as a fraction. None when insufficient data."""
    if closes is None or len(closes) < lookback + 1:
        return None
    past = closes.iloc[-(lookback + 1)]
    current = closes.iloc[-1]
    if past == 0:
        return None
    return float((current - past) / past)


def _vol_label(closes: pd.Series | None, lookback: int = 20) -> VolRegime | None:
    """Diagnostic vol label from ``ta.volatility`` vs its historical median.

    ``HIGH_VOL`` when the latest rolling vol exceeds 1.25x the historical
    median, ``LOW_VOL`` below 0.75x, else ``MED_VOL``. Diagnostic only — never
    gates direction. ``ponytail:`` the backtest's ``dual_online`` HMM vol is not
    reproduced here; it only scales size and does not affect entry direction.
    """
    if closes is None or len(closes) < lookback * 2:
        return None
    vols = _ta_volatility(closes, window=lookback, annualized=False).dropna()
    if len(vols) < 2:
        return None
    latest = float(vols.iloc[-1])
    if latest <= 0:
        return None
    med = float(vols.median())
    if med <= 0:
        return None
    ratio = latest / med
    if ratio > 1.25:
        return "HIGH_VOL"
    if ratio < 0.75:
        return "LOW_VOL"
    return "MED_VOL"


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------


def on_state(state: ScreenState, params: Params) -> tuple[ScreenResult, ...]:
    """Score every symbol's latest bar. Returns one result per symbol."""
    results: list[ScreenResult] = []
    for symbol, df in state.frames:
        if df is None or df.empty or "close" not in df.columns:
            continue
        closes = cast(pd.Series, df["close"])
        if len(closes) < params.warmup_bars or len(closes) < params.slow:
            results.append(_flat(state.ts, symbol))
            continue

        trend = _trend_label(
            closes, params.trend_fast, params.trend_slow, params.range_threshold_pct
        )
        crossed_up, crossed_down = _ema_cross(closes, params.fast, params.slow)
        mom = _momentum(closes, params.momentum_lookback)
        vol = _vol_label(closes)
        action = "flat"
        score = 0.0

        if trend == "BULL" and crossed_up and mom is not None and mom > 0:
            enough = mom > params.momentum_threshold
            if enough:
                action = "long"
                score = _score(mom)
                signals = ["ema cross up", f"trend {trend}"]
            else:
                signals = ["ema cross up", f"trend {trend}", "momentum below thr"]
        elif trend == "BEAR" and crossed_down and mom is not None and mom < 0:
            enough = mom < -params.momentum_threshold
            if enough:
                action = "short"
                score = _score(abs(mom))
                signals = ["ema cross down", f"trend {trend}"]
            else:
                signals = ["ema cross down", f"trend {trend}", "momentum below thr"]
        else:
            if trend:
                signals = [f"trend {trend}"]
            else:
                signals = ["warmup/inactive"]

        if vol:
            signals.append(f"vol {vol}")

        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=score,
                action=action,
                signals=tuple(signals),
                model_features={
                    "momentum": mom if mom is not None else 0.0,
                    "vol_ratio": _vol_ratio(closes),
                },
            )
        )
    return tuple(results)


def _score(mom_mag: float) -> float:
    """Map momentum magnitude to a 0..1 score (clamped)."""
    return min(1.0, abs(mom_mag) * 10.0)


def _vol_ratio(closes: pd.Series) -> float:
    """Normalized rolling-vol feature for diagnostics (0.0 on missing data)."""
    v = _ta_volatility(closes, window=20, annualized=False)
    if v.iloc[-1] is None or pd.isna(v.iloc[-1]):
        return 0.0
    return float(v.iloc[-1] / (closes.mean() + 1e-9))


def _flat(ts: pd.Timestamp, symbol: str) -> ScreenResult:
    return ScreenResult(
        symbol=symbol,
        timestamp=ts,
        score=0.0,
        action="flat",
        signals=("flat",),
        model_features={},
    )
