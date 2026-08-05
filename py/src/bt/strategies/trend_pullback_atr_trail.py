"""Weekly-trend-confirmed mean reversion — progressive trail exit.

Long-only:
  1. Weekly filter: price > 50-week SMA (structural uptrend)
  2. Daily overshoot: close <= (20-day MA - ATR_mult × ATR)
  3. Entry: next bar
  4. Exit: two-phase —
     a. Pre-reversion: close > MA20 → switch to trail mode (don't exit)
     b. Trail mode: close < lowest low of last N bars → exit

Key difference from original: instead of exiting at MA20, we switch to
a trailing stop. Dead-cat bounces still exit quickly (trail catches them).
Real trend reversions compound — the trail widens as price rises.

Factor attribution note from original:
  - Trailing stop: -0.06 Sharpe (cuts winners)
That was a continuous trail from entry. This trail only activates AFTER
MA20 is recovered — mean reversion is confirmed before trailing begins.

Sizing: risk a fixed % of capital per ATR unit.
  qty = (capital * risk_pct) / (ATR * atr_mult)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "trend_pullback_atr_trail"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Entry — daily overshoot
    atr_period: int = 14
    atr_mult: float = 1.5
    ma_period: int = 20

    # Weekly trend filter
    weekly_ma_period: int = 50
    weekly_return_threshold: float = -99.0

    # ATR-based position sizing
    risk_pct: float = 0.01

    # Trail stop — activated after price recovers above MA
    trail_lookback: int = 10  # bars for lowest-low trail

    # Warmup & cooldown
    warmup_bars: int = 60
    cooldown_bars: int = 3


# ---------------------------------------------------------------------------
# state
# Repo GLOBAL-dict convention + reset_global() for the split engine.
# ---------------------------------------------------------------------------

GLOBAL: dict = {
    "cooldowns": {},
    "weekly_cache": {},
    "in_trail": set(),  # position_ids currently in trail mode
}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldowns": {}, "weekly_cache": {}, "in_trail": set()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _weekly_bullish(
    symbol: str,
    daily_closes: pd.Series,
    params: Params,
    current_ts: pd.Timestamp,
) -> bool:
    if len(daily_closes) < params.weekly_ma_period * 5 + 1:
        return False

    weekly = daily_closes.resample("W").last().dropna()
    if len(weekly) < params.weekly_ma_period:
        return False

    last_weekly_ts = weekly.index[-1]
    cache_key = GLOBAL["weekly_cache"].get(symbol)
    if cache_key is not None and cache_key[0] == last_weekly_ts:
        return cache_key[1]

    sma50 = float(weekly.iloc[-params.weekly_ma_period :].mean())
    price_ok = float(weekly.iloc[-1]) > sma50

    if params.weekly_return_threshold > -99.0 and len(weekly) >= 2:
        weekly_return = (weekly.iloc[-1] - weekly.iloc[-2]) / weekly.iloc[-2]
        ret_ok = float(weekly_return) > params.weekly_return_threshold
    else:
        ret_ok = True

    result = price_ok and ret_ok
    GLOBAL["weekly_cache"][symbol] = (last_weekly_ts, result)
    return result


def _is_oversold(
    closes: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    params: Params,
) -> bool:
    needed = max(params.ma_period, params.atr_period) + 1
    if len(closes) < needed:
        return False

    ma20 = float(ta.sma(closes, params.ma_period).iloc[-1])
    atr_val = float(ta.atr(highs, lows, closes, params.atr_period).iloc[-1])

    if np.isnan(ma20) or np.isnan(atr_val):
        return False

    threshold = ma20 - params.atr_mult * atr_val
    return float(closes.iloc[-1]) <= threshold


def _above_ma(closes: pd.Series, params: Params) -> bool:
    if len(closes) < params.ma_period + 1:
        return False
    ma20 = float(ta.sma(closes, params.ma_period).iloc[-1])
    if np.isnan(ma20):
        return False
    return float(closes.iloc[-1]) > ma20


def _trail_broken(lows: pd.Series, lookback: int) -> bool:
    """True if close broke below the lowest low of last N bars."""
    if len(lows) < lookback + 1:
        return False
    trail_level = float(lows.iloc[-lookback:-1].min())
    return float(lows.iloc[-1]) < trail_level


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []

    interval = candle.interval or "1d"
    if interval != "1d":
        return signals

    symbols: list[str] = sorted({k[0] for k in state.candles.keys()})

    for symbol in symbols:
        pos_tup = state.portfolio.positions.get(symbol, ())
        if pos_tup:
            df = state.candles.get((symbol, "1d"))
            if df is None:
                continue
            closes = cast(pd.Series, df["close"])
            lows = cast(pd.Series, df["low"])
            exit_price = float(closes.iloc[-1])

            for position in pos_tup:
                pid = position.position_id

                # Phase 1: wait for MA20 recovery
                if pid not in GLOBAL["in_trail"]:
                    if _above_ma(closes, params):
                        GLOBAL["in_trail"].add(pid)
                    # Still below MA20 — hold, no exit
                    continue

                # Phase 2: trail mode — exit on trail stop break
                if _trail_broken(lows, params.trail_lookback):
                    GLOBAL["in_trail"].discard(pid)
                    start_cooldown(symbol, params.cooldown_bars)
                    trail_val = float(lows.iloc[-params.trail_lookback : -1].min())
                    signals.append(
                        TradeSignal(
                            action=ActionType.close,
                            symbol=symbol,
                            timestamp=candle.timestamp,
                            price=exit_price,
                            qty=abs(position.qty),
                            position_id=pid,
                            reason=f"[trail] MA20 recovered → trail stop {trail_val:.2f} broken",
                        )
                    )
            continue

        cd = GLOBAL["cooldowns"].get(symbol, 0)
        if cd > 0:
            GLOBAL["cooldowns"][symbol] = cd - 1
            continue

        df = state.candles.get((symbol, "1d"))
        if df is None:
            continue

        closes = cast(pd.Series, df["close"])
        highs = cast(pd.Series, df["high"])
        lows = cast(pd.Series, df["low"])

        if len(closes) < params.warmup_bars:
            continue

        if not _weekly_bullish(symbol, closes, params, candle.timestamp):
            continue

        if not _is_oversold(closes, highs, lows, params):
            continue

        entry_price = float(closes.iloc[-1])
        atr_val = float(ta.atr(highs, lows, closes, params.atr_period).iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            continue
        risk_dollars = state.portfolio.cash * params.risk_pct
        qty = risk_dollars / (atr_val * params.atr_mult)

        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=symbol,
                timestamp=candle.timestamp,
                price=entry_price,
                qty=round(qty, 4),
                reason=f"[trail] 50wSMA↑ + {params.atr_mult}×ATR oversold (ATR={atr_val:.2f})",
            )
        )

    return signals


def start_cooldown(symbol: str, bars: int) -> None:
    GLOBAL["cooldowns"][symbol] = bars
