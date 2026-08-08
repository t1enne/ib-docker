"""Weekly-trend-confirmed mean reversion with dual-position entry.

Long-only.
  1. Weekly filter: price > 50-week SMA (structural uptrend)
  2. Daily overshoot: close <= (20-day MA - ATR_mult × ATR)
  3. Entry: next bar, opens TWO positions:
     a. "reversion" leg: 50% of capital at risk, exit when close > MA20
     b. "trend" leg: 50% of capital at risk, exit on MA trail stop
  4. Two independent positions with separate exit logic — captures
     mean reversion quickly while letting trend compound.

Sizing: risk a fixed % of capital per ATR unit, split across both legs.
  qty_total = (capital * risk_pct) / (ATR * atr_mult)
  qty_per_leg = qty_total / 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.regime.gates import weekly_above_sma
from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import sl_tp_from_pct

STRATEGY_TYPE = "trend_pullback_atr_enhanced"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Entry — daily overshoot
    atr_period: int = 14
    atr_mult: float = 1.0
    ma_period: int = 20

    # Weekly trend filter
    weekly_ma_period: int = 50
    weekly_return_threshold: float = -99.0  # effectively disabled

    # ATR-based position sizing (total across both legs)
    risk_pct: float = 0.025  # 2.5% of capital at risk total

    # Per-trade SL/TP fractional pcts of entry; <=0 disables that leg.
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Exit
    trail_ma_period: int = 10  # MA trail for the trend leg

    # Warmup & cooldown
    warmup_bars: int = 60
    cooldown_bars: int = 5


# ---------------------------------------------------------------------------
# state
# Repo GLOBAL-dict convention + reset_global() for the split engine.
# ---------------------------------------------------------------------------

GLOBAL: dict = {
    "cooldowns": {},
    "gate_cache": {},
}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldowns": {}, "gate_cache": {}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _weekly_bullish(state: BacktestState, symbol: str, params: Params) -> bool:
    return weekly_above_sma(
        state,
        symbol,
        window=params.weekly_ma_period,
        bar="1d",
        min_weekly_return=params.weekly_return_threshold,
        cache=GLOBAL["gate_cache"],
    )


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


def _above_ma(closes: pd.Series, ma_period: int) -> bool:
    if len(closes) < ma_period + 1:
        return False
    ma = float(ta.sma(closes, ma_period).iloc[-1])
    if np.isnan(ma):
        return False
    return float(closes.iloc[-1]) > ma


def _below_ma(closes: pd.Series, ma_period: int) -> bool:
    if len(closes) < ma_period + 1:
        return False
    ma = float(ta.sma(closes, ma_period).iloc[-1])
    if np.isnan(ma):
        return False
    return float(closes.iloc[-1]) <= ma


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
            exit_price = float(closes.iloc[-1])

            for position in pos_tup:
                pid = position.position_id
                is_trend = pid.endswith("_trend")

                if is_trend:
                    # Trend leg: close on MA trail break
                    if _below_ma(closes, params.trail_ma_period):
                        start_cooldown(symbol, params.cooldown_bars)
                        signals.append(
                            TradeSignal(
                                action=ActionType.close,
                                symbol=symbol,
                                timestamp=candle.timestamp,
                                price=exit_price,
                                qty=abs(position.qty),
                                position_id=pid,
                                reason=f"[enhanced] trend leg: close below MA{params.trail_ma_period} trail",
                            )
                        )
                else:
                    # Reversion leg: close at MA20 recovery
                    if _above_ma(closes, params.ma_period):
                        start_cooldown(symbol, params.cooldown_bars)
                        signals.append(
                            TradeSignal(
                                action=ActionType.close,
                                symbol=symbol,
                                timestamp=candle.timestamp,
                                price=exit_price,
                                qty=abs(position.qty),
                                position_id=pid,
                                reason="[enhanced] reversion leg: close > 20MA — mean reversion complete",
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

        if not _weekly_bullish(state, symbol, params):
            continue

        if not _is_oversold(closes, highs, lows, params):
            continue

        entry_price = float(closes.iloc[-1])
        atr_val = float(ta.atr(highs, lows, closes, params.atr_period).iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            continue
        risk_dollars = state.portfolio.cash * params.risk_pct
        qty_total = risk_dollars / (atr_val * params.atr_mult)
        qty_per_leg = round(qty_total / 2, 4)
        sl, tp = sl_tp_from_pct(
            entry_price, params.stop_loss, params.take_profit, is_long=True
        )

        ts = candle.timestamp
        base_id = f"{symbol}_{ts.timestamp():.0f}"

        # Reversion leg: close > MA20
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=symbol,
                timestamp=ts,
                price=entry_price,
                qty=qty_per_leg,
                position_id=f"{base_id}_rev",
                stop_loss=sl,
                take_profit=tp,
                reason=f"[enhanced] reversion {params.atr_mult}×ATR oversold (ATR={atr_val:.2f} risk={params.risk_pct:.1%})",
            )
        )

        # Trend leg: trail with MA10
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=symbol,
                timestamp=ts,
                price=entry_price,
                qty=qty_per_leg,
                position_id=f"{base_id}_trend",
                stop_loss=sl,
                take_profit=tp,
                reason=f"[enhanced] trend leg (trail=MA{params.trail_ma_period})",
            )
        )

    return signals


def start_cooldown(symbol: str, bars: int) -> None:
    GLOBAL["cooldowns"][symbol] = bars
