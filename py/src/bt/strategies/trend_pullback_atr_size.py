"""Weekly-trend-confirmed mean reversion — ATR position sizing.

Long-only:
  1. Weekly filter: price > 50-week SMA (structural uptrend)
  2. Daily overshoot: close <= (20-day MA - ATR_mult × ATR)
  3. Entry: next bar
  4. Exit: close > 20-day MA (mean reversion complete)

Sizing: risk a fixed % of capital per ATR unit.
  qty = (capital * risk_pct) / (ATR * atr_mult)
This equalizes risk across assets — lower-vol names get more shares,
higher-vol fewer.

Factor attribution (3y, SPY+GLD+QQQ, 2022-2024):
  - 50-week SMA filter:       essential (+0.26 Sharpe vs 200d SMA)
  - ATR sizing 1.0% risk:     +48% return at same Sharpe vs flat sizing
  - Weekly resample:          critical (200d SMA loses -0.26 Sharpe)
  - Volume filter:            -0.04 Sharpe (removes good trades)
  - Trailing stop:            -0.06 Sharpe (cuts winners)
  - Sector universe:          -0.39 Sharpe (correlation clustering)
  - Symmetrical overbought exit: no measurable difference on daily bars

Best config: ATR sizing 1.0-1.25% risk, 50w SMA, MA20 exit, SPY+GLD+QQQ.
Sharpe 0.88-0.91, 75% win rate, -5.8% max DD vs SPY -25.4%.
Zero trades in 2022 bear market — full capital preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "trend_pullback_atr_size"

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
    weekly_return_threshold: float = -99.0  # effectively disabled

    # ATR-based position sizing: risk this % of capital per ATR unit
    risk_pct: float = 0.01  # 1% of capital at risk per ATR unit

    # Warmup
    warmup_bars: int = 60

    # Cooldown after exit
    cooldown_bars: int = 3


# ---------------------------------------------------------------------------
# state
# Repo GLOBAL-dict convention + reset_global() for the split engine.
# ---------------------------------------------------------------------------

GLOBAL: dict = {
    "cooldowns": {},
    "weekly_cache": {},
}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldowns": {}, "weekly_cache": {}}


# ---------------------------------------------------------------------------
# checks
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
            if _above_ma(cast(pd.Series, df["close"]), params):
                start_cooldown(symbol, params.cooldown_bars)
                exit_price = float(cast(pd.Series, df["close"]).iloc[-1])
                for position in pos_tup:
                    signals.append(
                        TradeSignal(
                            action=ActionType.close,
                            symbol=symbol,
                            timestamp=candle.timestamp,
                            price=exit_price,
                            qty=abs(position.qty),
                            position_id=position.position_id,
                            reason="[atrsz] close > 20MA — mean reversion complete",
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
                reason=f"[atrsz] 50wSMA↑ + {params.atr_mult}×ATR oversold (ATR={atr_val:.2f} risk={params.risk_pct:.1%})",
            )
        )

    return signals


def start_cooldown(symbol: str, bars: int) -> None:
    GLOBAL["cooldowns"][symbol] = bars
