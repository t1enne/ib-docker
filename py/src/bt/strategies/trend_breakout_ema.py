"""Trend-following breakout system — EMA bias + Donchian breakout + ATR risk.

Long/short trend following across daily bars:

  1. Trend filter  : EMA(100) sets bias. Close above -> long only,
                     close below -> short only (parametrised). Optional
                     min_trend_pct dead-zone keeps the strategy flat until
                     price clearly clears the EMA, so short-only/long-only
                     configs avoid trading chop near fair value.
  2. Entry         : long on a new 20-bar high, short on a new 20-bar low
                     (Donchian breakout), only in the matching bias.
  3. Pyramid       : up to N stacked positions per symbol during the trend.
                     Each add requires a fresh breakout strictly beyond the
                     prior entry and a min-bar gap since the last add.
  4. Stop          : per-position ATR stop. Long stop = entry - k*ATR,
                     short stop = entry + k*ATR. Explicit per level.
  5. Sizing        : fixed risk scaled by ATR. risk_dollars = cash * risk_pct,
                     qty = risk_dollars / (k * ATR). Equalises per-trade risk
                     across high/low-vol regimes.

All params are swept-later-friendly via the frozen `Params` dataclass.
State uses the repo GLOBAL-dict + reset_global() convention for the split engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    Position,
    TradeSignal,
)
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "trend_following_breakout"

FEEL = "tf-bo"  # reason tag prefix


# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Trend filter (bias)
    ema_trend_period: int = 100
    allow_short: bool = True
    allow_long: bool = True
    bias_smoothing: int = 1  # 1 = use last close; >1 = N-close mean vs EMA
    min_trend_pct: float = 0.0  # min |close-EMA|/EMA to hold a bias; 0 = off

    # Donchian breakout entry
    entry_lookback: int = 20  # new high/low over this many prior bars
    breakout_strict: bool = True  # must exceed prior extreme (strict) vs equal

    # ATR stop + sizing
    atr_period: int = 14
    stop_atr_mult: float = 2.0  # stop distance in ATR units
    risk_pct: float = 0.01  # % of capital risked per pyramid level

    # Pyramid
    max_positions: int = 3  # stacked pyramid levels per symbol
    pyramid_min_bars: int = 5  # min bars between adds

    # Exit
    trend_exit: bool = True  # close all on trend-bias flip
    take_profit_atr: float = 0.0  # optional TP distance in ATR (0 = disabled)

    # Warmup
    warmup_bars: int = 150


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def _blank() -> dict:
    return {
        # per-symbol persistence for pyramid gating
        "last_entry_price": {},
        "last_add_bar": {},
        "bar_idx": {},
    }


GLOBAL: dict = _blank()


def reset_global() -> None:
    global GLOBAL
    GLOBAL = _blank()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bias_sep(
    state: BacktestState,
    symbol: str,
    params: Params,
    base_interval: str,
) -> float | None:
    """Signed separation (ref-EMA)/EMA, or None when there isn't enough data.

    ref is the last close (or mean of the last ``bias_smoothing`` closes when
    ``bias_smoothing > 1``). Positive => price above trend, negative => below.
    """
    df = state.candles.get((symbol, base_interval))
    if df is None or len(df) < params.ema_trend_period + 1:
        return None
    closes = cast(pd.Series, df["close"])
    ema_val = float(ta.ema(closes, params.ema_trend_period).iloc[-1])
    if np.isnan(ema_val):
        return None
    ref = (
        closes.tail(params.bias_smoothing).mean()
        if params.bias_smoothing > 1
        else float(closes.iloc[-1])
    )
    return (ref - ema_val) / ema_val


def _bias_is_long(
    state: BacktestState,
    symbol: str,
    params: Params,
    base_interval: str,
) -> bool:
    sep = _bias_sep(state, symbol, params, base_interval)
    if sep is None:
        return False
    return sep > 0


def _compute_atr(df: pd.DataFrame, params: Params) -> float | None:
    atr_val = float(
        ta.atr(
            cast(pd.Series, df["high"]),
            cast(pd.Series, df["low"]),
            cast(pd.Series, df["close"]),
            params.atr_period,
        ).iloc[-1]
    )
    if np.isnan(atr_val) or atr_val <= 0:
        return None
    return atr_val


def _new_extreme(df: pd.DataFrame, params: Params, is_long: bool) -> bool:
    """True when current close broke above/below the prior N-bar extreme."""
    lookback = max(params.entry_lookback + 1, params.atr_period + 2)
    if len(df) < lookback:
        return False
    if is_long:
        prior = float(
            cast(pd.Series, df["high"]).iloc[-params.entry_lookback : -1].max()
        )
        cur = float(cast(pd.Series, df["close"]).iloc[-1])
        return cur > prior if params.breakout_strict else cur >= prior
    prior = float(cast(pd.Series, df["low"]).iloc[-params.entry_lookback : -1].min())
    cur = float(cast(pd.Series, df["close"]).iloc[-1])
    return cur < prior if params.breakout_strict else cur <= prior


def _open_signal(
    state: BacktestState,
    candle: Candle,
    df: pd.DataFrame,
    symbol: str,
    is_long: bool,
    params: Params,
) -> TradeSignal | None:
    atr_val = _compute_atr(df, params)
    if atr_val is None:
        return None
    entry_price = float(cast(pd.Series, df["close"]).iloc[-1])

    stop_dist = params.stop_atr_mult * atr_val
    if stop_dist <= 0:
        return None
    risk_dollars = state.portfolio.cash * params.risk_pct
    qty = round(risk_dollars / stop_dist, 4)
    if qty <= 0:
        return None

    if is_long:
        stop = entry_price - stop_dist
        tp = (
            entry_price + params.take_profit_atr * atr_val
            if params.take_profit_atr > 0
            else None
        )
    else:
        stop = entry_price + stop_dist
        tp = (
            entry_price - params.take_profit_atr * atr_val
            if params.take_profit_atr > 0
            else None
        )

    return TradeSignal(
        action=ActionType.long if is_long else ActionType.short,
        symbol=symbol,
        timestamp=candle.timestamp,
        price=entry_price,
        qty=qty,
        reason=(
            f"[{FEEL}] {'long' if is_long else 'short'} "
            f"{params.entry_lookback}d break EMA{params.ema_trend_period} "
            f"ATR={atr_val:.2f} stop={stop:.2f} risk={params.risk_pct:.1%}"
        ),
        stop_loss=stop,
        take_profit=tp,
    )


def _close_signal(position: Position, candle: Candle, reason: str) -> TradeSignal:
    return TradeSignal(
        action=ActionType.close,
        symbol=position.symbol,
        timestamp=candle.timestamp,
        price=candle.close,
        qty=abs(position.qty),
        position_id=position.position_id,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []

    # The engine calls on_candle only for base-interval bars (HTF-only bars
    # skip the pipeline). candle.interval is therefore the signal/base interval
    # (e.g. "1d", "1h") — no longer a hardcoded "1d".
    base_interval = candle.interval if candle.interval is not None else "1d"

    symbols: list[str] = sorted({k[0] for k in state.candles.keys()})

    for symbol in symbols:
        df = state.candles.get((symbol, base_interval))
        if df is None or len(df) < params.warmup_bars:
            continue

        closes = cast(pd.Series, df["close"])
        bar_idx = GLOBAL["bar_idx"].get(symbol, 0)
        GLOBAL["bar_idx"][symbol] = bar_idx + 1

        raw_is_long = _bias_is_long(state, symbol, params, base_interval)
        # Side gating: the bias is always derived from the raw EMA slope but is
        # masked per side. Disabling a side sets its bias False (no entries and,
        # via the trend-flip exit, any existing position on that side is flat)
        # while the other side keeps following the raw trend. Asymmetric configs
        # are thus clean long-only / short-only mirrors of the both-sides mode.
        is_long_bias = params.allow_long and raw_is_long
        is_short_bias = params.allow_short and not raw_is_long

        # Optional symmetric dead-zone (min_trend_pct > 0): a side only trades when
        # price clearly clears the EMA by this % ; inside the band both sides are
        # off, so positions flatten and no new entries fire on chop near fair value.
        if params.min_trend_pct > 0:
            sep = _bias_sep(state, symbol, params, base_interval)
            if sep is not None:
                is_long_bias = is_long_bias and sep > params.min_trend_pct
                is_short_bias = is_short_bias and sep < -params.min_trend_pct

        positions = state.portfolio.positions.get(symbol, ())

        # -- exit path: trend flip closes every stacked level ---------------
        if positions and params.trend_exit:
            head_is_long = positions[0].type == ActionType.long
            flip_long = head_is_long and not is_long_bias
            flip_short = not head_is_long and not is_short_bias
            if flip_long or flip_short:
                for pos in positions:
                    signals.append(
                        _close_signal(
                            pos,
                            candle,
                            f"[{FEEL}] trend flip — exit "
                            f"{'long' if pos.type == ActionType.long else 'short'}",
                        )
                    )
                continue

        # -- entry / pyramid path -------------------------------------------
        active = len(positions)
        if active >= params.max_positions:
            continue  # pyramid full

        if active > 0:
            last_add = GLOBAL["last_add_bar"].get(symbol, -10_000)
            if bar_idx - last_add < params.pyramid_min_bars:
                continue  # not enough bars since last add

        cur = float(closes.iloc[-1])
        last = GLOBAL["last_entry_price"].get(symbol, cur)

        if is_long_bias and _new_extreme(df, params, is_long=True):
            # pyramid must ladder strictly higher than the prior level
            if active > 0 and cur < last:
                continue
            sig = _open_signal(state, candle, df, symbol, True, params)
            if sig is not None:
                signals.append(sig)
                GLOBAL["last_entry_price"][symbol] = cur
                GLOBAL["last_add_bar"][symbol] = bar_idx
        elif is_short_bias and _new_extreme(df, params, is_long=False):
            # pyramid must ladder strictly lower than the prior level
            if active > 0 and cur > last:
                continue
            sig = _open_signal(state, candle, df, symbol, False, params)
            if sig is not None:
                signals.append(sig)
                GLOBAL["last_entry_price"][symbol] = cur
                GLOBAL["last_add_bar"][symbol] = bar_idx

    return signals
