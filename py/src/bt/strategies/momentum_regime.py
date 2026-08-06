"""Momentum strategy with regime filter.

Core idea: trend-following on SMA crossover, gated by regime.
- BULL: long-only aggressive entries
- BEAR: short-only entries
- RANGE: stand aside
- HIGH VOL (HMM high-vol regime): reduced position size

Uses state.model_state.current_regime from the regime model updater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import pandas as pd

from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    Position,
    TradeSignal,
)
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import close, open

from src.bt.regime.gates import TrendGate, current_trend, current_vol
from src.bt.regime.types import VolRegime

STRATEGY_TYPE = "momentum_regime"


# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # SMA crossover
    fast: int = 20
    slow: int = 50

    # Momentum filter: require N-day return > threshold to enter
    momentum_lookback: int = 20
    momentum_threshold: float = 0.02  # 2% over lookback

    # Exit: fast crosses below slow
    exit_threshold_pct: float = 0.0  # minimal spread to avoid whipsaw

    # Regime gating
    regime_long: bool = True  # go long in BULL
    regime_short: bool = True  # go short in BEAR
    regime_range_flat: bool = True  # stay flat in RANGE
    regime_unknown_flat: bool = True  # stay flat when regime is None (warmup)

    # Position sizing by regime
    position_size_pct: float = 0.2  # base % of cash per position
    size_bull: float = 1.0
    size_bear: float = 1.0
    size_high_vol: float = 0.5

    # Warmup
    warmup_bars: int = 60


# ---------------------------------------------------------------------------
# regime helpers — thin wrappers over src.bt.regime.gates
# ---------------------------------------------------------------------------


def _regime_size_mult(vol: VolRegime | None, params: Params) -> float:
    """Return position size multiplier based on vol regime."""
    if vol is None:
        return 1.0
    if vol == "HIGH_VOL":
        return params.size_high_vol
    return 1.0


# ---------------------------------------------------------------------------
# signal logic
# ---------------------------------------------------------------------------


def _momentum_ok(closes: pd.Series, params: Params, direction: str) -> bool:
    """Check N-day momentum aligns with trade direction."""
    if len(closes) < params.momentum_lookback + 1:
        return False

    past = closes.iloc[-(params.momentum_lookback + 1)]
    current = closes.iloc[-1]
    ret = (current - past) / past

    if direction == "long":
        return ret > params.momentum_threshold
    else:
        return ret < -params.momentum_threshold


def _sma_cross(
    closes: pd.Series,
    fast_window: int,
    slow_window: int,
    params: Params,
) -> tuple[float, float, bool, bool]:
    """Compute SMAs and cross signals."""
    if len(closes) < slow_window:
        return (0.0, 0.0, False, False)

    sma_fast = closes.rolling(fast_window).mean().iloc[-1]
    sma_slow = closes.rolling(slow_window).mean().iloc[-1]

    # Two-bar cross detection (fast was below, now above = bullish cross)
    if len(closes) >= slow_window + 1:
        sma_fast_prev = closes.rolling(fast_window).mean().iloc[-2]
        sma_slow_prev = closes.rolling(slow_window).mean().iloc[-2]
        crossed_up = sma_fast_prev <= sma_slow_prev and sma_fast > sma_slow
        crossed_down = sma_fast_prev >= sma_slow_prev and sma_fast < sma_slow
    else:
        crossed_up = sma_fast > sma_slow
        crossed_down = sma_fast < sma_slow

    return (sma_fast, sma_slow, crossed_up, crossed_down)


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    symbol = candle.symbol
    pos_tup = state.portfolio.positions.get(symbol, ())
    position: Optional[Position] = pos_tup[0] if pos_tup else None
    candles_df = state.candles.get((symbol, candle.interval or "1h"))

    if candles_df is None or len(candles_df) < params.warmup_bars:
        return []

    closes = cast(pd.Series, candles_df["close"])
    if len(closes) < params.slow:
        return []

    trend = current_trend(state)
    vol = current_vol(state)
    gate = TrendGate(trend)
    _, _, crossed_up, crossed_down = _sma_cross(
        closes, params.fast, params.slow, params
    )
    size_mult = _regime_size_mult(vol, params)
    price = float(closes.iloc[-1])
    base_qty = (state.portfolio.cash * params.position_size_pct) / price
    qty = round(base_qty * size_mult, 4)

    # ---- Exit in-position ----
    if position is not None:
        direction = "long" if position.type == ActionType.long else "short"
        if position.type == ActionType.long and crossed_down:
            return close(candle, position, f"[{trend or '?'}] sma cross down")
        if position.type == ActionType.short and crossed_up:
            return close(candle, position, f"[{trend or '?'}] sma cross up")
        # Also exit if regime turns hostile
        if gate.hostile_to(direction, allow_range=params.regime_range_flat):
            if gate.bear:
                reason = "[BEAR] exit long"
            elif gate.bull:
                reason = "[BULL] exit short"
            else:
                reason = "[RANGE] flat"
            return close(candle, position, reason)
        return []

    # ---- Entry ----
    if (
        crossed_up
        and gate.allows_long(
            allow_bull=params.regime_long,
            allow_range=not params.regime_range_flat,
            allow_unknown=not params.regime_unknown_flat,
        )
        and _momentum_ok(closes, params, "long")
    ):
        return open(
            candle,
            ActionType.long,
            qty,
            f"[{trend or '?'}] mom cross up ({size_mult:.1f}x)",
        )

    if (
        crossed_down
        and gate.allows_short(
            allow_bear=params.regime_short,
            allow_range=not params.regime_range_flat,
            allow_unknown=not params.regime_unknown_flat,
        )
        and _momentum_ok(closes, params, "short")
    ):
        return open(
            candle,
            ActionType.short,
            qty,
            f"[{trend or '?'}] mom cross down ({size_mult:.1f}x)",
        )

    return []
