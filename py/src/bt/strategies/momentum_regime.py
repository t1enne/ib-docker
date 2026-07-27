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
from typing import cast

import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import close, open
from src.bt.types import PlotConfig
from src.bt.regime.types import (
    TREND_INT_TO_LABEL,
    VOL_INT_TO_LABEL,
    TrendRegime,
    VolRegime,
)

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
    size_bull: float = 1.0
    size_bear: float = 1.0
    size_high_vol: float = 0.5

    # Warmup
    warmup_bars: int = 60


# ---------------------------------------------------------------------------
# regime helpers
# ---------------------------------------------------------------------------


def _current_trend(state: BacktestState) -> TrendRegime | None:
    """Extract current trend regime label from model state."""
    trend_int = state.model_state.current_trend
    if trend_int is None:
        return None
    return TREND_INT_TO_LABEL.get(trend_int)


def _current_vol(state: BacktestState) -> VolRegime | None:
    """Extract current vol regime label from model state."""
    vol_int = state.model_state.current_vol
    if vol_int is None:
        return None
    return VOL_INT_TO_LABEL.get(vol_int)


def _should_go_long(trend: TrendRegime | None, params: Params) -> bool:
    """Check if trend regime allows long entries."""
    if trend is None:
        return not params.regime_unknown_flat
    if trend == "BULL":
        return params.regime_long
    if trend == "BEAR":
        return False  # don't buy in bear
    # RANGE
    return not params.regime_range_flat


def _should_go_short(trend: TrendRegime | None, params: Params) -> bool:
    """Check if trend regime allows short entries."""
    if trend is None:
        return not params.regime_unknown_flat
    if trend == "BEAR":
        return params.regime_short
    if trend == "BULL":
        return False
    return not params.regime_range_flat


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
    position = state.portfolio.positions.get(symbol)
    candles_df = state.candles.get(symbol)

    if candles_df is None or len(candles_df) < params.warmup_bars:
        return []

    closes = cast(pd.Series, candles_df["close"])
    if len(closes) < params.slow:
        return []

    trend = _current_trend(state)
    vol = _current_vol(state)
    _, _, crossed_up, crossed_down = _sma_cross(
        closes, params.fast, params.slow, params
    )
    size_mult = _regime_size_mult(vol, params)

    # ---- Exit in-position ----
    if position is not None:
        if position.type == ActionType.long and crossed_down:
            return close(candle, position, f"[{trend or '?'}] sma cross down")
        if position.type == ActionType.short and crossed_up:
            return close(candle, position, f"[{trend or '?'}] sma cross up")
        # Also exit if regime turns hostile
        if position.type == ActionType.long and trend == "BEAR":
            return close(candle, position, f"[{trend}] exit long in bear")
        if position.type == ActionType.short and trend == "BULL":
            return close(candle, position, f"[{trend}] exit short in bull")
        if trend == "RANGE" and params.regime_range_flat:
            return close(candle, position, "[RANGE] flat")
        return []

    # ---- Entry ----
    if (
        crossed_up
        and _should_go_long(trend, params)
        and _momentum_ok(closes, params, "long")
    ):
        return open(
            candle,
            ActionType.long,
            f"[{trend or '?'}] mom cross up ({size_mult:.1f}x)",
        )

    if (
        crossed_down
        and _should_go_short(trend, params)
        and _momentum_ok(closes, params, "short")
    ):
        return open(
            candle,
            ActionType.short,
            f"[{trend or '?'}] mom cross down ({size_mult:.1f}x)",
        )

    return []


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot(state: BacktestState, config: object) -> PlotConfig:
    from src.bt.types import StrategyConfig as SC

    strategy_config = cast(SC, config)

    if not strategy_config.strategy_params:
        return PlotConfig()

    params = Params.from_dict(strategy_config.strategy_params)
    price_overlays: dict[str, dict[str, pd.Series]] = {}
    subplots: list[tuple[str, pd.Series]] = []

    for symbol in strategy_config.symbols:
        candles_df = state.candles.get(symbol)
        if candles_df is None or len(candles_df) < params.slow:
            continue

        closes = cast(pd.Series, candles_df["close"])
        sma_fast = closes.rolling(params.fast).mean()
        sma_slow = closes.rolling(params.slow).mean()

        price_overlays[symbol] = {
            f"sma_{params.fast}": sma_fast,
            f"sma_{params.slow}": sma_slow,
        }

        # Trend regime as subplot (if available)
        trend_int = state.model_state.current_trend
        if trend_int is not None:
            subplots.append(("regime", pd.Series(trend_int, index=closes.index[:1])))

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
