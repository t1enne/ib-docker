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

import numpy as np
import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import close, open
from src.bt.types import PlotConfig
from src.bt.regime.types import REGIME_INT_TO_LABEL, RegimeLabel

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


def _current_regime(state: BacktestState) -> RegimeLabel | None:
    """Extract current regime label from model state."""
    regime_int = state.model_state.current_regime
    if regime_int is None:
        return None
    return REGIME_INT_TO_LABEL.get(regime_int)


def _should_go_long(regime: RegimeLabel | None, params: Params) -> bool:
    """Check if regime allows long entries."""
    if regime is None:
        return not params.regime_unknown_flat
    if regime == "BULL":
        return params.regime_long
    if regime == "BEAR":
        return False  # don't buy in bear
    # RANGE
    return not params.regime_range_flat


def _should_go_short(regime: RegimeLabel | None, params: Params) -> bool:
    """Check if regime allows short entries."""
    if regime is None:
        return not params.regime_unknown_flat
    if regime == "BEAR":
        return params.regime_short
    if regime == "BULL":
        return False
    return not params.regime_range_flat


def _regime_size_mult(regime: RegimeLabel | None, params: Params) -> float:
    """Return position size multiplier based on regime."""
    if regime is None:
        return 1.0
    if regime == "BULL":
        return params.size_bull
    if regime == "BEAR":
        return params.size_bear
    return params.size_high_vol  # RANGE → conservative


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
    spread_pct = abs(sma_fast - sma_slow) / sma_slow

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

    regime = _current_regime(state)
    _, _, crossed_up, crossed_down = _sma_cross(
        closes, params.fast, params.slow, params
    )
    size_mult = _regime_size_mult(regime, params)

    # ---- Exit in-position ----
    if position is not None:
        if position.type == ActionType.long and crossed_down:
            return close(candle, position, f"[{regime or '?'}] sma cross down")
        if position.type == ActionType.short and crossed_up:
            return close(candle, position, f"[{regime or '?'}] sma cross up")
        # Also exit if regime turns hostile
        if position.type == ActionType.long and regime == "BEAR":
            return close(candle, position, f"[{regime}] exit long in bear")
        if position.type == ActionType.short and regime == "BULL":
            return close(candle, position, f"[{regime}] exit short in bull")
        if regime == "RANGE" and params.regime_range_flat:
            return close(candle, position, f"[RANGE] flat")
        return []

    # ---- Entry ----
    if (
        crossed_up
        and _should_go_long(regime, params)
        and _momentum_ok(closes, params, "long")
    ):
        return open(
            candle,
            ActionType.long,
            f"[{regime or '?'}] mom cross up ({size_mult:.1f}x)",
        )

    if (
        crossed_down
        and _should_go_short(regime, params)
        and _momentum_ok(closes, params, "short")
    ):
        return open(
            candle,
            ActionType.short,
            f"[{regime or '?'}] mom cross down ({size_mult:.1f}x)",
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

        # Regime as subplot (if available)
        regime_int = state.model_state.current_regime
        if regime_int is not None:
            subplots.append(("regime", pd.Series(regime_int, index=closes.index[:1])))

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
