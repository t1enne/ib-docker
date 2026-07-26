"""Factory functions for creating model updaters with regime detection.

A model updater is a ModelUpdaterFn: (BacktestState, Candle) -> BacktestState.
It runs on every candle to update ModelState fields like current_regime.

Two modes:
  create_regime_model_updater  — uses a RegimeDetector Protocol (batch)
  create_hmm_online_updater    — uses MarketRegimeHMMOnline directly (fast)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Optional

import pandas as pd

from src.bt.regime.types import RegimeDetector
from src.bt.state import BacktestState, Candle
from src.bt.engine.utils import merge_bt_state


def create_regime_model_updater(
    regime_detector: RegimeDetector,
    price_col: str = "close",
    symbol: Optional[str] = None,
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Create a ModelUpdaterFn that sets current_regime on each candle.

    The regime detector is called once per symbol per candle tick.
    For multi-symbol strategies, regime is updated for all symbols
    on the last symbol's tick (when state.candles has all rows).

    Args:
        regime_detector: Any RegimeDetector callable
        price_col: Column name for price data (default: "close")
        symbol: If single-symbol, skip the "last symbol" gating.
                If None (multi-symbol), update only on last symbol tick.

    Returns:
        ModelUpdaterFn: (state, candle) -> state
    """

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        if not state.candles:
            return state

        # Determine which symbol to use for regime detection
        sym = symbol if symbol else candle.symbol
        candles_df = state.candles.get(sym)
        if candles_df is None or len(candles_df) < 20:
            return state

        try:
            regimes = regime_detector(candles_df)
        except Exception:
            return state

        if len(regimes) == 0:
            return state

        current = regimes.iloc[-1]
        if pd.isna(current) or current == -1:
            return state

        new_ms = replace(state.model_state, current_regime=int(current))
        return merge_bt_state(state, dict(model_state=new_ms))

    return update


def create_regime_model_updater_for_symbols(
    regime_detector: RegimeDetector,
    symbols: list[str],
    price_col: str = "close",
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Like create_regime_model_updater, but updates regimes for
    the last candle's symbol (used in multi-symbol backtests)."""

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        if not state.candles:
            return state

        # Only update on last symbol tick to have full state.candles built
        if candle.symbol != (symbols[-1] if symbols else None):
            return state

        new_ms = state.model_state
        for sym in symbols:
            candles_df = state.candles.get(sym)
            if candles_df is None or len(candles_df) < 20:
                continue
            try:
                regimes = regime_detector(candles_df)
            except Exception:
                continue
            if len(regimes) == 0:
                continue
            current = regimes.iloc[-1]
            if pd.isna(current) or current == -1:
                continue
            new_ms = replace(new_ms, current_regime=int(current))

        if new_ms is not state.model_state:
            return merge_bt_state(state, dict(model_state=new_ms))
        return state

    return update


# ---------------------------------------------------------------------------
# Online HMM updater — uses MarketRegimeHMMOnline (O(1) per tick)
# ---------------------------------------------------------------------------


def create_hmm_online_updater(
    n_regimes: int = 3,
    window_size: int = 500,
    vol_window: int = 20,
    momentum_window: int = 10,
    retrain_interval: int = 50,
    random_state: int = 42,
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Create a ModelUpdaterFn using the online HMM.

    Uses MarketRegimeHMMOnline which maintains a rolling window and
    refits periodically. Per-tick cost is O(1) — just feature
    extraction + single-row predict. Refit is O(window_size) and
    runs every retrain_interval bars.

    Args:
        n_regimes: Number of hidden states
        window_size: Rolling window for HMM fitting
        vol_window: Window for volatility feature
        momentum_window: Window for momentum feature
        retrain_interval: Refit model every N bars
        random_state: Reproducibility seed

    Returns:
        ModelUpdaterFn: (state, candle) -> state
    """
    from src.indicators.hmm.online import MarketRegimeHMMOnline

    hmm = MarketRegimeHMMOnline(
        n_regimes=n_regimes,
        window_size=window_size,
        vol_window=vol_window,
        momentum_window=momentum_window,
        retrain_interval=retrain_interval,
        random_state=random_state,
    )

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        regime = hmm.update(candle.close)
        if regime < 0:
            return state
        new_ms = replace(state.model_state, current_regime=regime)
        return merge_bt_state(state, dict(model_state=new_ms))

    return update
