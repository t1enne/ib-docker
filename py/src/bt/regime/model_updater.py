"""Factory functions for creating model updaters with regime detection.

A model updater is a ModelUpdaterFn: (BacktestState, Candle) -> BacktestState.
It runs on every candle to update ModelState fields.

Modes:
  create_regime_model_updater   — single RegimeDetector → current_regime (legacy)
  create_hmm_online_updater     — online HMM → current_regime (legacy, vol labels)
  create_dual_online_updater    — SMA trend + online HMM vol → current_trend + current_vol

HTF support:
  trend_bar / vol_bar parameters let you run regime detection on higher-timeframe
  bars while trading on lower timeframes. For example, bar="1h" + trend_bar="1d"
  computes trend from daily SMA while entries/exits happen intraday.
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

    Legacy — prefer create_dual_online_updater for new strategies.
    """

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        if not state.candles:
            return state

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
    """Multi-symbol variant of create_regime_model_updater."""

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        if not state.candles:
            return state

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
# Dual updater: trend (SMA) + vol (HMM online) with HTF support
# ---------------------------------------------------------------------------


def create_dual_online_updater(
    n_regimes: int = 3,
    window_size: int = 252,
    vol_window: int = 20,
    momentum_window: int = 10,
    retrain_interval: int = 50,
    random_state: int = 42,
    trend_fast: int = 50,
    trend_slow: int = 200,
    range_threshold_pct: float = 0.005,
    trend_bar: str | None = None,
    vol_bar: str | None = None,
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Create a ModelUpdaterFn that sets both current_trend and current_vol.

    Trend: SMA crossover (fast/slow MA) — O(1) per tick.
           If trend_bar is set, SMA reads from HTF data for that interval.
    Vol:   online HMM (vol-ranked) — O(1) per tick, refit every N bars.
           If vol_bar is set, HMM receives HTF closes for that interval.

    Example: bar="1h" + trend_bar="1d" → trend from daily SMA, vol from hourly.
    """
    from src.indicators.hmm.online import MarketRegimeHMMOnline
    from src.bt.regime.detectors import TREND_LABEL_TO_INT

    hmm = MarketRegimeHMMOnline(
        n_regimes=n_regimes,
        window_size=window_size,
        vol_window=vol_window,
        momentum_window=momentum_window,
        retrain_interval=retrain_interval,
        random_state=random_state,
    )

    # Cache materialized HTF DataFrames to avoid rebuilding on every tick
    _htf_cache: dict[str, pd.DataFrame] = {}
    _htf_cache_len: dict[str, int] = {}

    def _get_htf_df(state: BacktestState, bar_key: str) -> pd.DataFrame | None:
        rows = state.htf_data.get(bar_key)
        if rows is None:
            return None
        if isinstance(rows, pd.DataFrame):
            return rows if not rows.empty else None
        # List of dicts — rebuild only if it grew
        n = len(rows)
        if _htf_cache_len.get(bar_key) == n and bar_key in _htf_cache:
            return _htf_cache[bar_key]
        df = pd.DataFrame(rows).set_index(["symbol", "timestamp"])
        _htf_cache[bar_key] = df
        _htf_cache_len[bar_key] = n
        return df

    def _htf_close(state: BacktestState, bar_key: str) -> float | None:
        df = _get_htf_df(state, bar_key)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    def _htf_closes(state: BacktestState, sym: str, bar_key: str) -> pd.Series | None:
        df = _get_htf_df(state, bar_key)
        if df is None or df.empty or sym not in df.index.get_level_values("symbol"):
            return None
        sym_df = df.xs(sym, level="symbol")
        if sym_df.empty:
            return None
        return sym_df["close"]

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        # --- Vol regime ---
        vol_close: float | None = None
        if vol_bar:
            vol_close = _htf_close(state, vol_bar)
        if vol_close is None:
            vol_close = candle.close
        vol_regime = hmm.update(vol_close)

        # --- Trend regime ---
        sym = candle.symbol
        trend_closes: pd.Series | None = None

        if trend_bar:
            trend_closes = _htf_closes(state, sym, trend_bar)

        if trend_closes is None:
            candles_df = state.candles.get(sym)
            if candles_df is not None:
                trend_closes = candles_df["close"]

        trend_regime: int | None = None
        if trend_closes is not None and len(trend_closes) >= trend_slow:
            fast_sma = trend_closes.rolling(trend_fast).mean().iloc[-1]
            slow_sma = trend_closes.rolling(trend_slow).mean().iloc[-1]
            spread = abs(fast_sma - slow_sma) / slow_sma

            if spread <= range_threshold_pct:
                trend_regime = TREND_LABEL_TO_INT["RANGE"]
            elif fast_sma > slow_sma:
                trend_regime = TREND_LABEL_TO_INT["BULL"]
            else:
                trend_regime = TREND_LABEL_TO_INT["BEAR"]

        new_ms = state.model_state
        if vol_regime >= 0:
            new_ms = replace(new_ms, current_vol=vol_regime)
        if trend_regime is not None:
            new_ms = replace(new_ms, current_trend=trend_regime)

        if new_ms is not state.model_state:
            return merge_bt_state(state, dict(model_state=new_ms))
        return state

    return update


# ---------------------------------------------------------------------------
# Online HMM updater (legacy — sets current_regime)
# ---------------------------------------------------------------------------


def create_hmm_online_updater(
    n_regimes: int = 3,
    window_size: int = 500,
    vol_window: int = 20,
    momentum_window: int = 10,
    retrain_interval: int = 50,
    random_state: int = 42,
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Create a ModelUpdaterFn using the online HMM (legacy).

    Sets state.model_state.current_regime with vol-ranked HMM labels.
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
