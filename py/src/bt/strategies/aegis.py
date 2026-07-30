"""AEGIS — Adaptive Equity Generation and Immunisation System.

Core: risk-adjusted momentum filter → minimax correlation selection
→ equal-weight allocation, gated by market regime.

Rebalances monthly. Requires dual_online model_updater for trend regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams
from src.bt.regime.types import TREND_INT_TO_LABEL

STRATEGY_TYPE = "aegis"


# ---------------------------------------------------------------------------
# Typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Momentum filter
    mom_lookback: int = 63  # days
    vol_window: int = 20
    mom_threshold: float = 0.01  # min risk-adjusted momentum

    # Correlation filter
    corr_lookback: int = 60
    max_corr: float = 0.65  # max pairwise

    # Allocation
    min_held: int = 3
    max_held: int = 5
    min_candidates: int = 3  # need at least N qualifying symbols to trade

    # Regime — trade only in BULL, stand aside otherwise
    flat_in_bear: bool = True
    flat_in_range: bool = True

    # Schedule
    warmup: int = 260
    rebalance_offset: str = "ME"  # month-end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ram(closes: pd.Series, mom_lb: int, vol_w: int) -> float:
    """Risk-adjusted momentum: ROC / rolling vol."""
    if len(closes) < max(mom_lb, vol_w) + 1:
        return -np.inf
    ret = (closes.iloc[-1] / closes.iloc[-(mom_lb + 1)]) - 1.0
    vol = closes.pct_change().rolling(vol_w).std().iloc[-1]
    if pd.isna(vol) or vol == 0:
        return -np.inf
    return float(ret / vol)


def _minimax_select(
    symbols: list[str],
    corr: pd.DataFrame,
    max_corr: float,
    min_n: int,
    max_n: int,
) -> list[str]:
    """Greedy: seed with lowest avg correlation, add while max_pairwise < threshold."""
    present = [s for s in symbols if s in corr.columns]
    if len(present) <= min_n:
        return present[:max_n]

    sub = corr.loc[present, present]
    avg = sub.mean().sort_values()
    selected = [avg.index[0]]
    remaining = [s for s in present if s != selected[0]]

    while remaining and len(selected) < max_n:
        best = None
        best_max = 1.0
        for s in remaining:
            mx = max(abs(sub.loc[s, sel]) for sel in selected if sel in sub.columns)
            if mx < best_max:
                best_max = mx
                best = s
        if best is None or (best_max > max_corr and len(selected) >= min_n):
            break
        if best is not None:
            selected.append(best)
            remaining.remove(best)

    return selected


def _trend(state: BacktestState) -> str | None:
    t = state.model_state.current_trend
    return TREND_INT_TO_LABEL.get(t) if t is not None else None


# ---------------------------------------------------------------------------
# Rebalance gate (module-level mutable — ok for single-run CLI)
# ---------------------------------------------------------------------------

_last: dict[str, pd.Timestamp] = {}


def _is_month_end(ts: pd.Timestamp) -> bool:
    return ts.is_month_end


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    # ---- Warmup check ----
    first = next(iter(state.candles.values()), None)
    if first is None or len(first) < params.warmup:
        return _close_all(state, candle, "warmup")

    # ---- Rebalance gate ----
    cache_key = "aegis"
    ts = candle.timestamp
    if not _is_month_end(ts):
        return []
    if _last.get(cache_key) is not None and _last[cache_key] == ts:  # type: ignore[union-attr]
        return []
    _last[cache_key] = ts

    # ---- Regime gate ----
    regime = _trend(state)
    if regime == "BEAR" and params.flat_in_bear:
        return _close_all(state, candle, "[BEAR] flat")
    if regime == "RANGE" and params.flat_in_range:
        return _close_all(state, candle, "[RANGE] flat")

    interval = candle.interval or "1h"
    symbols = sorted({s for s, _ in state.candles})

    # ---- Step 1: RAM filter ----
    scores: dict[str, float] = {}
    for sym in symbols:
        df = state.candles.get((sym, interval))
        if df is None or len(df) < max(params.mom_lookback, params.vol_window) + 1:
            continue
        s = _ram(cast(pd.Series, df["close"]), params.mom_lookback, params.vol_window)
        if s > params.mom_threshold:
            scores[sym] = s

    qualifying = sorted(scores, key=lambda s: scores[s], reverse=True)
    if len(qualifying) < params.min_candidates:
        return _close_all(state, candle, "no candidates")

    # ---- Step 2: correlation matrix ----
    rets: dict[str, pd.Series] = {}
    for sym in qualifying:
        df = state.candles.get((sym, interval))
        if df is None:
            continue
        r = cast(pd.Series, df["close"]).pct_change().dropna()
        if len(r) >= params.corr_lookback:
            rets[sym] = r
    if len(rets) < params.min_held:
        return _close_all(state, candle, "no returns data")

    corr = pd.concat(rets, axis=1).iloc[-params.corr_lookback :].corr()

    # ---- Step 3: minimax selection ----
    selected = _minimax_select(
        qualifying, corr, params.max_corr, params.min_held, params.max_held
    )
    if len(selected) < params.min_held:
        return _close_all(state, candle, "few selected")

    # ---- Step 4: equal-weight allocation ----
    w = 1.0 / len(selected)
    held = set(state.portfolio.positions.keys())
    target = set(selected)

    signals: list[TradeSignal] = []

    # Close deselected
    for sym in held - target:
        pos = state.portfolio.positions.get(sym)
        if pos is None:
            continue
        signals.append(
            TradeSignal(
                action=ActionType.close,
                symbol=sym,
                timestamp=ts,
                price=candle.close,
                qty=abs(pos.qty),
                reason=f"deselected [{regime or '?'}]",
            )
        )

    # Open new
    capital = state.portfolio.cash + sum(
        p.last_price * p.qty for p in state.portfolio.positions.values()
    )
    for sym in target - held:
        qty = (capital * w) / candle.close if candle.close > 0 else 0.0
        if qty <= 1e-8:
            continue
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=ts,
                price=candle.close,
                qty=qty,
                reason=f"alloc {w:.1%} [{regime or '?'}]",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# Close-all helper
# ---------------------------------------------------------------------------


def _close_all(state: BacktestState, candle: Candle, reason: str) -> list[TradeSignal]:
    return [
        TradeSignal(
            action=ActionType.close,
            symbol=sym,
            timestamp=candle.timestamp,
            price=candle.close,
            qty=abs(pos.qty),
            reason=reason,
        )
        for sym, pos in state.portfolio.positions.items()
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(state: BacktestState, config: object) -> object:
    from src.bt.types import StrategyConfig as SC, PlotConfig

    strategy_config = cast(SC, config)
    if not strategy_config.strategy_params:
        return PlotConfig()

    params = Params.from_dict(strategy_config.strategy_params)
    overlays: dict[str, dict[str, pd.Series]] = {}

    for symbol in strategy_config.symbols:
        df = state.candles.get((symbol, strategy_config.bar))
        if df is None or len(df) < params.mom_lookback:
            continue
        closes = cast(pd.Series, df["close"])
        mom = closes.pct_change(params.mom_lookback)
        vol = closes.pct_change().rolling(params.vol_window).std()
        overlays[symbol] = {"ram": mom / vol.replace(0, np.nan)}

    return PlotConfig(price_overlays=overlays)
