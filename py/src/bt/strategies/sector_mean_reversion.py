"""Sector rotation: buy worst 6-month performers, sell when they recover.

Core idea: sector ETFs mean-revert over 6-12 month horizons.
Rank sectors by past 6-month return (skipping last month).
Go long the worst N performers. Exit when a position's rank
improves to the top K (it's no longer a "loser").

Daily bars. No trend filter — we want to catch falling knives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "sector_mean_reversion"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Mean-reversion: rank by returns from lookback_end to lookback_start days ago
    momentum_lookback: int = 126  # 6 months
    skip_recent: int = 21  # skip last month to avoid short-term reversal noise

    # Entry: buy the worst N ranked symbols
    top_n: int = 2
    max_positions: int = 3
    max_positions_bear: int = 1

    # Exit: close when rank improves to this threshold (1 = best)
    exit_rank_threshold: int = 3

    # Warmup
    warmup_bars: int = 150

    # Cooldown after any exit
    cooldown_bars: int = 10

    # Regime (optional)
    regime_filter: str = ""  # "SPY" for bear regime gating
    bar: str = "1d"
    regime_sma: int = 200

    # obsolete / kept for YAML compat — ignored
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    velocity_smooth: int = 8
    drop_rank: int = 5
    cooldown_vel_exit_bars: int = 24
    exit_velocity_threshold: float = -2.0
    vol_filter_window: int = 20
    vol_spike_mult: float = 2.5
    trend_sma: int = 50
    require_above_trend: bool = False
    momentum_lookbacks: tuple = ()
    momentum_weights: tuple = ()

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        d = dict(d)
        if "exit_rank_threshold" not in d and "drop_rank" in d:
            d["exit_rank_threshold"] = d["drop_rank"]
        return super().from_dict(d)


# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------

_COOLDOWNS: dict[str, int] = {}
_REGIME_CACHE: dict[str, bool] = {}
_REGIME_TS: pd.Timestamp | None = None
_REGIME_CANDLES_CACHE: dict[str, pd.DataFrame | None] = {}


def _reset_state() -> None:
    global _COOLDOWNS, _REGIME_CACHE, _REGIME_TS
    _COOLDOWNS = {}
    _REGIME_CACHE = {}
    _REGIME_TS = None


# ---------------------------------------------------------------------------
# scoring — rank by past returns (LOWEST = best for mean-reversion)
# ---------------------------------------------------------------------------


def _score_symbol(closes: pd.Series, params: Params) -> float | None:
    """Return past return over [lookback, skip_recent]. Lower = more beaten down."""
    needed = params.momentum_lookback + 1
    if len(closes) < needed:
        return None

    past_idx = len(closes) - params.momentum_lookback - 1
    recent_idx = len(closes) - params.skip_recent - 1
    if recent_idx < 0:
        recent_idx = 0

    past_price = float(closes.iloc[past_idx])
    recent_price = float(closes.iloc[recent_idx])
    if past_price <= 0:
        return None

    return (recent_price - past_price) / past_price


def _rank_symbols(state: BacktestState, params: Params) -> dict[str, float]:
    """Score every symbol. Lower score = more beaten down = better buy."""
    scores: dict[str, float] = {}
    for sym in state.candles:
        closes = cast(pd.Series, state.candles[sym]["close"])
        s = _score_symbol(closes, params)
        if s is not None:
            scores[sym] = s
    return scores


def _rank_of(symbol: str, rankings: dict[str, float]) -> int:
    """1 = best-performing (highest score), N = worst (lowest score)."""
    sorted_syms = sorted(rankings, key=lambda s: rankings[s], reverse=True)
    try:
        return sorted_syms.index(symbol) + 1
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


def _is_bear_regime(
    state: BacktestState,
    regime_symbol: str,
    bar: str,
    sma_window: int,
) -> bool:
    global _REGIME_CACHE, _REGIME_TS

    ts = state.timestamp
    if ts is not None and _REGIME_TS == ts:
        return _REGIME_CACHE.get(regime_symbol, False)

    candles = state.candles.get(regime_symbol)
    if candles is None:
        candles = _load_regime_candles(regime_symbol, bar)
    if candles is None or len(candles) < sma_window:
        _REGIME_CACHE[regime_symbol] = False
        _REGIME_TS = ts
        return False

    closes = cast(pd.Series, candles["close"])
    current = float(closes.iloc[-1])
    sma = float(closes.iloc[-sma_window:].mean())
    result = current < sma
    _REGIME_CACHE[regime_symbol] = result
    _REGIME_TS = ts
    return result


def _load_regime_candles(symbol: str, bar: str) -> pd.DataFrame | None:
    cache_key = f"{symbol}:{bar}"
    if cache_key in _REGIME_CANDLES_CACHE:
        return _REGIME_CANDLES_CACHE[cache_key]

    try:
        from src.bt.data_feed import load_candles

        now = pd.Timestamp.now()
        end = cast(
            pd.Timestamp, now if now is not pd.NaT else pd.Timestamp("2099-01-01")
        )
        df = load_candles(
            [symbol],
            cast(pd.Timestamp, pd.Timestamp("2010-01-01")),
            end,
            bar,
        )
        if df.empty:
            _REGIME_CANDLES_CACHE[cache_key] = None
            return None
        result = df.xs(symbol, axis=1, level=0)
        _REGIME_CANDLES_CACHE[cache_key] = result
        return result
    except Exception:
        _REGIME_CANDLES_CACHE[cache_key] = None
        return None


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    scores = _rank_symbols(state, params)
    if not scores:
        return []

    # Lower score = worse past performance = better buy
    sorted_worst_first = sorted(scores, key=lambda s: scores[s])
    signals: list[TradeSignal] = []
    exiting: set[str] = set()

    # -- Exit: close positions whose rank has improved to top tier --
    for sym, position in list(state.portfolio.positions.items()):
        if sym not in scores:
            continue
        rank = _rank_of(sym, scores)
        # Rank 1=best, so exit when position is now a top performer
        if rank <= params.exit_rank_threshold:
            sym_closes = cast(pd.Series, state.candles[sym]["close"])
            sym_price = float(sym_closes.iloc[-1])
            if np.isnan(sym_price):
                continue
            _COOLDOWNS[sym] = params.cooldown_bars
            exiting.add(sym)
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=sym_price,
                    qty=abs(position.qty),
                    reason=f"[smrv] recovered rank {rank} <= {params.exit_rank_threshold}",
                )
            )

    # -- Max positions --
    if params.regime_filter:
        bear = _is_bear_regime(
            state, params.regime_filter, params.bar, params.regime_sma
        )
        max_pos = params.max_positions_bear if bear else params.max_positions
    else:
        max_pos = params.max_positions

    effective = len(state.portfolio.positions) - len(exiting)

    # -- Entry: buy the worst performers --
    for sym in sorted_worst_first:
        pending = len([s for s in signals if s.action == ActionType.long])
        if effective + pending >= max_pos:
            break

        if sym in state.portfolio.positions and sym not in exiting:
            continue
        if any(s.symbol == sym for s in signals):
            continue

        rank = _rank_of(sym, scores)
        n_total = len(scores)
        # Buy worst performers: ranks near N (= worst). Need rank > (N - top_n)
        if rank <= n_total - params.top_n:
            continue

        # Allow entry even when score >= 0 — cross-sectional relative value matters more
        # than absolute sign. The worst sector among all-positive is still the worst.

        cd = _COOLDOWNS.get(sym, 0)
        if cd > 0:
            _COOLDOWNS[sym] = cd - 1
            continue

        closes = cast(pd.Series, state.candles[sym]["close"])
        if len(closes) < params.warmup_bars:
            continue

        entry_price = float(closes.iloc[-1])
        if np.isnan(entry_price):
            continue

        _COOLDOWNS[sym] = params.cooldown_bars
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=candle.timestamp,
                price=entry_price,
                reason=f"[smrv] rank {rank}/{len(scores)} ret={scores[sym]:.2%}",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot(state: BacktestState, config: object) -> object:
    from src.bt.types import PlotConfig

    return PlotConfig(price_overlays={}, subplots=[])
