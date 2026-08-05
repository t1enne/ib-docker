"""Sector mean reversion with model-regime gating, ATR sizing, trail exit.

Same thesis as sector_mean_reversion: sector ETFs mean-revert. The worst
6-month performers snap back up — but only in a bullish market regime.

  - BULL:  long the worst performers (they mean-revert up).
  - otherwise: stand aside (no new entries).

The regime gates ENTRIES only — open positions are never closed by the
regime; they exit on the trailing stop below (or by SL/TP). So a position
entered in a bull run rides out regardless of later regime flips. This
avoids whipsaw exits near regime boundaries.

Regime gate: SPY close > its `regime_sma` simple moving average, computed
inline from state.candles (SPY is part of config.symbols, last). No
model_updater required — this is a fast O(1) rolling gate. SPY itself is
never traded.

Sizing: ATR-volatility-targeted. qty = cash * risk_pct / (ATR * atr_mult).

Exit: hold-then-trail per position.
  - While a position is still a "loser" (rank worse than exit_rank_threshold)
    we HOLD — dead-cat bounces don't shake us.
  - Once rank improves to <= exit_rank_threshold we arm a trail
    (recent low) and ride the reversion. We exit only when close breaks
    the trail (an upward ratcheting stop), letting winners run instead
    of clipping them at a fixed rank.

The hard stop_loss / take_profit from config still apply at the engine level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "sector_mean_reversion_trail"


@dataclass(frozen=True)
class Params(StrategyParams):
    # Mean-reversion ranking
    momentum_lookback: int = 126  # 6 months
    skip_recent: int = 21  # skip last month
    top_n: int = 2
    max_positions: int = 3

    # Recovery threshold (arms the trail)
    exit_rank_threshold: int = 3
    warmup_bars: int = 150
    cooldown_bars: int = 10

    # Regime instrument (never traded) + SMA trend gate
    regime_symbol: str = "SPY"
    regime_sma: int = 200  # slow SMA for the bull filter
    regime_fast_sma: int = 50  # fast SMA; BULL when fast > slow

    # ATR sizing
    atr_period: int = 14
    atr_mult: float = 2.0
    risk_pct: float = 0.01  # % of cash risked per ATR unit

    # Trail (activated after rank recovery)
    trail_lookback: int = 10  # bars for the trail low


# ---------------------------------------------------------------------------
# module-level state — per-position trail tracking
# Repo GLOBAL-dict convention + reset_global() for the split engine.
# ---------------------------------------------------------------------------

GLOBAL: dict = {
    "cooldown": {},
    "trails": {},
}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldown": {}, "trails": {}}


def start_cooldown(symbol: str, bars: int) -> None:
    GLOBAL["cooldown"][symbol] = bars


# ---------------------------------------------------------------------------
# ranking — score past return (LOWEST = most beaten down)
# ---------------------------------------------------------------------------


def _score_symbol(closes: pd.Series, params: Params) -> float | None:
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


def _rankings(state: BacktestState, params: Params, interval: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sym, _ in state.candles:
        if sym == params.regime_symbol:
            continue  # regime instrument is not a tradable sector
        closes = cast(pd.Series, state.candles[(sym, interval)]["close"])
        s = _score_symbol(closes, params)
        if s is not None:
            scores[sym] = s
    return scores


def _rank_of(symbol: str, rankings: dict[str, float]) -> int:
    sorted_syms = sorted(rankings, key=lambda s: rankings[s], reverse=True)
    try:
        return sorted_syms.index(symbol) + 1
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


def _is_bull_regime(state: BacktestState, params: Params) -> bool:
    """True when SPY fast SMA > slow SMA (inline O(1) rolling gate).

    No model_updater needed — SPY must be in config.symbols.
    """
    df = state.candles.get((params.regime_symbol, "1d"))
    if df is None or len(df) < params.regime_sma:
        return False
    closes = cast(pd.Series, df["close"])
    fast = float(closes.iloc[-params.regime_fast_sma :].mean())
    slow = float(closes.iloc[-params.regime_sma :].mean())
    return fast > slow


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    interval = candle.interval or "1d"
    signals: list[TradeSignal] = []
    bull = _is_bull_regime(state, params)

    rankings = _rankings(state, params, interval)
    if not rankings:
        return []

    # ---- Exit: per-position, hold-then-trail only (regime does NOT exit) ----
    for sym, pos_tup in list(state.portfolio.positions.items()):
        if not pos_tup or sym == params.regime_symbol:
            continue
        if sym not in rankings:
            continue

        df = state.candles.get((sym, interval))
        if df is None:
            continue
        closes = cast(pd.Series, df["close"])
        lows = cast(pd.Series, df["low"])
        close = float(closes.iloc[-1])

        rank = _rank_of(sym, rankings)
        recovered = rank <= params.exit_rank_threshold

        for position in pos_tup:
            pid = position.position_id

            trail = GLOBAL["trails"].get(pid)

            if trail is None:
                # Loser phase — hold while rank is unfavorable.
                if not recovered:
                    continue
                # Recovery confirmed — arm the trail at the recent low.
                GLOBAL["trails"][pid] = float(lows.iloc[-params.trail_lookback :].min())
                continue

            # Trail mode: ratchet the low up, exit on a break.
            recent_low = float(lows.iloc[-params.trail_lookback :].min())
            if recent_low > trail:
                GLOBAL["trails"][pid] = recent_low
                continue
            if close <= trail:
                GLOBAL["trails"].pop(pid, None)
                start_cooldown(sym, params.cooldown_bars)
                signals.append(
                    TradeSignal(
                        action=ActionType.close,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=close,
                        qty=abs(position.qty),
                        position_id=pid,
                        reason=f"[trail] break {trail:.2f} rank {rank} <= {params.exit_rank_threshold}",
                    )
                )

    # ---- Entry: long only in a bull regime ----
    if not bull:
        return signals

    # Count live (non-exiting-this-bar) positions.
    exiting_syms = {s.symbol for s in signals}
    effective = sum(
        len(t)
        for s, t in state.portfolio.positions.items()
        if s not in exiting_syms and s != params.regime_symbol
    )

    sorted_worst_first = sorted(rankings, key=lambda s: rankings[s])
    for sym in sorted_worst_first:
        pending = len([s for s in signals if s.action == ActionType.long])
        if effective + pending >= params.max_positions:
            break

        if sym in state.portfolio.positions and sym not in exiting_syms:
            continue
        if any(s.symbol == sym for s in signals):
            continue

        rank = _rank_of(sym, rankings)
        n_total = len(rankings)
        if rank <= n_total - params.top_n:
            continue  # not beaten-down enough

        cd = GLOBAL["cooldown"].get(sym, 0)
        if cd > 0:
            GLOBAL["cooldown"][sym] = cd - 1
            continue

        df = state.candles.get((sym, interval))
        if df is None:
            continue
        closes = cast(pd.Series, df["close"])
        if len(closes) < params.warmup_bars:
            continue

        entry_price = float(closes.iloc[-1])
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        highs = cast(pd.Series, df["high"])
        lows = cast(pd.Series, df["low"])
        atr_val = float(ta.atr(highs, lows, closes, params.atr_period).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            continue  # ATR sizing only — no silent fallback
        qty = (state.portfolio.cash * params.risk_pct) / (atr_val * params.atr_mult)

        GLOBAL["cooldown"][sym] = params.cooldown_bars
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=candle.timestamp,
                price=entry_price,
                qty=round(qty, 4),
                reason=f"[tmr] rank {rank}/{len(rankings)} ret={rankings[sym]:.2%} ATR {atr_val:.2f}",
            )
        )

    return signals
