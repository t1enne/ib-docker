"""Shannon's Demon — volatility-harvesting via periodic rebalancing.

Classic two-asset rebalancing from Shannon's MIT lectures (c. 1960).
Hold equal positions in two volatile assets. Periodically rebalance to
target weights. The "demon" harvests mean-reversion gains from the
noise-rich, trend-poor random walk.

Theory: for two uncorrelated assets with equal expected returns and
nonzero volatility, rebalancing creates a diversification return
(geometric mean > arithmetic mean — variance drag reduction).

Classic pair: SPY (risky) + TLT (bonds/cash proxy). Can use any volatile
pair — the key is volatility, not direction.

Multi-timeframe: when config.bars has multiple intervals (e.g. ["1h", "1d"]),
the strategy uses the longest interval for signal/weight decisions and the
candle's interval for execution pricing.

References:
  - Poundstone, "Fortune's Formula" (2005), Ch. 12
  - Luenberger, "Investment Science" (1998), §15.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "shannons_demon"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Rebalance frequency: "daily", "weekly" (5 trading days), "monthly" (21 days)
    rebalance_frequency: str = "monthly"
    rebalance_days: int = 21

    # Target weights — must sum to 1.0. [0.5, 0.5] = equal weight.
    target_weights: tuple[float, ...] = (0.5, 0.5)

    # Tolerance — don't rebalance if drift < this fraction of portfolio
    drift_tolerance: float = 0.05

    # Warmup
    warmup_bars: int = 21

    # Use cash as second leg when True; else use symbols[0], symbols[1]
    cash_leg: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        d = dict(d)
        if "target_weights" in d and isinstance(d["target_weights"], list):
            d["target_weights"] = tuple(d["target_weights"])
        return super().from_dict(d)


# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------

_last_rebalance: dict[str, int] = {}  # cache_key → signal bar index
_bar_idx: int = 0  # count of signal-interval bars seen
_last_signal_close: dict[str, float] = {}  # symbol → last seen signal close


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _interval_bars(params: Params) -> int:
    if params.rebalance_days > 0:
        return params.rebalance_days
    mapping = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}
    return mapping.get(params.rebalance_frequency, 21)


def _read_closes(
    state: BacktestState,
    symbols: list[str],
    interval: str,
) -> dict[str, float]:
    """Return {symbol: latest close} from given interval's candles."""
    closes: dict[str, float] = {}
    for sym in symbols:
        df = state.candles.get((sym, interval))
        if df is not None and len(df) > 0:
            closes[sym] = float(cast(pd.Series, df["close"]).iloc[-1])
    return closes


def _portfolio_closes(
    state: BacktestState,
    symbols: list[str],
    interval: str,
) -> tuple[dict[str, float], float, float]:
    """Return (closes dict, pos_value, total_portfolio_value)."""
    closes: dict[str, float] = {}
    for sym in symbols:
        df = state.candles.get((sym, interval))
        if df is not None and len(df) > 0:
            closes[sym] = float(cast(pd.Series, df["close"]).iloc[-1])

    pos_value = sum(
        abs(p.qty) * closes.get(sym, 0.0)
        for sym, pos_tup in state.portfolio.positions.items()
        for p in pos_tup
    )
    cash = state.portfolio.cash
    total = pos_value + cash
    return closes, pos_value, total


def _current_weights(
    state: BacktestState,
    symbols: list[str],
    cash_leg: bool,
    interval: str,
) -> dict[str, float]:
    """Return {symbol: weight} for each leg and optionally cash."""
    closes, pos_value, total = _portfolio_closes(state, symbols, interval)
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for sym, pos_tup in state.portfolio.positions.items():
        if sym in closes and pos_tup:
            total_qty = sum(abs(p.qty) for p in pos_tup)
            weights[sym] = (total_qty * closes[sym]) / total
    if cash_leg:
        weights["__cash__"] = state.portfolio.cash / total
    return weights


def _drift_exceeds(
    current: dict[str, float],
    target: dict[str, float],
    tolerance: float,
) -> bool:
    for key, tw in target.items():
        cw = current.get(key, 0.0)
        if abs(cw - tw) > tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def _pick_intervals(state: BacktestState) -> tuple[str, str]:
    """Return (signal_interval, entry_interval).

    Signal = longest available interval (e.g. "1d").
    Entry  = shortest available interval (e.g. "1h").

    Single-interval backtests: both are the same.
    """
    intervals = sorted(
        {i for _, i in state.candles},
        key=lambda x: (0 if x.endswith("d") else 1 if x.endswith("h") else 2, x),
    )
    if len(intervals) <= 1:
        iv = intervals[0] if intervals else "1d"
        return iv, iv
    # Sorted: daily first, then hourly — so [0] is signal (longest), [-1] is entry (shortest)
    return intervals[0], intervals[-1]


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    global _bar_idx, _last_signal_close

    signal_interval, entry_interval = _pick_intervals(state)
    symbols = sorted({s for s, _ in state.candles})

    # Determine legs early — needed for signal-close tracking
    if params.cash_leg:
        risk_symbols = [symbols[0]] if symbols else []
    else:
        if len(symbols) < 2:
            return []
        risk_symbols = symbols[:2]

    # New signal bar detection: only count when signal-interval close changes
    sig_closes = _read_closes(state, risk_symbols, signal_interval)
    if not sig_closes:
        return []

    new_signal_bar = not _last_signal_close or any(
        sig_closes.get(sym) != _last_signal_close.get(sym) for sym in risk_symbols
    )
    _last_signal_close = dict(sig_closes)

    if new_signal_bar:
        _bar_idx += 1

    # Warmup on signal interval
    sig_df = state.candles.get((risk_symbols[0], signal_interval))
    if sig_df is None or len(sig_df) < params.warmup_bars:
        return []

    # Only act on new signal bars (not every entry bar)
    if not new_signal_bar:
        return []

    all_legs = risk_symbols + (["__cash__"] if params.cash_leg else [])

    # Build target weight dict
    tw = params.target_weights
    if len(tw) != len(all_legs):
        tw = (0.5, 0.5) if len(all_legs) == 2 else (1.0,)
    target = dict(zip(all_legs, tw))

    # --- Rebalance gate ---
    cache_key = "shannons"
    interval_bars = _interval_bars(params)

    bars_since = _bar_idx - _last_rebalance.get(cache_key, -interval_bars)
    if bars_since < interval_bars:
        return []

    _last_rebalance[cache_key] = _bar_idx

    # Signal decisions: use signal-interval closes (e.g. daily)
    signal_closes, _, total = _portfolio_closes(state, risk_symbols, signal_interval)
    if total <= 0:
        return []

    current_weights = _current_weights(
        state, risk_symbols, params.cash_leg, signal_interval
    )

    # Entry/exit prices: use entry-interval closes (e.g. hourly)
    entry_closes, _, _ = _portfolio_closes(state, risk_symbols, entry_interval)

    # First deployment: buy both legs at target weights
    if not current_weights:
        return _deploy_initial(
            candle, risk_symbols, all_legs, target, entry_closes, total
        )

    # Drift gate
    if not _drift_exceeds(current_weights, target, params.drift_tolerance):
        return []

    # Rebalance: close all positions, then reopen at target weights
    return _full_rebalance(
        state,
        candle,
        risk_symbols,
        all_legs,
        target,
        entry_closes,
        total,
        current_weights,
    )


def _deploy_initial(
    candle: Candle,
    risk_symbols: list[str],
    all_legs: list[str],
    target: dict[str, float],
    entry_closes: dict[str, float],
    total: float,
) -> list[TradeSignal]:
    """First deployment: buy at entry-interval prices."""
    signals: list[TradeSignal] = []
    for sym in risk_symbols:
        price = entry_closes.get(sym) or candle.close
        if price <= 0:
            continue
        tw_sym = target.get(sym, 0.5)
        target_value = total * tw_sym
        qty = target_value / price
        if qty < 1e-8:
            continue
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=candle.timestamp,
                price=price,
                qty=qty,
                reason=f"[shannon] deploy {tw_sym:.0%} @ {price:.2f}",
            )
        )
    return signals


def _full_rebalance(
    state: BacktestState,
    candle: Candle,
    risk_symbols: list[str],
    all_legs: list[str],
    target: dict[str, float],
    entry_closes: dict[str, float],
    total: float,
    current_weights: dict[str, float],
) -> list[TradeSignal]:
    """Emit net rebalance signals — close extras, then delta-adjust.

    For each symbol:
      1. Close any positions beyond the first (consolidation).
      2. Send one rebalance signal on the remaining position with net delta.

    Uses entry-interval prices (e.g. 1h) for execution.
    """
    signals: list[TradeSignal] = []

    for sym in risk_symbols:
        price = entry_closes.get(sym) or candle.close
        if price <= 0:
            continue

        pos_tup = state.portfolio.positions.get(sym, ())
        total_qty = sum(p.qty for p in pos_tup) if pos_tup else 0.0

        tw_sym = target.get(sym, 0.5)
        target_value = total * tw_sym
        target_qty = target_value / price

        delta = round(target_qty - total_qty, 4)

        # When multiple positions exist (shouldn't happen normally),
        # close all and reopen at target. Otherwise emit single rebalance.
        if len(pos_tup) > 1:
            for pos in pos_tup:
                signals.append(
                    TradeSignal(
                        action=ActionType.close,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=price,
                        qty=abs(pos.qty),
                        position_id=pos.position_id,
                        reason=f"[shannon] consolidate close @ {price:.2f}",
                    )
                )
            if target_qty > 0:
                cw = current_weights.get(sym, 0)
                signals.append(
                    TradeSignal(
                        action=ActionType.long,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=price,
                        qty=target_qty,
                        reason=(
                            f"[shannon] consolidate open {target_qty:.4f} "
                            f"({cw:.1%}->{tw_sym:.0%}) @ {price:.2f}"
                        ),
                    )
                )
            continue

        if abs(delta) < 1e-8:
            continue

        pid: str | None = pos_tup[0].position_id if pos_tup else None
        cw = current_weights.get(sym, 0)
        direction = "buy" if delta > 0 else "sell"
        signals.append(
            TradeSignal(
                action=ActionType.rebalance,
                symbol=sym,
                timestamp=candle.timestamp,
                price=price,
                qty=delta,
                position_id=pid,
                reason=(
                    f"[shannon] rebalance {direction} {abs(delta):.4f} "
                    f"({cw:.1%}→{tw_sym:.0%}) @ {price:.2f}"
                ),
            )
        )

    return signals
