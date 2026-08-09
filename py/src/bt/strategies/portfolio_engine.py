"""Shared rebalancing engine for correlation-driven portfolio strategies.

Implements the same "correlation determines allocation" scheme the removed
``bt pf`` command used, but as a live, `bt run`-able strategy. At each
rebalance boundary it derives target weights for a whole universe from the
trailing-return covariance matrix (GMV / inverse-vol / risk-parity) and emits
net ``rebalance`` signals that move every held position toward those targets.

It is deliberately **not** a registered strategy: it defines no
``STRATEGY_TYPE``/``on_candle`` and is imported by the thin per-method strategy
modules (``pf_gmv``, ``pf_invvol``, ``pf_risk_parity``, ``pf_alloc``). All
functions are pure over ``CandleStore`` inputs except ``pf_on_candle``, which
mutates the calling module's ``GLOBAL`` dict (the repo state-convention).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.portfolio_weights import (
    WeightMethodFn,
    cap_weights,
    resolve_rebalance_period,
)


def _cadence_offset(rebalance: str) -> pd.tseries.offsets.BaseOffset:
    """Build the calendar offset for a rebalance cadence.

    ``resolve_rebalance_period`` turns named/``Nd | Nw | Nm`` shorthands into a
    pandas offset alias (e.g. ``ME``, ``W-MON``, ``1D``); this maps that alias
    Concrete ``BaseOffset`` used for calendar boundary computation (P1-4).
    """
    return to_offset(resolve_rebalance_period(rebalance))


# ---------------------------------------------------------------------------
# weight computation over the CandleStore
# ---------------------------------------------------------------------------


def build_returns_window(
    state: BacktestState,
    symbols: list[str],
    interval: str,
    lookback: int,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    """Build a trailing-return matrix and current closes for the universe.

    Uses only rows up to the strategy cursor (no lookahead). Returns
    ``(returns_wide, latest_closes, min_bar_count)`` where ``returns_wide`` has
    ``pd.Timestamp`` index x symbol columns of periodic returns, ``latest_closes``
    is ``{symbol: close}`` for the most recent bar, and ``min_bar_count`` is the
    fewest bars any symbol has (used for warmup gating).
    """
    closes_frames: list[pd.Series] = []
    latest: dict[str, float] = {}
    min_count = 2**31
    for sym in symbols:
        df = state.candles.get((sym, interval))
        if df is None or df.empty:
            return pd.DataFrame(), {}, 0
        close = cast(pd.Series, df["close"])
        min_count = min(min_count, int(close.notna().sum()))
        # Trailing window: last `lookback` rows only.
        window = close.dropna().tail(lookback)
        if len(window) < 2:
            return pd.DataFrame(), {}, 0
        closes_frames.append(window.astype(float).rename(sym))
        latest[sym] = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else 0.0

    aligned = pd.concat(closes_frames, axis=1, sort=True)
    aligned = aligned[~aligned.index.duplicated(keep="first")].sort_index()
    returns = aligned.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns, latest, min_count


def target_weights(
    state: BacktestState,
    symbols: list[str],
    interval: str,
    lookback: int,
    weight_fn: WeightMethodFn,
    max_weight: float = 1.0,
) -> pd.Series:
    """Derive target portfolio weights from the trailing correlation structure."""
    if len(symbols) < 2:
        raise ValueError("correlation-driven strategies need >= 2 symbols")

    returns, _, _ = build_returns_window(state, symbols, interval, lookback)
    if returns.empty:
        return pd.Series(0.0, index=symbols)

    weights = weight_fn(returns, long_only=True)
    out = pd.Series(0.0, index=symbols)
    for sym in symbols:
        if sym in returns.columns:
            out.loc[sym] = float(weights.loc[sym])

    if max_weight < 1.0:
        out = cap_weights(out, max_weight)
    return out


# ---------------------------------------------------------------------------
# current portfolio weights
# ---------------------------------------------------------------------------


def current_weights(
    state: BacktestState,
    symbols: list[str],
    closes: dict[str, float],
) -> dict[str, float]:
    """Current dollar weights of held positions, plus cash as implicit residual.

    Returns ``{symbol: weight}`` for held symbols only; unallocated cash is
    implicitly the residual (1 - sum), not listed as a key.
    """
    pos_value = portfolio_position_value(state, closes)
    total = pos_value + state.portfolio.cash
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for sym, pos_tup in state.portfolio.positions.items():
        price = closes.get(sym)
        if price is None or not pos_tup:
            continue
        total_qty = sum(abs(p.qty) for p in pos_tup)
        weights[sym] = (total_qty * price) / total
    return weights


def _one_way_turnover(
    current: dict[str, float],
    targets: pd.Series,
    symbols: list[str],
) -> float:
    """One-way gross turnover = sum of |target - current| weight deltas.

    Uses the full universe (current weight defaults to 0 for unheld symbols).
    ``targets`` is a Series indexed by symbol (already summed via target_weights).
    """
    turnover = 0.0
    for sym in symbols:
        tw = float(targets.get(sym, 0.0))
        cw = current.get(sym, 0.0)
        turnover += abs(tw - cw)
    return turnover


# ---------------------------------------------------------------------------
# signal emission
# ---------------------------------------------------------------------------


def emit_rebalance_signals(
    state: BacktestState,
    candle: Candle,
    symbols: list[str],
    targets: pd.Series,
    closes: dict[str, float],
    plan: str,
) -> list[TradeSignal]:
    """Emit net-``rebalance`` signals moving positions toward ``targets``.

    For every symbol with an existing position we compute the delta between the
    target share count (``target_weight * portfolio_value / price``) and the
    current share count, and emit a single ``rebalance`` signal with that delta
    (positive = buy, negative = sell). Symbols without a position and a target
    weight > 0 are opened with an explicit ``long`` signal sized to the target.
    Closing a symbol whose target is 0 is handled by the rebalance delta making
    the resulting position <= 0.
    """
    signals: list[TradeSignal] = []
    total = state.portfolio.cash + portfolio_position_value(state, closes)
    if total <= 0:
        return signals
    for sym in symbols:
        price = closes.get(sym)
        if price is None or price <= 0:
            continue
        tw = float(targets.get(sym, 0.0))
        pos_tup = state.portfolio.positions.get(sym, ())
        current_qty = sum(p.qty for p in pos_tup) if pos_tup else 0.0
        target_qty = total * tw / price

        if not pos_tup:
            # Fresh entry.
            if target_qty > 1e-8:
                signals.append(
                    TradeSignal(
                        action=ActionType.long,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=price,
                        qty=round(target_qty, 4),
                        reason=f"[pf] init {tw:.1%} [{plan}]",
                    )
                )
            continue

        pid = pos_tup[0].position_id
        delta = round(target_qty - current_qty, 4)
        if abs(delta) < 1e-8:
            continue
        # Extra positions beyond the first are consolidated away first.
        for extra in pos_tup[1:]:
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=price,
                    qty=abs(extra.qty),
                    position_id=extra.position_id,
                    reason=f"[pf] consolidate @ {price:.2f}",
                )
            )
        signals.append(
            TradeSignal(
                action=ActionType.rebalance,
                symbol=sym,
                timestamp=candle.timestamp,
                price=price,
                qty=delta,
                position_id=pid,
                reason=f"[pf] {('buy' if delta > 0 else 'sell')} {abs(delta):.4f} "
                f"to {tw:.1%} [{plan}]",
            )
        )
    return signals


def read_closes(
    state: BacktestState,
    symbols: list[str],
    interval: str,
) -> dict[str, float]:
    """Return ``{symbol: latest close}`` for the universe at the current cursor."""
    out: dict[str, float] = {}
    for sym in symbols:
        latest = state.candles.latest(sym, interval)
        if latest is not None:
            out[sym] = float(latest)
    return out


def portfolio_position_value(
    state: BacktestState,
    closes: dict[str, float],
) -> float:
    """Dollar value of all open positions at the current closes."""
    return sum(
        abs(p.qty) * closes.get(sym, 0.0)
        for sym, pos_tup in state.portfolio.positions.items()
        for p in pos_tup
    )


# GLOBAL keys written by ``pf_on_candle`` are seeded via ``_init_globals``;
# each pf_* module also declares them in its own ``GLOBAL``/``reset_global``.


def _init_globals(g: dict) -> None:
    """Seed any missing ``pf_on_candle`` GLOBAL keys with defaults."""
    defaults = {
        "bar_idx": 0,
        "last_signal_close": {},
        "next_rebalance": None,
        "n_rebalances": 0,
        "gross_turnover": 0.0,
        "last_plan": "init",
    }
    for k, v in defaults.items():
        g.setdefault(k, v)


def pf_on_candle(
    state: BacktestState,
    candle: Candle,
    g: dict,
    weight_fn: WeightMethodFn,
    interval: str,
    rebalance: str,
    lookback: int,
    max_weight: float,
    warmup_bars: int,
) -> list[TradeSignal]:
    """Generic correlation-rebalance driver shared by the pf_* strategies.

    Semantics (per the faithful PRP spec):

    * **Unconditional cadence (P1-3):** rebalances run on every cadence
      boundary — no drift-tolerance skipping. Targets move every period anyway
      for the correlation methods, so this keeps turnover and allocation path
      comparable across methods.
    * **Calendar cadence (P1-4):** boundaries are calendar dates (month-end,
      week-start, …) derived from the signal bar's date, not a ``N``-bar
      counter, so exposure is not tied to a sliding bar count.
    * **Consistent warmup (P1-5):** the first rebalance fires at the first
      cadence boundary after ``max(warmup_bars, lookback)`` trailing bars are
      available (all four schemes share one rule and start together for a
      given warmup_bars).
    * **Go-to-cash (P2-7):** when targets sum to <= 0 (e.g. GMV's unallocated
      sentinel) and positions exist, positions are closed back to cash rather
      than silently held.

    ``g`` is the calling strategy module's ``GLOBAL`` dict; it is mutated in
    place to track bars, the next calendar boundary, rebalance count and gross
    turnover (the latter two feed the portfolio report). Returns the list of
    ``TradeSignal`` emitted this bar (may be empty).
    """
    _init_globals(g)

    symbols = sorted({s for s, _ in state.candles})
    if len(symbols) < 2:
        return []

    # The engine fires ``on_candle`` only on the last symbol in config order
    # (see AGENTS.md §9), so state.candles is complete for every symbol here.
    # Gate on the signal interval only — the symbol/order gate is the engine's.
    if candle.interval != interval:
        return []

    closes = read_closes(state, symbols, interval)
    latest = closes.get(candle.symbol)
    if latest is None:
        return []

    new_bar = not g["last_signal_close"] or (
        g["last_signal_close"].get(candle.symbol) != latest
    )
    if new_bar:
        g["bar_idx"] += 1
    g["last_signal_close"] = dict(closes)

    # Warmup: every symbol needs at least `warmup_bars` trailing rows before we
    # will even consider a boundary (P1-5). Correlation methods pass their real
    # lookback; fixed-alloc passes a small explicit value.
    _, _, min_count = build_returns_window(state, symbols, interval, lookback)
    if min_count < max(warmup_bars, 2):
        return []

    bar_date = candle.timestamp
    offset = _cadence_offset(rebalance)

    # First eligible bar (post-warmup) anchors the rebalance calendar: the
    # first boundary is the first cadence boundary at/after this bar's date.
    if g["next_rebalance"] is None:
        g["next_rebalance"] = offset.rollforward(bar_date)
        if g["next_rebalance"] > bar_date:
            return []
        # Boundary is this bar (date lands exactly on one): fall through and fire.

    if bar_date < g["next_rebalance"]:
        return []

    targets = target_weights(state, symbols, interval, lookback, weight_fn, max_weight)

    # P2-7: all-zero/unallocated targets -> go to cash (close held positions),
    # unless no positions are held (then nothing to do).
    if float(targets.sum()) <= 1e-12:
        if state.portfolio.positions:
            g["last_plan"] = "cash"
            signals = _go_to_cash(state, candle, symbols, closes)
            _advance_boundary(g, bar_date, offset)
            return signals
        g["next_rebalance"] = None  # no anchors yet; re-arm after warmup
        return []

    plan = "drift" if state.portfolio.positions else "init"
    g["last_plan"] = plan

    # Track gross one-way turnover for the portfolio report (P1-6). Cash is not
    # a rebalancable weight; turnover spans the traded symbols only.
    current = current_weights(state, symbols, closes)
    g["gross_turnover"] += _one_way_turnover(current, targets, symbols)
    g["n_rebalances"] += 1

    signals = emit_rebalance_signals(state, candle, symbols, targets, closes, plan)
    _advance_boundary(g, bar_date, offset)
    return signals


def _go_to_cash(
    state: BacktestState,
    candle: Candle,
    symbols: list[str],
    closes: dict[str, float],
) -> list[TradeSignal]:
    """Emit closes for every held position (P2-7 go-to-cash on zero targets)."""
    signals: list[TradeSignal] = []
    for sym in symbols:
        pos_tup = state.portfolio.positions.get(sym, ())
        for pos in pos_tup:
            price = closes.get(sym)
            if price is None or price <= 0:
                continue
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=price,
                    qty=abs(pos.qty),
                    position_id=pos.position_id,
                    reason="[pf] go-to-cash @ 0 target",
                )
            )
    return signals


def _advance_boundary(
    g: dict,
    bar_date: pd.Timestamp,
    offset: pd.tseries.offsets.BaseOffset,
) -> None:
    """Move ``g["next_rebalance"]`` to the first boundary strictly after the fired bar."""
    nb = g["next_rebalance"]
    while nb is not None and nb <= bar_date:
        nb = nb + offset
    g["next_rebalance"] = nb
