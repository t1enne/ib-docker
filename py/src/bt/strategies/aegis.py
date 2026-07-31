"""AEGIS — Adaptive Equity Generation and Immunisation System.

Paper: "Taming the Black Swan: A Momentum-Gated Hierarchical Optimisation
Framework for Asymmetric Alpha Generation" (Chakraborty & Singh, 2025).

Three-stage pipeline:
  1. Signal Generation  — VAM (12mo skip-month) → per-GICS-sector leader → top-3 = Anchor Triad
  2. Immunisation Layer  — Minimax correlation on remaining N-3 slots (47 diversifiers)
  3. Allocation Engine    — SLSQP convex optimisation maximising Sortino ratio (5% cap, long-only)

Rebalances monthly. No regime gating — structural diversification handles drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "aegis"

# ---------------------------------------------------------------------------
# GICS sector map — ETF proxy → sector label
# Covers the 11 GICS sectors from the paper's 5-indice universe.
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    "XLK": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "XLF": "Financials",
    "JPM": "Financials",
    "BAC": "Financials",
    "XLV": "Healthcare",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "XLY": "Consumer Discretionary",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "XLE": "Energy",
    "XOM": "Energy",
    "CVX": "Energy",
    "XLI": "Industrials",
    "CAT": "Industrials",
    "HON": "Industrials",
    "XLB": "Materials",
    "LIN": "Materials",
    "XLU": "Utilities",
    "NEE": "Utilities",
    "SO": "Utilities",
    "XRT": "Consumer Discretionary",
    "XBI": "Healthcare",
    "SPY": "Large-Cap Blend",
    "QQQ": "Technology",
    "DIA": "Industrials",
    "MDY": "Mid-Cap Blend",
    "IJR": "Small-Cap Blend",
}

_UNMAPPED_SECTOR = "Unknown"


def _sector(sym: str) -> str:
    return SECTOR_MAP.get(sym, _UNMAPPED_SECTOR)


# ---------------------------------------------------------------------------
# Typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Momentum filter (paper: 12-month skip-month)
    mom_lookback_days: int = 252  # 12 months
    skip_days: int = 21  # skip most recent month
    vol_window_days: int = 252  # annualised vol over lookback

    # Correlation filter
    corr_lookback_days: int = 252
    max_corr: float = 0.70  # max pairwise in minimax

    # Basket sizing (paper: 3 anchors + 47 diversifiers = 50)
    n_anchors: int = 3
    n_diversifiers: int = 47

    # SLSQP allocation (paper: 3-month covariance window)
    alloc_lookback_days: int = 63  # ~3 months
    max_position_weight: float = 0.05  # 5% cap
    risk_free_rate: float = 0.04  # 4% annualised (paper's Rf)

    # Friction
    friction_bps: float = 10.0  # 10 basis points

    # Warmup
    warmup_days: int = 504  # 2 years of data

    # Schedule
    rebalance_monthly: bool = True


# ---------------------------------------------------------------------------
# Pure computation functions
# ---------------------------------------------------------------------------


def _log_returns(closes: pd.Series) -> pd.Series:
    """Daily log-returns (Equation 8)."""
    return np.log(closes / closes.shift(1))


def _vam(closes: pd.Series, mom_lb: int, skip: int, vol_w: int) -> float:
    """Volatility-Adjusted Momentum (Equation 11).

    VAM = cumulative log-return (skip-month) / annualised volatility.
    Returns -inf if insufficient data.
    """
    if len(closes) < mom_lb + 2:
        return -np.inf

    log_r = _log_returns(closes)

    # Cum return over [t-mom_lb, t-skip] (Equation 9)
    window = log_r.iloc[-(mom_lb):-(skip)] if skip > 0 else log_r.iloc[-mom_lb:]
    cum_ret = float(window.sum())

    # Annualised volatility (Equation 10, but over return window)
    full_window = log_r.iloc[-max(mom_lb, vol_w) :]
    ann_vol = float(full_window.std()) * np.sqrt(252)

    if pd.isna(ann_vol) or ann_vol == 0.0:
        return -np.inf

    return cum_ret / ann_vol


def _select_anchors(
    symbols: list[str],
    closes_map: dict[str, pd.Series],
    mom_lb: int,
    skip: int,
    vol_w: int,
    n_anchors: int,
) -> list[str]:
    """Hierarchical filtration (Algorithm 1).

    1. Group universe by GICS sector.
    2. Per sector: pick asset with highest raw cumulative return → sector leader.
    3. Compute VAM for each sector leader.
    4. Top-N VAM leaders = Anchor Triad.
    """
    # Compute raw cumulative return (skip-month) for every symbol
    raw_ret: dict[str, float] = {}
    for sym in symbols:
        c = closes_map.get(sym)
        if c is None or len(c) < mom_lb + 2:
            continue
        log_r = _log_returns(c)
        window = log_r.iloc[-(mom_lb):-(skip)] if skip > 0 else log_r.iloc[-mom_lb:]
        raw_ret[sym] = float(window.sum())

    # Group by sector
    by_sector: dict[str, list[str]] = {}
    for sym in raw_ret:
        by_sector.setdefault(_sector(sym), []).append(sym)

    # Sector leader = highest raw return per sector
    sector_leaders: list[str] = []
    for sec, members in by_sector.items():
        leader = max(members, key=lambda s: raw_ret[s])
        sector_leaders.append(leader)

    # VAM score each leader
    leader_scores: dict[str, float] = {}
    for sym in sector_leaders:
        c = closes_map[sym]
        leader_scores[sym] = _vam(c, mom_lb, skip, vol_w)

    # Top N by VAM
    sorted_leaders = sorted(leader_scores, key=lambda s: leader_scores[s], reverse=True)
    return sorted_leaders[:n_anchors]


def _momentum_gate(
    symbols: list[str],
    closes_map: dict[str, pd.Series],
    mom_lb: int,
    skip: int,
) -> list[str]:
    """Filter to assets with positive cumulative return (Ri > 0)."""
    qualifying: list[str] = []
    for sym in symbols:
        c = closes_map.get(sym)
        if c is None or len(c) < mom_lb + 2:
            continue
        log_r = _log_returns(c)
        window = log_r.iloc[-(mom_lb):-(skip)] if skip > 0 else log_r.iloc[-mom_lb:]
        if float(window.sum()) > 0:
            qualifying.append(sym)
    return qualifying


def _minimax_diversifiers(
    anchors: list[str],
    candidates: list[str],
    corr: pd.DataFrame,
    n_slots: int,
) -> list[str]:
    """Immunisation Layer (Algorithm 2).

    Start with anchors locked in. Greedily add diversifiers that minimise
    the maximum pairwise correlation against the current basket.

    Args:
        anchors: Already-selected anchor symbols.
        candidates: Eligible pool (momentum-gated, excluding anchors).
        corr: Pairwise correlation DataFrame (all symbols).
        n_slots: Number of additional slots to fill.
    """
    basket: list[str] = list(anchors)
    remaining: list[str] = [c for c in candidates if c in corr.columns]
    # Remove anchors from candidate pool (anchors may be in corr.columns)
    remaining = [c for c in remaining if c not in anchors]

    for _ in range(n_slots):
        if not remaining:
            break
        best = None
        best_max_corr = 2.0  # > max possible |ρ|
        for c in remaining:
            mx = max(
                abs(corr.loc[c, b])
                for b in basket
                if b in corr.columns and c in corr.index
            )
            if mx < best_max_corr:
                best_max_corr = mx
                best = c
        if best is None:
            break
        basket.append(best)
        remaining.remove(best)

    return basket[len(anchors) :]  # return only new diversifiers


def _slsqp_optimise(
    symbols: list[str],
    returns_df: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
) -> dict[str, float]:
    """SLSQP convex optimisation maximising Sortino ratio (Algorithm 3).

    maximise  F(w) = (wᵀμ - Rf) / DD(w)     [Sortino]
    s.t.      Σw = 1,  0 ≤ wᵢ ≤ max_weight

    scipy's SLSQP handles the non-linear objective with linear constraints.
    """
    n = len(symbols)
    if n == 0:
        return {}
    if n == 1:
        return {symbols[0]: 1.0}

    daily_rf = risk_free_rate / 252

    def objective(w: np.ndarray) -> float:
        """Negative Sortino ratio (minimised)."""
        daily_port = returns_df.to_numpy(dtype=np.float64) @ w
        ann_ret = float(daily_port.mean()) * 252
        # Downside deviation (LPM degree 2, Equation 14)
        downside = np.minimum(daily_port - daily_rf, 0.0)
        lpm2 = float((downside**2).mean()) * 252
        dd = np.sqrt(lpm2) if lpm2 > 0 else 1e-12
        sortino = (ann_ret - risk_free_rate) / dd
        return -sortino

    # Initial guess: equal weight
    x0 = np.full(n, 1.0 / n, dtype=np.float64)

    # Constraints: Σw = 1
    constraints = LinearConstraint(np.ones(n, dtype=np.float64), 1.0, 1.0)

    # Bounds: 0 ≤ wᵢ ≤ max_weight
    bounds = Bounds(
        np.zeros(n, dtype=np.float64), np.full(n, max_weight, dtype=np.float64)
    )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        # Fallback: equal weight, capped at max_weight
        w_eq = min(1.0 / n, max_weight)
        raw = {s: w_eq for s in symbols}
        total = sum(raw.values())
        if total > 0:
            raw = {s: w / total for s, w in raw.items()}
        return raw

    w_opt = result.x
    w_opt = np.maximum(w_opt, 0.0)
    total = w_opt.sum()
    if total > 0:
        w_opt /= total

    return dict(zip(symbols, w_opt.tolist()))


# ---------------------------------------------------------------------------
# Rebalance gate — fires on first trading day of each calendar month.
# Detects month transitions via module-level memory of last-seen month.
# ---------------------------------------------------------------------------

_last: dict[str, pd.Timestamp] = {}
_last_month: dict[str, tuple[int, int]] = {}  # (year, month)


def _is_monthly_signal(ts: pd.Timestamp) -> bool:
    """True on the first trading day of a new calendar month."""
    cache_key = "aegis"
    current = (ts.year, ts.month)
    prev = _last_month.get(cache_key)
    _last_month[cache_key] = current
    return prev is not None and current != prev


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    # ---- Warmup ----
    first = next(iter(state.candles.values()), None)
    if first is None or len(first) < params.warmup_days:
        return _close_all(state, candle, "warmup")

    # ---- Rebalance gate: monthly ----
    cache_key = "aegis"
    ts = candle.timestamp
    if not _is_monthly_signal(ts):
        return []
    if _last.get(cache_key) is not None and _last[cache_key] == ts:  # type: ignore[union-attr]
        return []
    _last[cache_key] = ts

    interval = candle.interval or "1h"
    symbols = sorted({s for s, _ in state.candles})

    # ---- Build closes and returns maps ----
    closes_map: dict[str, pd.Series] = {}
    returns_map: dict[str, pd.Series] = {}
    for sym in symbols:
        df = state.candles.get((sym, interval))
        if df is None or len(df) < params.mom_lookback_days + 2:
            continue
        closes_map[sym] = cast(pd.Series, df["close"])
        returns_map[sym] = _log_returns(closes_map[sym]).dropna()

    if len(closes_map) < max(params.n_anchors, 3):
        return _close_all(state, candle, "insufficient data")

    # ---- Stage 1: Anchor Triad (Algorithm 1) ----
    anchors = _select_anchors(
        symbols=list(closes_map.keys()),
        closes_map=closes_map,
        mom_lb=params.mom_lookback_days,
        skip=params.skip_days,
        vol_w=params.vol_window_days,
        n_anchors=params.n_anchors,
    )
    if len(anchors) < params.n_anchors:
        return _close_all(state, candle, "too few anchors")

    # ---- Stage 2: Momentum gate (Ri > 0) ----
    candidates = _momentum_gate(
        symbols=list(closes_map.keys()),
        closes_map=closes_map,
        mom_lb=params.mom_lookback_days,
        skip=params.skip_days,
    )
    # Remove anchors from candidate pool
    anchor_set = set(anchors)
    candidates = [c for c in candidates if c not in anchor_set]

    # ---- Stage 3: Build correlation matrix ----
    corr_symbols = anchors + [c for c in candidates if c in returns_map]
    if len(corr_symbols) < 2:
        # Only anchors, allocate equally
        selected = list(anchors)
        weights = {s: 1.0 / len(selected) for s in selected}
    else:
        rets = {s: returns_map[s] for s in corr_symbols if s in returns_map}
        if len(rets) < 2:
            selected = list(anchors)
            weights = {s: 1.0 / len(selected) for s in selected}
        else:
            corr_df = pd.DataFrame(rets).iloc[-params.corr_lookback_days :].corr()

            # ---- Stage 4: Minimax diversifiers (Algorithm 2) ----
            diversifiers = _minimax_diversifiers(
                anchors=anchors,
                candidates=candidates,
                corr=corr_df,
                n_slots=params.n_diversifiers,
            )
            selected = anchors + diversifiers

            # ---- Stage 5: SLSQP Sortino allocation (Algorithm 3) ----
            alloc_ret = {
                s: returns_map[s].iloc[-params.alloc_lookback_days :]
                for s in selected
                if s in returns_map
                and len(returns_map[s]) >= params.alloc_lookback_days
            }
            if len(alloc_ret) < 2:
                weights = {s: 1.0 / len(selected) for s in selected}
            else:
                alloc_df = pd.DataFrame(alloc_ret)
                weights = _slsqp_optimise(
                    symbols=list(alloc_df.columns),
                    returns_df=alloc_df,
                    risk_free_rate=params.risk_free_rate,
                    max_weight=params.max_position_weight,
                )
                # Ensure all selected are in weights (fallback to 0)
                for s in selected:
                    weights.setdefault(s, 0.0)
                # Re-normalise
                total = sum(weights.values())
                if total > 0:
                    weights = {s: w / total for s, w in weights.items()}

    # ---- Stage 6: Generate signals with friction ----
    held = set(state.portfolio.positions.keys())
    target = set(selected)

    signals: list[TradeSignal] = []

    # Close deselected — close all positions for each deselected symbol
    for sym in held - target:
        sym_positions = state.portfolio.positions.get(sym, ())
        for pos in sym_positions:
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=ts,
                    price=candle.close,
                    qty=abs(pos.qty),
                    position_id=pos.position_id,
                    reason="deselected",
                )
            )

    # Compute turnover cost
    # Friction = 10bp × Σ |Δw| applied as cash deduction, not per-signal
    # The paper deducts from net return; we estimate turnover cost upfront
    # by computing Δw and deducting from available cash.
    available_cash = state.portfolio.cash
    if available_cash <= 1.0:
        return signals

    # Estimate turnover: sum of absolute weight changes.
    # Use per-position last_price (updated by mark-to-market) — NOT
    # candle.close, which is the trigger candle's close and may be NaN
    # for symbols with sparse/incomplete data (e.g. PYPL).
    curr_weights: dict[str, float] = {}
    total_pos_value = sum(
        abs(p.qty) * p.last_price
        for pos_tup in state.portfolio.positions.values()
        for p in pos_tup
    )
    portfolio_value = available_cash + total_pos_value
    for sym, pos_tup in state.portfolio.positions.items():
        if portfolio_value > 0 and pos_tup:
            total_qty = sum(abs(p.qty) for p in pos_tup)
            # All positions for the same symbol share the same last_price
            curr_weights[sym] = (total_qty * pos_tup[0].last_price) / portfolio_value

    turnover = sum(
        abs(weights.get(s, 0.0) - curr_weights.get(s, 0.0)) for s in target | held
    )
    friction_cost = turnover * (params.friction_bps / 10_000.0) * portfolio_value
    tradable_cash = max(available_cash - friction_cost, 0.0)

    new_symbols = target - held
    if not new_symbols:
        return signals

    # Re-normalize weights for new symbols
    new_weights = {s: weights[s] for s in new_symbols}
    total = sum(new_weights.values())
    if total > 0:
        for s in new_weights:
            new_weights[s] /= total

    for sym in new_symbols:
        w = new_weights.get(sym, 0.0)
        if w <= 1e-8:
            continue
        # Use per-symbol close price from state, not the trigger candle's close
        sym_df = state.candles.get((sym, interval))
        if sym_df is None or len(sym_df) == 0:
            continue
        sym_close = float(sym_df["close"].iloc[-1])
        if sym_close <= 0 or np.isnan(sym_close):
            continue
        qty = (tradable_cash * w) / sym_close
        if qty <= 1e-8:
            continue
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=ts,
                price=sym_close,
                qty=qty,
                reason=f"alloc {w:.2%} (AEGIS)",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# Close-all helper
# ---------------------------------------------------------------------------


def _close_all(state: BacktestState, candle: Candle, reason: str) -> list[TradeSignal]:
    signals: list[TradeSignal] = []
    for sym, pos_tup in state.portfolio.positions.items():
        for pos in pos_tup:
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=candle.close,
                    qty=abs(pos.qty),
                    position_id=pos.position_id,
                    reason=reason,
                )
            )
    return signals
