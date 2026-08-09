"""Inverse-volatility correlation-driven portfolio strategy.

Replicates the removed ``bt pf --weight-method invvol`` scheme as a `bt run`-able
strategy. At each rebalance boundary it holds inverse-volatility weights
(``w_i ∝ 1/σ_i``) over the trailing returns of a whole universe. Weights move
only when trailing volatilities move, so turnover is typically much lower than
the covariance-based GMV scheme. Long-only; undeployed capital is held as cash.
Single-interval (``1d``) universe backtests.

No lookahead: weights at bar ``t`` use only returns through ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.state import BacktestState, Candle, TradeSignal
from src.bt.strategies.portfolio_engine import pf_on_candle
from src.bt.strategies.portfolio_weights import inverse_vol_weights
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "pf_invvol"

# Default turnover cost rate (bps), used by runtime_stats() when the caller
# does not supply the resolved Params-carrying bps.
_DEFAULT_COST_BPS = 10.0


@dataclass(frozen=True)
class Params(StrategyParams):
    # Rebalance cadence name matching the former `bt pf` (monthly default).
    rebalance: str = "monthly"
    # Trailing bar count used to estimate volatility at each rebalance.
    lookback: int = 252
    # Per-asset maximum weight in (0, 1); 1.0 disables the cap.
    max_weight: float = 1.0
    # Warmup (daily bars) before the first rebalance.
    warmup_bars: int = 252
    # Bar interval to read closes from (must be present in config.bars).
    interval: str = "1d"
    # One-way turnover cost rate in bps, charged once per rebalance on gross
    # turnover (replaces per-fill commission+slippage for pf runs).
    cost_bps: float = 10.0


# ---------------------------------------------------------------------------
# module-level state
# Repo GLOBAL-dict convention + reset_global() for the split engine.
# ---------------------------------------------------------------------------

GLOBAL: dict = {
    "last_signal_close": {},  # symbol -> last close (new-bar detection)
    "bar_idx": 0,  # count of signal-interval bars seen
    "next_rebalance": None,  # next calendar cadence boundary (pd.Timestamp)
    "n_rebalances": 0,  # cadence firings (portfolio report)
    "gross_turnover": 0.0,  # cumulative one-way weight turnover
    "last_plan": "init",
}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {
        "last_signal_close": {},
        "bar_idx": 0,
        "next_rebalance": None,
        "n_rebalances": 0,
        "gross_turnover": 0.0,
        "last_plan": "init",
    }


def runtime_stats(cost_bps: float | None = None) -> dict:
    """Runtime pf report inputs read after a run (see src.bt.metrics)."""
    return {
        "n_rebalances": GLOBAL["n_rebalances"],
        "gross_turnover": GLOBAL["gross_turnover"],
        "turnover_cost_bps": cost_bps if cost_bps is not None else _DEFAULT_COST_BPS,
    }


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    return pf_on_candle(
        state,
        candle,
        GLOBAL,
        inverse_vol_weights,
        interval=params.interval,
        rebalance=params.rebalance,
        lookback=params.lookback,
        max_weight=params.max_weight,
        warmup_bars=params.warmup_bars,
    )
