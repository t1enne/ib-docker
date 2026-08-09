"""Fixed-allocation periodically-rebalanced (PRP) portfolio strategy.

Replicates the ``--allocations`` mode of the removed ``bt pf`` command as a
`bt run`-able strategy. Holds constant target allocations across a universe and
drifts with price between rebalances, snapping back to those targets on cadence
(a classic fixed-weight, periodically-rebalanced portfolio). No estimation —
weights are the constant targets; unallocated capital is held as cash.

Example exact allocations:
    {"SPY": 0.50, "GLD": 0.15, "TLT": 0.20, "DBA": 0.15}

Targets need not sum to 1 — they are normalised and any remainder is cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.bt.state import BacktestState, Candle, TradeSignal
from src.bt.strategies.portfolio_engine import pf_on_candle
from src.bt.strategies.portfolio_weights import fixed_alloc_weights
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "pf_alloc"

# Fixed allocations don't need a trailing-return estimate; this is just the
# minimum window the shared driver needs to build a returns frame (2 bars).
_MIN_LOOKBACK = 2

# Default turnover cost rate (bps), used by runtime_stats() when the caller
# does not supply the resolved Params-carrying bps.
_DEFAULT_COST_BPS = 10.0


@dataclass(frozen=True)
class Params(StrategyParams):
    # Constant target allocations: {symbol: weight}. Normalised; residual = cash.
    allocations: dict[str, float] = field(default_factory=dict)
    # Rebalance cadence name (monthly default).
    rebalance: str = "monthly"
    # Per-asset maximum weight in (0, 1); 1.0 disables the cap.
    max_weight: float = 1.0
    # Bar interval to read closes from (must be present in config.bars).
    interval: str = "1d"
    # Warmup (daily bars) before the first rebalance (P1-5 shared rule).
    warmup_bars: int = 2
    # One-way turnover cost rate in bps, charged once per rebalance on gross
    # turnover (replaces per-fill commission+slippage for pf runs).
    cost_bps: float = 10.0

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        params = super().from_dict(d)
        if not params.allocations:
            raise ValueError("pf_alloc requires a non-empty 'allocations' map")
        return params


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
    "weight_fn": None,  # cached WeightMethodFn built from allocations
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
        "weight_fn": None,
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
    if not params.allocations:
        return []
    if GLOBAL["weight_fn"] is None:
        GLOBAL["weight_fn"] = fixed_alloc_weights(params.allocations)

    return pf_on_candle(
        state,
        candle,
        GLOBAL,
        GLOBAL["weight_fn"],
        interval=params.interval,
        rebalance=params.rebalance,
        lookback=_MIN_LOOKBACK,
        max_weight=params.max_weight,
        warmup_bars=params.warmup_bars,
    )
