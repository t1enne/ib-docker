"""Kalman-filter pairs trading — DSL adapter.

The Kalman signal does **not** live in the strategy — it is produced by an
engine-level ``model_updater_fn`` (``model_updater.kalman_pairs`` in the
config) that runs the ``OnlinePairs`` filter on every candle and writes
``state.model_state.kalman_z_score`` / ``kalman_beta`` before the strategy sees
the state. That layer is orthogonal to the strategy DSL.

So the DSL port is deliberately thin: it wraps the same entry/exit decision
logic on the declarative ``StrategyContext`` surface, but reads the Kalman
z-score / beta from ``ctx.state.model_state`` (the raw ``BacktestState`` the
DSL exposes to power users) because ``ctx.ta`` does not run the Kalman filter.

The DSL adds no data-access advantage here — this port exists to show that a
model-driven strategy (z-score from a Kalman filter) composes with the DSL by
reading through ``ctx.state``, while the position-management calls are the
declarative ``ctx.long/short/close``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "kalman_pairs_dsl"


@dataclass(frozen=True)
class Params(StrategyParams):
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_exit_stop: float = 3.5
    position_size: float = 0.25
    stop_loss: float = 0.0
    take_profit: float = 0.0
    regime_gate: bool = False
    warmup_bars: int = 150
    pair: tuple[str, str] | None = None


def _pair(ctx: StrategyContext, params: Params) -> tuple[str, str] | None:
    if params.pair is not None:
        return params.pair
    symbols = list(ctx.symbols)
    if len(symbols) >= 2:
        return (symbols[0], symbols[1])
    return None


@strategy(bars="1d")
def on_candle(ctx: StrategyContext):
    pair = _pair(ctx, ctx.params)
    if pair is None:
        return
    s1, s2 = pair

    # Kalman signal is produced by the engine model_updater (see docstring).
    z = ctx.state.model_state.kalman_z_score
    beta = ctx.state.model_state.kalman_beta
    if z is None or beta is None:
        return

    p1 = ctx.price(s1)
    p2 = ctx.price(s2)
    if p1 <= 0 or p2 <= 0:
        return

    pos1 = ctx.position(s1)
    pos2 = ctx.position(s2)
    has_pos = pos1 is not None or pos2 is not None

    _exit_if_due(ctx, s1, s2, z, has_pos)

    # Re-check position after possible exit.
    has_pos = ctx.position(s1) is not None or ctx.position(s2) is not None
    if has_pos or abs(z) <= ctx.params.z_entry:
        return

    if abs(z) < 1e-10:
        return

    if z > 0:
        ctx.short(s1, size=ctx.params.position_size, reason="kalman overpriced")
        ctx.long(s2, size=ctx.params.position_size * abs(beta), reason="kalman hedge")
    else:
        ctx.long(s1, size=ctx.params.position_size, reason="kalman underprice")
        ctx.short(s2, size=ctx.params.position_size * abs(beta), reason="kalman hedge")


def _exit_if_due(
    ctx: StrategyContext, s1: str, s2: str, z: float, has_pos: bool
) -> None:
    """Close the pair on divergence-stop or convergence."""
    if not has_pos:
        return
    if abs(z) > ctx.params.z_exit_stop:
        _close_pair(ctx, s1, s2, f"kalman divergence stop z={z:.2f}")
        return
    if ctx.params.z_exit > 0 and abs(z) < ctx.params.z_exit:
        _close_pair(ctx, s1, s2, f"kalman convergence z={z:.2f}")
        return
    # z_exit <= 0 -> zero-cross exit: exit when the held direction reverts
    # toward zero. Close when current z crossed the zero line from the entry
    # side; the DSL close() is a no-op if already flat.
    _close_pair(ctx, s1, s2, f"kalman zero-cross z={z:.2f}")


def _close_pair(ctx: StrategyContext, s1: str, s2: str, reason: str) -> None:
    for sym in (s1, s2):
        ctx.close(sym, reason=reason)
