"""Kalman-filter pairs trading — pure signal logic.

Relies on a kalman_pairs model_updater_fn to compute:
  - state.model_state.kalman_z_score   (rolling z-score of Kalman spread)
  - state.model_state.kalman_beta      (hedge ratio)
  - state.model_state.kalman_spread    (raw innovation)
  - state.model_state.kalman_n_steps   (for warmup gating)

The model updater runs the PairsKalmanOnline filter on every candle
before this strategy sees the state. kalman_z_score is the rolling
z-score of the Kalman innovation (spread). The Kalman's intercept α
makes the spread mean-zero by construction, so the rolling-z is a
tradable signal in the ±2–3 range.

Entry: |z| > z_entry  → short overpriced (z>0) or long underpriced (z<0)
Exit:  |z| < z_exit   → convergence, OR divergence stop at |z| > z_exit_stop

Beta-weighted pairs trading. Each leg is sized proportionally to the
Kalman hedge ratio β so that notional exposure of leg2 ≈ β × leg1.
Pair resolved via strategy_params.pair or auto-detected from first two
symbols in state.candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.bt.regime.types import TREND_INT_TO_LABEL
from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "kalman_pairs"


# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    """Strategy parameters for kalman_pairs.

    Kalman hyperparameters (process_noise, measurement_noise, etc.)
    live in the model_updater config, not here. These are signal-only.
    """

    # Entry/exit thresholds on Kalman t-stat (standardized innovation)
    z_entry: float = 2.0
    z_exit: float = 0.5

    # Asymmetric risk — divergence stop
    z_exit_stop: float = 3.5

    # Position sizing
    position_size_pct: float = 0.25

    # Regime gate
    regime_gate: bool = False

    # Warmup bars (checked against kalman_n_steps indirectly;
    # the model updater returns early before warmup_bars,
    # so kalman_z_score stays None during warmup)
    warmup_bars: int = 150

    # Pairs — auto-detected from symbols[0], symbols[1] if None
    pair: tuple[str, str] | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_pair(state: BacktestState, params: Params) -> tuple[str, str] | None:
    """Resolve the pair from params or state.candles."""
    if params.pair is not None:
        return params.pair
    symbols = sorted({s for s, _ in state.candles})
    if len(symbols) >= 2:
        return (symbols[0], symbols[1])
    return None


def _close_pair(
    state: BacktestState,
    candle: Candle,
    s1: str,
    s2: str,
    reason: str,
    z: float | None = None,
) -> list[TradeSignal]:
    """Close any open positions for the pair, scoped to s1 and s2."""
    signals: list[TradeSignal] = []
    for sym in (s1, s2):
        for pos in state.portfolio.positions.get(sym, ()):
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=candle.close,
                    qty=abs(pos.qty),
                    position_id=pos.position_id,
                    reason=reason,
                    z_score=z,
                )
            )
    return signals


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    ms = state.model_state

    # ---- warmup guard ----
    z = ms.kalman_z_score
    if z is None:
        return []

    beta = ms.kalman_beta
    if beta is None:
        return []

    # ---- pair resolution ----
    pair = _resolve_pair(state, params)
    if pair is None:
        return []
    s1, s2 = pair

    interval = candle.interval or "1h"

    # ---- get prices for sizing ----
    df1 = state.candles.get((s1, interval))
    df2 = state.candles.get((s2, interval))
    if df1 is None or df2 is None:
        return []

    closes1 = cast(pd.Series, df1["close"])
    closes2 = cast(pd.Series, df2["close"])
    aligned = pd.concat([closes1.rename("a"), closes2.rename("b")], axis=1).dropna()
    if len(aligned) == 0:
        return []

    p1 = float(aligned["a"].iloc[-1])
    p2 = float(aligned["b"].iloc[-1])

    # ---- Regime gate ----
    if params.regime_gate:
        trend = state.model_state.current_trend
        if trend is not None:
            label = TREND_INT_TO_LABEL.get(trend)
            if label == "BEAR":
                return _close_pair(state, candle, s1, s2, "[BEAR] regime", z)

    # ---- current positions ----
    pos1 = state.portfolio.positions.get(s1, ())
    pos2 = state.portfolio.positions.get(s2, ())
    has_pos = bool(pos1) or bool(pos2)

    # ---- Exit ----
    if has_pos:
        # Divergence stop
        if abs(z) > params.z_exit_stop:
            return _close_pair(
                state, candle, s1, s2, f"kalman divergence stop z={z:.2f}", z
            )

        # Convergence exit.
        # z_exit > 0  → absolute threshold: exit when |z| < z_exit
        # z_exit <= 0 → zero-cross exit: exit when z crosses zero
        if params.z_exit > 0:
            exit_trigger = abs(z) < params.z_exit
            reason = f"kalman convergence z={z:.2f}"
        else:
            first_leg = pos1[0] if pos1 else pos2[0]
            entry_z_pos = first_leg.type == ActionType.short
            exit_trigger = z <= 0 if entry_z_pos else z >= 0
            reason = f"kalman zero-cross z={z:.2f}"

        if exit_trigger:
            return _close_pair(state, candle, s1, s2, reason, z)

    # ---- Entry ----
    if not has_pos and abs(z) > params.z_entry:
        if abs(z) < 1e-10 or p1 <= 0 or p2 <= 0:
            return []

        direction = "overpriced" if z > 0 else "underpriced"
        leg1_action = ActionType.short if z > 0 else ActionType.long
        leg2_action = ActionType.long if z > 0 else ActionType.short

        cash = state.portfolio.cash
        pos_pct = params.position_size_pct
        leg1_value = cash * pos_pct
        leg2_value = leg1_value * abs(beta) if abs(beta) > 1e-12 else leg1_value

        qty1 = round(leg1_value / p1, 4)
        qty2 = round(leg2_value / p2, 4)

        return [
            TradeSignal(
                action=leg1_action,
                symbol=s1,
                timestamp=candle.timestamp,
                price=p1,
                qty=qty1,
                reason=f"kalman({direction}) z={z:.1f} β={beta:.2f}",
                z_score=z,
                hedge_beta=beta,
            ),
            TradeSignal(
                action=leg2_action,
                symbol=s2,
                timestamp=candle.timestamp,
                price=p2,
                qty=qty2,
                reason=f"kalman({direction}) z={z:.1f} β={beta:.2f}",
                z_score=z,
                hedge_beta=beta,
            ),
        ]

    return []
