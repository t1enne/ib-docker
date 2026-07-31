"""Kalman-filter pairs trading — adaptive cointegration.

Uses PairsKalmanOnline for real-time [α, β] estimation.
Trading signal: rolling z-score of the Kalman's standardized innovation
(t_stat = spread / √S).  The Kalman normalizes by its own uncertainty;
the rolling z-score over t_stat provides the historical context needed
for interpretable entry/exit thresholds.

Entry: |z| > z_entry  → short overpriced (z>0) or long underpriced (z<0)
Exit:  |z| < z_exit   → convergence, OR divergence stop

Enhancements:
  - Asymmetric risk: divergence stop at |z| > z_exit_stop
  - Vol-scaled sizing: position size inversely proportional to spread vol
  - Momentum entry gate: only enter when |z| is peaking (not expanding)

Dollar-neutral with beta hedging.  β is log-space elasticity from the
Kalman, converted to price-level hedge ratio: qty2 = qty1 * |β| * p1/p2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams
from src.indicators.kalman import PairsKalmanOnline, PairsKalmanConfig

STRATEGY_TYPE = "kalman_pairs"


# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Kalman filter config
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    ols_warmup: int = 50
    adaptive: bool = True
    vol_window: int = 20

    # Rolling z-score window over the Kalman t_stat
    z_window: int = 20

    # Entry/exit thresholds on the rolling z-score of t_stat
    z_entry: float = 2.0
    z_exit: float = 0.5

    # Asymmetric risk — divergence stop.  Exit if |z| exceeds this.
    z_exit_stop: float = 3.5

    # Position sizing
    position_size_pct: float = 0.25
    vol_scale_enabled: bool = True
    vol_scale_lookback: int = 60

    # Entry gate: momentum filter
    # Only enter if |z| is not still expanding
    momentum_gate: bool = True

    # Regime gate
    regime_gate: bool = False

    # Warmup
    warmup_bars: int = 150

    # Pairs — auto-detected from symbols[0], symbols[1] if None
    pair: tuple[str, str] | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ols_warmup(
    kf: PairsKalmanOnline,
    closes1: pd.Series,
    closes2: pd.Series,
    window: int,
) -> None:
    n = min(len(closes1), len(closes2), window)
    if n < 3:
        return
    lp1 = np.log(closes1.iloc[-n:].values)
    lp2 = np.log(closes2.iloc[-n:].values)
    kf.init_from_ols(lp1, lp2)


def _rolling_zscore(
    history: deque[float],
    current: float,
    window: int,
) -> float:
    """Compute rolling z-score of *current* against *history*."""
    if len(history) < window:
        history.append(current)
        return 0.0

    history.append(current)
    vals = list(history)[-window:]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((current - mean) / std)


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


# Module-level mutable state — ponytail: global state, replace with
# state.model_state.kalman_pairs when model_state persistence is ready.
_kf: PairsKalmanOnline | None = None
_pair: tuple[str, str] | None = None
_t_history: deque[float] = deque()  # t_stat for rolling z-score + momentum gate
_z_history: deque[float] = deque()  # computed z-scores for momentum gate + vol scaling


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    global _kf, _pair, _t_history, _z_history

    interval = candle.interval or "1h"

    # Resolve pair
    symbols = sorted({s for s, _ in state.candles})
    if params.pair is not None:
        s1, s2 = params.pair
    elif len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
    else:
        return []

    # Re-init Kalman when pair changes
    if _pair != (s1, s2) or _kf is None:
        _pair = (s1, s2)
        _t_history.clear()
        _z_history.clear()
        kf_cfg = PairsKalmanConfig(
            process_noise=params.process_noise,
            measurement_noise=params.measurement_noise,
            mean_halflife=params.ols_warmup,
            adaptive=params.adaptive,
            vol_window=params.vol_window,
        )
        _kf = PairsKalmanOnline(config=kf_cfg)

    # Get closes for both legs
    df1 = state.candles.get((s1, interval))
    df2 = state.candles.get((s2, interval))
    if df1 is None or df2 is None or len(df1) < 2 or len(df2) < 2:
        return []

    closes1 = cast(pd.Series, df1["close"])
    closes2 = cast(pd.Series, df2["close"])

    # Align on inner join
    aligned = pd.concat([closes1.rename("a"), closes2.rename("b")], axis=1).dropna()
    if len(aligned) < params.warmup_bars:
        return []

    # Warm-start Kalman if not yet fitted
    if _kf.n_steps < params.ols_warmup:
        _ols_warmup(
            _kf,
            cast(pd.Series, aligned["a"]),
            cast(pd.Series, aligned["b"]),
            params.ols_warmup,
        )
        if _kf.n_steps < 3:
            return []

    # Update Kalman with latest log-prices
    log_p1 = float(np.log(aligned["a"].iloc[-1]))
    log_p2 = float(np.log(aligned["b"].iloc[-1]))
    _kf.update(log_p1, log_p2)
    t_stat = _kf.t_stat
    beta = _kf.beta

    # Rolling z-score over t_stat — THE trading signal
    z = _rolling_zscore(_t_history, t_stat, params.z_window)
    _z_history.append(z)

    # ---- Regime gate ----
    if params.regime_gate:
        trend = state.model_state.current_trend
        if trend is not None:
            from src.bt.regime.types import TREND_INT_TO_LABEL

            label = TREND_INT_TO_LABEL.get(trend)
            if label == "BEAR":
                return _close_all(state, candle, "[BEAR] regime")

    # ---- Current positions ----
    pos1 = state.portfolio.positions.get(s1, ())
    pos2 = state.portfolio.positions.get(s2, ())
    has_pos = bool(pos1) or bool(pos2)

    p1 = float(aligned["a"].iloc[-1])
    p2 = float(aligned["b"].iloc[-1])

    # ---- Exit ----
    if has_pos:
        # Divergence stop
        if abs(z) > params.z_exit_stop:
            return _close_all(state, candle, f"kalman divergence stop z={z:.2f}")

        # Convergence exit (or zero-cross if z_exit=0)
        if params.z_exit > 0:
            exit_trigger = abs(z) < params.z_exit
            reason = f"kalman convergence z={z:.2f}"
        else:
            first_leg = pos1[0] if pos1 else pos2[0]
            entry_z_positive = first_leg.type == ActionType.short
            exit_trigger = z <= 0 if entry_z_positive else z >= 0
            reason = f"kalman zero-cross z={z:.2f}"

        if exit_trigger:
            return _close_all(state, candle, reason)

    # ---- Entry ----
    if not has_pos and abs(z) > params.z_entry:
        if abs(z) < 1e-10:
            return []

        # --- Momentum gate: only enter if |z| is not expanding ---
        if params.momentum_gate and len(_z_history) >= 3:
            recent_abs = [abs(_z_history[-3]), abs(_z_history[-2])]
            if abs(z) > max(recent_abs):
                return []

        # --- Vol-scaled position sizing ---
        pos_pct = params.position_size_pct
        if params.vol_scale_enabled and len(_z_history) >= params.vol_scale_lookback:
            recent_abs_z = [
                abs(v) for v in list(_z_history)[-params.vol_scale_lookback :]
            ]
            current_vol = float(np.mean(recent_abs_z))
            baseline_vol = float(np.median(recent_abs_z))
            if baseline_vol > 1e-10 and current_vol > 1e-10:
                scalar = baseline_vol / current_vol
                scalar = max(0.5, min(1.5, scalar))
            else:
                scalar = 1.0
        else:
            scalar = 1.0

        direction = "overpriced" if z > 0 else "underpriced"
        leg1_action = ActionType.short if z > 0 else ActionType.long
        leg2_action = ActionType.long if z > 0 else ActionType.short

        cash = state.portfolio.cash
        if p1 <= 0 or p2 <= 0:
            return []

        effective_pct = pos_pct * scalar
        leg1_value = cash * effective_pct
        qty1 = round(leg1_value / p1, 4)

        # Beta hedging: β is log-space elasticity. Convert to price-level
        # hedge ratio so the dollar exposure matches.
        beta_abs = abs(beta) if abs(beta) > 1e-12 else 1.0
        qty2 = round(qty1 * beta_abs * p1 / p2, 4)

        scale_tag = f" vol={scalar:.2f}" if params.vol_scale_enabled else ""

        return [
            TradeSignal(
                action=leg1_action,
                symbol=s1,
                timestamp=candle.timestamp,
                price=p1,
                qty=qty1,
                reason=f"kalman({direction}) z={z:.1f} β={beta:.2f}{scale_tag}",
            ),
            TradeSignal(
                action=leg2_action,
                symbol=s2,
                timestamp=candle.timestamp,
                price=p2,
                qty=qty2,
                reason=f"kalman({direction}) z={z:.1f} β={beta:.2f}{scale_tag}",
            ),
        ]

    return []


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
