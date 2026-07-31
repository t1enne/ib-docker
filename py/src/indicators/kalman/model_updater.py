"""Kalman pairs-trading model updater factory.

Creates a ModelUpdaterFn: (BacktestState, Candle) -> BacktestState
that runs the pairs Kalman filter on every candle and writes the
outputs into ModelState.kalman_* fields.

kalman_z_score = rolling z-score of the Kalman innovation (spread).
The Kalman's intercept α makes the spread mean-zero by construction,
so the rolling z-score is spread / σ_spread_window — a tradable
signal in the ±2–3 range, not the raw t_stat (which is spread / √S
and typically 0.005–0.05 due to large √S).

Usage — in backtest engine _resolve_model_updater:

    "model_updater": {
        "type": "kalman_pairs",
        "kalman_pairs": {
            "process_noise": 1e-4,
            "measurement_noise": 1e-3,
            "ols_warmup": 50,
            "adaptive": true,
            "vol_window": 20,
            "z_window": 20,
            "warmup_bars": 150
        }
    }

The strategy reads state.model_state.kalman_z_score directly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd

from src.bt.state import BacktestState, Candle
from src.bt.engine.utils import merge_bt_state
from src.indicators.kalman.online import PairsKalmanOnline
from src.indicators.kalman.types import PairsKalmanConfig


def _rolling_zscore(
    spread_history: deque[float],
    current: float,
    window: int,
) -> float:
    """Rolling z-score of spread: (current - mean) / std."""
    spread_history.append(current)
    if len(spread_history) < max(window, 3):
        return 0.0
    vals = list(spread_history)[-window:]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((current - mean) / std)


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


def create_kalman_pairs_updater(
    pair: tuple[str, str] | None = None,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
    ols_warmup: int = 50,
    adaptive: bool = True,
    vol_window: int = 20,
    z_window: int = 20,
    warmup_bars: int = 150,
) -> Callable[[BacktestState, Candle], BacktestState]:
    """Create a ModelUpdaterFn that runs the pairs Kalman filter.

    Closed-over mutable state (one instance per backtest run):
      - PairsKalmanOnline instance
      - deque for rolling z-score of spread
      - resolved pair (for re-init on config changes)

    Writes ModelState fields:
      - kalman_z_score   = rolling z-score of Kalman spread (tradable ~±2)
      - kalman_spread    = kf.spread (raw innovation, mean-zero by construction)
      - kalman_beta      = kf.beta
      - kalman_alpha     = kf.alpha
      - kalman_n_steps   = kf.n_steps
      - hedge_beta       = kf.beta (synced for downstream consumers)
    """

    # ---- closure state ----
    kf: PairsKalmanOnline | None = None
    resolved_pair: tuple[str, str] | None = None
    spread_history: deque[float] = deque(maxlen=max(z_window, warmup_bars))

    def update(state: BacktestState, candle: Candle) -> BacktestState:
        nonlocal kf, resolved_pair, spread_history

        interval = candle.interval or "1h"

        # ---- resolve pair ----
        symbols = sorted({s for s, _ in state.candles})
        if pair is not None:
            s1, s2 = pair
        elif len(symbols) >= 2:
            s1, s2 = symbols[0], symbols[1]
        else:
            return state

        # Re-init Kalman when pair changes
        if resolved_pair != (s1, s2) or kf is None:
            resolved_pair = (s1, s2)
            spread_history.clear()
            kf_cfg = PairsKalmanConfig(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                mean_halflife=ols_warmup,
                adaptive=adaptive,
                vol_window=vol_window,
            )
            kf = PairsKalmanOnline(config=kf_cfg)

        # ---- get closes for both legs ----
        df1 = state.candles.get((s1, interval))
        df2 = state.candles.get((s2, interval))
        if df1 is None or df2 is None or len(df1) < 2 or len(df2) < 2:
            return state

        closes1 = cast(pd.Series, df1["close"])
        closes2 = cast(pd.Series, df2["close"])

        aligned = pd.concat([closes1.rename("a"), closes2.rename("b")], axis=1).dropna()
        if len(aligned) < warmup_bars:
            return state

        # ---- warm-start Kalman if not yet fitted ----
        if kf.n_steps < ols_warmup:
            _ols_warmup(
                kf,
                cast(pd.Series, aligned["a"]),
                cast(pd.Series, aligned["b"]),
                ols_warmup,
            )
            if kf.n_steps < 3:
                return state

        # ---- update Kalman with latest log-prices ----
        log_p1 = float(np.log(aligned["a"].iloc[-1]))
        log_p2 = float(np.log(aligned["b"].iloc[-1]))
        kf.update(log_p1, log_p2)

        # Rolling z-score of the Kalman innovation (spread).
        # Spread is mean-zero by construction (intercept α in state),
        # so the rolling-z is spread / σ_spread — tradable ~±2.
        spread = kf.spread
        z = _rolling_zscore(spread_history, spread, z_window)

        # ---- write to ModelState ----
        new_ms = replace(
            state.model_state,
            kalman_spread=float(spread),
            kalman_z_score=z,
            kalman_beta=float(kf.beta),
            kalman_alpha=float(kf.alpha),
            kalman_n_steps=kf.n_steps,
            hedge_beta=float(kf.beta),
        )
        return merge_bt_state(state, dict(model_state=new_ms))

    return update
