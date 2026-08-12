"""Strategy-facing online pairs Kalman owner.

``OnlinePairs`` is the strategy-level owner of the pairs Kalman filter — the
successor to the removed engine ``model_updater`` channel. Construct one in the
strategy's ``reset_global()`` (or lazily in the DSL's ``ctx.shared``), hold it
in state, and call ``observe()`` once per candle. It reads both legs' closes
from ``CandleStore`` (cursor-safe, no look-ahead), OLS-warm-starts the filter,
updates it, and tracks the rolling z-score of the Kalman spread (the tradable
signal, ~±2).

Usage (inside a strategy's ``on_candle``)::

    GLOBAL = {"kf": OnlinePairs(process_noise=1e-4, measurement_noise=1e-3)}

    def reset_global() -> None:
        global GLOBAL
        GLOBAL = {"kf": OnlinePairs(process_noise=1e-4, measurement_noise=1e-3)}

    def on_candle(state, candle, params):
        result = GLOBAL["kf"].observe(state, "SPY", "QQQ", candle.interval or "1h")
        if not result.ready:
            return []  # warmup
        if abs(result.z_score) > params.z_entry:
            ...  # enter
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.indicators.kalman.online import PairsKalmanOnline
from src.indicators.kalman.types import PairsKalmanConfig
from src.bt.state import BacktestState


@dataclass(frozen=True)
class OnlinePairsResult:
    """One observation's Kalman outputs."""

    z_score: float | None  # rolling z of Kalman spread (tradable ~±2)
    beta: float | None
    alpha: float | None
    n_steps: int
    ready: bool  # False during warmup (insufficient aligned history)


def _rolling_zscore(deq: deque[float], current: float, window: int) -> float:
    """Append ``current`` and return its rolling z-score within ``deq``."""
    deq.append(current)
    if len(deq) < max(window, 3):
        return 0.0
    vals = list(deq)[-window:]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((current - mean) / std)


class OnlinePairs:
    """Online pairs Kalman + rolling z-score, owned by a strategy.

    Parameters mirror the removed engine ``model_updater`` factory. Re-constructing
    in ``reset_global()`` (or fresh in ``ctx.shared``) rebuilds filter and history
    for a clean IS/OOS split.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        ols_warmup: int = 50,
        adaptive: bool = True,
        vol_window: int = 20,
        z_window: int = 20,
        warmup_bars: int = 150,
    ) -> None:
        self._ols_warmup = ols_warmup
        self._z_window = z_window
        self._warmup_bars = warmup_bars

        kf_cfg = PairsKalmanConfig(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            mean_halflife=ols_warmup,
            adaptive=adaptive,
            vol_window=vol_window,
        )
        self._kf = PairsKalmanOnline(config=kf_cfg)
        self._spread_history: deque[float] = deque(maxlen=max(z_window, warmup_bars))

    @property
    def kf(self) -> PairsKalmanOnline:
        """Underlying filter (for diagnostics / custom warm-start)."""
        return self._kf

    def _ols_warmstart(
        self, closes1: pd.Series, closes2: pd.Series, window: int
    ) -> None:
        if self._kf.n_steps >= window:
            return
        n = min(len(closes1), len(closes2), window)
        if n < 3:
            return
        lp1 = np.log(closes1.iloc[-n:].values)
        lp2 = np.log(closes2.iloc[-n:].values)
        self._kf.init_from_ols(lp1, lp2)

    def observe(
        self,
        state: BacktestState,
        s1: str,
        s2: str,
        interval: str,
    ) -> OnlinePairsResult:
        """Feed the pair's latest closes to the Kalman filter.

        Returns a result with ``ready=False`` until sufficient aligned history
        exists (warmup) or either leg is missing. ``ready`` becomes True after
        warmup, even if the rolling z-score is still 0.
        """
        result = OnlinePairsResult(None, None, None, self._kf.n_steps, False)

        df1 = state.candles.get((s1, interval))
        df2 = state.candles.get((s2, interval))
        if df1 is None or df2 is None or len(df1) < 2 or len(df2) < 2:
            return result

        c1 = cast(pd.Series, df1["close"])
        c2 = cast(pd.Series, df2["close"])
        aligned = pd.concat([c1.rename("a"), c2.rename("b")], axis=1).dropna()
        if len(aligned) < self._warmup_bars:
            return result

        # OLS warm-start the filter on the initial window if not yet fitted.
        self._ols_warmstart(
            cast(pd.Series, aligned["a"]),
            cast(pd.Series, aligned["b"]),
            self._ols_warmup,
        )
        if self._kf.n_steps < 3:
            return result

        log_p1 = float(np.log(aligned["a"].iloc[-1]))
        log_p2 = float(np.log(aligned["b"].iloc[-1]))
        self._kf.update(log_p1, log_p2)

        spread = self._kf.spread
        z = _rolling_zscore(self._spread_history, spread, self._z_window)

        return OnlinePairsResult(
            z_score=z,
            beta=float(self._kf.beta),
            alpha=float(self._kf.alpha),
            n_steps=self._kf.n_steps,
            ready=True,
        )
