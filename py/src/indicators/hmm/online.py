"""Online (step-by-step) HMM regime detector for real-time / backtest use.

Since hmmlearn has no partial_fit, we simulate online behaviour:
- Refit the full HMM every retrain_interval bars on a rolling window.
- Between refits, predict the current regime from the latest feature row.
- The ``update()`` method returns the current regime label (0/1/2).

Usage in the backtest engine's model_updater_fn::

    hmm = MarketRegimeHMMOnline(retrain_interval=50, window_size=500)
    ...
    for each tick:
        regime = hmm.update(price)
        state.model_state.current_regime = regime
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import warnings

from hmmlearn import hmm


class MarketRegimeHMMOnline:
    """Online HMM regime detector — periodic batch refit on rolling window.

    Maintains a rolling window of prices internally. Every retrain_interval
    bars, refits the full HMM on the window. Between refits, runs a
    single-step predict on the latest feature vector.

    This avoids the O(n) full-series predict on every tick — only the
    rolling-window fit (O(window_size)) is expensive, and it runs every
    N bars. The per-tick cost is O(1): compute 3 features, predict one row.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        window_size: int = 500,
        vol_window: int = 20,
        momentum_window: int = 10,
        retrain_interval: int = 50,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.retrain_interval = retrain_interval
        self.random_state = random_state

        # Rolling window of prices
        self._prices: deque[float] = deque(maxlen=window_size)

        # Fitted model (None until first refit)
        self._model: hmm.GaussianHMM | None = None
        self._state_to_regime: dict[int, int] = {}
        self._fitted: bool = False

        # Step counter
        self._n: int = 0
        self._last_refit: int = -window_size  # force refit at bar 0

        # Current regime
        self._current_regime: int = -1

        # Regime labels (filled after first fit)
        self.regime_labels: dict[int, str] = {
            0: "Low Vol",
            1: "Medium Vol",
            2: "High Vol",
        }

    # ------------------------------------------------------------------
    # read-only properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> int:
        """Current regime label (0=low vol, 1=med, 2=high). -1 if not yet fitted."""
        return self._current_regime

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_steps(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, price: float) -> int:
        """Process one price observation, return current regime.

        Returns -1 until enough data for first fit.
        """
        self._prices.append(price)
        self._n += 1

        # Need enough data for features before anything
        needed = max(self.vol_window, self.momentum_window) + 1
        if len(self._prices) < needed:
            return -1

        # Refit every retrain_interval bars
        bars_since_refit = self._n - self._last_refit
        if (
            bars_since_refit >= self.retrain_interval
            and len(self._prices) >= self.window_size
        ):
            self._refit()

        if not self._fitted:
            return -1

        # Predict current regime from latest feature row
        features = self._latest_features()
        assert self._model is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.predict(features.reshape(1, -1))[0]
        self._current_regime = self._state_to_regime.get(raw, raw)
        return self._current_regime

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _to_series(self) -> pd.Series:
        return pd.Series(list(self._prices))

    def _compute_features(self) -> pd.DataFrame:
        """Build feature DataFrame from the rolling window."""
        prices = self._to_series()
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(self.vol_window).std() * np.sqrt(252)
        momentum = returns.rolling(self.momentum_window).mean() * 252
        return pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
            }
        ).dropna()

    def _latest_features(self) -> np.ndarray:
        """Compute feature vector for the latest bar only."""
        prices = self._to_series()
        if len(prices) < 2:
            return np.zeros(3)
        ret = np.log(prices.iloc[-1] / prices.iloc[-2])
        vol = prices.pct_change().rolling(self.vol_window).std().iloc[-1] * np.sqrt(252)
        mom = prices.pct_change().rolling(self.momentum_window).mean().iloc[-1] * 252
        return np.array(
            [ret, vol if not pd.isna(vol) else 0.0, mom if not pd.isna(mom) else 0.0]
        )

    def _refit(self) -> None:
        """Fit HMM on current rolling window and build vol-ranked remap."""
        import sys
        import io

        features = self._compute_features()
        if len(features) < 50:
            return

        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_iter=200,
            tol=1e-3,
        )

        # hmmlearn prints convergence warnings to stderr from Cython.
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(features.values)
        finally:
            sys.stderr = old_stderr

        # Build vol-ranked remap: 0=lowest vol, N-1=highest
        raw_states = self._model.predict(features.values)
        vol_by_state: dict[int, float] = {}
        for s in range(self.n_regimes):
            mask = raw_states == s
            vol_by_state[s] = (
                float(features["volatility"][mask].mean()) if mask.any() else 0.0
            )

        sorted_states = sorted(vol_by_state, key=lambda s: vol_by_state[s])
        self._state_to_regime = {old: new for new, old in enumerate(sorted_states)}

        self._fitted = True
        self._last_refit = self._n

        # Update labels
        label_map = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
        self.regime_labels = {
            i: label_map.get(i, f"Regime {i}") for i in range(self.n_regimes)
        }
