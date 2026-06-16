"""
Online (step-by-step) Kalman filters for real-time / backtest use.

Two-state [α, β] model — canonical in practitioner literature
(Robot Wealth, Quantopian, Montana et al. 2009).

These classes manage internal state via explicit init / update methods.
No filterpy dependency in the hot path — Kalman math uses 2×2 matrix
arithmetic directly.

Usage in the backtest engine's model_updater_fn::

    cfg = PairsKalmanConfig(process_noise=1e-4, measurement_noise=1e-3)
    kf = PairsKalmanOnline(config=cfg)
    # Warm-start from initial OLS window
    kf.init_from_ols(log_p1_arr, log_p2_arr)
    ...
    for each tick:
        t = kf.update(log_p1_t, log_p2_t)
        state.model_state.z_score = t       # t_stat IS the trading signal
        state.model_state.hedge_beta = kf.beta

Note on batch vs. online numerical differences
-----------------------------------------------
The batch API (run_pairs_kalman) uses filterpy's full-matrix Kalman
filter internally, while the online class uses manual 2×2 matrix
arithmetic and the Joseph-form covariance update.  These produce
numerically similar — but not bit-identical — results.  The batch
path is the reference; the online path is the hot-path optimised
version.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.kalman.pure import _warm_start_ols
from src.kalman.types import PairsKalmanConfig

_EPS = 1e-12


class PairsKalmanOnline:
    """Online pairs-trading Kalman filter for the backtest hot path.

    State model (random walk on both α and β):
      [α_{t+1}] = [1 0] [α_t] + w_t     w_t ~ N(0, Q)
      [β_{t+1}]   [0 1] [β_t]
    Observation:
      log(P1_t) = α_t + β_t · log(P2_t) + v_t    v_t ~ N(0, R)

    The standardized innovation t_stat = spread / √S is the complete
    trading signal — no EWMA post-processing needed.  The intercept α
    in the state makes the innovation genuinely mean-zero by construction.

    Warm-start via ``init_from_ols()`` uses a batch OLS over the initial
    observations to seed [α₀, β₀] and P₀ — no burn-in delay.

    All hyper-parameters are sourced from a single PairsKalmanConfig
    dataclass, the canonical source of truth for config.

    Parameters
    ----------
    config : PairsKalmanConfig, optional
        Filter hyper-parameters.  Defaults to PairsKalmanConfig().
    """

    def __init__(
        self,
        config: PairsKalmanConfig | None = None,
    ) -> None:
        if config is None:
            cfg = PairsKalmanConfig()
        else:
            cfg = config
        self._q = cfg.process_noise
        self._r = cfg.measurement_noise
        self._halflife = float(cfg.mean_halflife)
        self._adaptive = cfg.adaptive
        self._vol_window = cfg.vol_window

        # State — 2D: [α, β]
        self._alpha: float = 0.0
        self._beta: float = 0.0
        self._P: np.ndarray = np.eye(2, dtype=float)

        # Innovation (spread) = log(P1) - (α_pred + β_pred · log(P2))
        self._spread: float = 0.0

        # Innovation covariance S = H·P·H' + R
        self._innovation_S: float = 0.0

        # t-statistic = spread / √S — THE trading signal
        self._t_stat: float = 0.0

        # Adaptive R tracking — deque for O(1) pop-left in hot path
        self._returns_log: deque[float] = deque(maxlen=cfg.vol_window)
        self._baseline_vol: float = 0.0

        # Number of steps processed
        self._n: int = 0

    # ------------------------------------------------------------------
    # read-only properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current intercept estimate (dimensionless, log-space)."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Current hedge-ratio estimate (dimensionless, log-space elasticity)."""
        return self._beta

    @property
    def t_stat(self) -> float:
        """Current standardized innovation — THE trading signal.

        spread / √S where S = H·P·H' + R is the Kalman innovation
        covariance.  Naturally downweights observations the filter
        considers noisy.
        """
        return self._t_stat

    @property
    def innovation_S(self) -> float:
        """Current innovation covariance S = H·P·H' + R.

        Large values indicate the filter is uncertain about the current
        observation (either noisy data or the state has drifted).
        """
        return self._innovation_S

    @property
    def spread(self) -> float:
        """Current raw spread (Kalman innovation)."""
        return self._spread

    @property
    def n_steps(self) -> int:
        """Number of observations processed so far."""
        return self._n

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def init(self, log_p1: float, log_p2: float) -> float:
        """Initialise the filter with a single observation pair.

        Sets α = 0, β = 1.0 with wide initial uncertainty.
        For proper warm-start, use ``init_from_ols()`` instead.

        Parameters
        ----------
        log_p1 : float
            log-price of symbol 1.
        log_p2 : float
            log-price of symbol 2.

        Returns
        -------
        float
            Initial t_stat for this single observation.
        """
        self._alpha = 0.0
        self._beta = 1.0
        self._P = np.eye(2, dtype=float)

        spread = float(log_p1 - self._alpha - self._beta * log_p2)
        self._spread = spread

        # Innovation covariance: S = H·P·H' + R
        H = np.array([[1.0, log_p2]])
        S = float((H @ self._P @ H.T)[0, 0] + self._r)
        self._innovation_S = S

        if S > _EPS:
            self._t_stat = float(spread / np.sqrt(S))
        else:
            self._t_stat = 0.0

        # Seed adaptive returns tracker
        self._returns_log.clear()

        self._n = 1
        return self._t_stat

    def init_from_ols(self, log_p1_arr: np.ndarray, log_p2_arr: np.ndarray) -> float:
        """Warm-start from batch OLS over an initial window.

        Uses ``_warm_start_ols`` to estimate [α₀, β₀] and P₀, then
        seeds the filter state.  This eliminates burn-in — the filter
        is signalling-usable immediately.

        Parameters
        ----------
        log_p1_arr : np.ndarray
            Log-prices of symbol 1 (initial window).
        log_p2_arr : np.ndarray
            Log-prices of symbol 2 (initial window).

        Returns
        -------
        float
            Initial t_stat for the last observation in the window.
        """
        window = int(self._halflife)
        if window < 3:
            window = 3

        alpha0, beta0, P0 = _warm_start_ols(log_p1_arr, log_p2_arr, window)
        self._alpha = alpha0
        self._beta = beta0
        self._P = P0

        # Compute spread, S, and t_stat for the last observation
        last_lp1 = float(log_p1_arr[-1])
        last_lp2 = float(log_p2_arr[-1])
        spread = float(last_lp1 - alpha0 - beta0 * last_lp2)
        self._spread = spread

        H = np.array([[1.0, last_lp2]])
        S = float((H @ P0 @ H.T)[0, 0] + self._r)
        self._innovation_S = float(max(S, _EPS))

        if S > _EPS:
            self._t_stat = float(spread / np.sqrt(S))
        else:
            self._t_stat = 0.0

        self._returns_log.clear()
        self._n = len(log_p1_arr)
        return self._t_stat

    def update(self, log_p1: float, log_p2: float) -> float:
        """Process one observation and return the updated t_stat.

        Performs a standard Kalman predict-update cycle on the 2D
        [α, β] state, computes the innovation covariance S and the
        standardized innovation t_stat = spread / √S.

        Adaptive R (when enabled) is applied *before* predict, so the
        adaptive scale affects both the Kalman update and the
        diagnostics.

        Parameters
        ----------
        log_p1 : float
            log-price of symbol 1.
        log_p2 : float
            log-price of symbol 2.

        Returns
        -------
        float
            Updated t_stat (the trading signal).
        """
        if self._n == 0:
            return self.init(log_p1, log_p2)

        # ----------------------------------------------------------
        # Adaptive measurement noise (applied BEFORE predict/update
        # so it affects both state update and diagnostics)
        # ----------------------------------------------------------
        r_eff = self._r
        if self._adaptive and self._n > 1:
            # Track log-return of P2: diff of log-prices
            if self._returns_log:
                prev_logp2 = self._returns_log[-1]
                ret = float(log_p2 - prev_logp2)
            else:
                ret = 0.0
            self._returns_log.append(ret)
            if len(self._returns_log) >= self._vol_window:
                recent_vol = float(np.std(self._returns_log))
                if self._baseline_vol == 0.0 and recent_vol > 0:
                    self._baseline_vol = recent_vol
                if self._baseline_vol > 0 and recent_vol > 0:
                    scale = (recent_vol / self._baseline_vol) ** 2
                    r_eff = self._r * scale

        # ----------------------------------------------------------
        # Kalman predict
        # ----------------------------------------------------------
        beta_prior = self._beta
        alpha_prior = self._alpha
        P_prior = self._P + np.eye(2, dtype=float) * self._q  # 2×2

        # ----------------------------------------------------------
        # Kalman update
        # ----------------------------------------------------------
        H = np.array([[1.0, log_p2]], dtype=float)  # 1×2
        y = float(log_p1 - (alpha_prior + beta_prior * log_p2))  # innovation

        # Innovation covariance S = H·P·H' + R
        S = float((H @ P_prior @ H.T)[0, 0] + r_eff)

        # Kalman gain: K = P_prior @ H' / S  (2×1)
        if S > _EPS:
            K = (P_prior @ H.T) / S  # (2×2) @ (2×1) / scalar → (2×1)
        else:
            K = np.zeros((2, 1), dtype=float)

        # State update: x_post = x_prior + K * y
        self._alpha = float(alpha_prior + K[0, 0] * y)
        self._beta = float(beta_prior + K[1, 0] * y)

        # Joseph-form P update (correct for 2D, not the scalar shortcut)
        I = np.eye(2, dtype=float)
        P_post = (I - K @ H) @ P_prior @ (I - K @ H).T + K @ K.T * r_eff
        # Force symmetry for numerical stability
        self._P = np.asarray((P_post + P_post.T) / 2.0, dtype=float)

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------
        self._spread = y
        self._innovation_S = float(max(S, _EPS))

        if S > _EPS:
            self._t_stat = float(y / np.sqrt(S))
        else:
            self._t_stat = 0.0

        self._n += 1
        return self._t_stat
