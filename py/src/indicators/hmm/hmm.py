"""
Hidden Markov Model for market regime detection.

This module provides HMM-based regime detection for financial time series,
designed to identify low, medium, and high volatility regimes.
"""

from src.indicators.hmm.types import RegimeStats

import pickle
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm


class MarketRegimeHMM:
    """
    Hidden Markov Model for detecting market regimes based on return characteristics.

    Uses three features:
    - Log returns (current period)
    - Rolling volatility (standard deviation of returns)
    - Rolling momentum (mean return over shorter window)

    Identifies regimes:
    - Regime 0: Low volatility (mean-reversion friendly)
    - Regime 1: Medium volatility (normal market conditions)
    - Regime 2: High volatility (high uncertainty, avoid trading)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_window: int = 20,
        momentum_window: int = 10,
        min_train_size: int = 252,
        update_interval: int = 50,
        random_state: int = 42,
        bars_per_year: int = 252,
    ):
        """
        Initialize HMM model for regime detection.

        Args:
            n_regimes: Number of hidden states (2 or 3 recommended)
            vol_window: Window size for volatility calculation
            momentum_window: Window size for momentum calculation
            min_train_size: Minimum observations needed for initial training
            update_interval: Retrain model every N observations
            random_state: Random seed for reproducibility
            bars_per_year: Average number of bars in a trading year, used to
                annualise return volatility and momentum features. Defaults to
                252 (daily bars). For intraday bars pass e.g. ~3780 for 1h bars
                (15 trading hours/day * 252) so the features annualise correctly
                and the model separates regimes by volatility rather than by
                raw return sign.
        """
        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.min_train_size = min_train_size
        self.update_interval = update_interval
        self.random_state = random_state
        self.bars_per_year = bars_per_year

        self.model: Optional[hmm.GaussianHMM] = None
        self.fitted = False
        self.last_train_idx = 0
        self.regime_labels = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}

    def _create_features(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix from price series.

        Features:
        1. Log returns (r_t = log(P_t / P_{t-1}))
        2. Rolling volatility (std of returns over vol_window)
        3. Rolling momentum (mean return over momentum_window)

        Args:
            prices: Price series (Close prices)
            returns: Optional pre-computed log returns

        Returns:
            DataFrame with columns: returns, volatility, momentum
        """
        if returns is None:
            returns = np.log(prices / prices.shift(1))

        # Rolling volatility (annualised to the bar's year-length).
        # Squart-root scaling so the vol and momentum features are comparable
        # across bar sizes and the model can separate regimes by volatility.
        volatility = returns.rolling(window=self.vol_window).std() * np.sqrt(
            self.bars_per_year
        )

        # Rolling momentum (annualised mean return)
        momentum = (
            returns.rolling(window=self.momentum_window).mean() * self.bars_per_year
        )

        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
            }
        )

        return features.dropna()

    def fit(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> "MarketRegimeHMM":
        """
        Fit HMM model on price data.

        Args:
            prices: Price series
            returns: Optional pre-computed log returns

        Returns:
            self for method chaining
        """
        features = self._create_features(prices, returns)

        if len(features) < self.min_train_size:
            raise ValueError(
                f"Insufficient data: {len(features)} observations, "
                f"minimum required: {self.min_train_size}"
            )

        # Initialize Gaussian HMM with diagonal covariance for better convergence
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_iter=200,
            tol=1e-3,
        )

        # Fit model (suppress hmmlearn Cython stderr convergence noise)
        import sys
        import io

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(features.values)
        finally:
            sys.stderr = old_stderr

        # Train a secondary model to figure out which HMM state = which vol regime.
        # We remap the raw HMM state numbers (arbitrary component order) into
        # semantically meaningful labels: 0=lowest vol, 1=med vol, 2=highest vol.
        self._build_regime_remap()

        self.fitted = True
        self.last_train_idx = len(features)

        return self

    def _build_regime_remap(self) -> None:
        """Build a state→ranked-regime mapping sorted by fitted return variance.

        HMM components are numbered arbitrarily (unidentifiable gauge), so the
        raw state index is meaningless on its own. This builds a deterministic
        bijective remap that canonicalises the permutation: each component is
        sorted by the *fitted* variance of its return emission (covars_[·, 0]),
        a fitted model parameter that is stable across refits and does not
        depend on how many points happen to be assigned to a state.

        Mapping: raw_state → [0=lowest return-var, ..., N-1=highest return-var].
        Because the sort key is a fitted parameter (not a noisy empirical mean
        over predict assignments), the semantic ordering is stable across refits
        even when raw component labels flip. All predict/predict_proba outputs
        are run through this remap.
        """
        assert self.model is not None
        self._state_to_regime, self._regime_to_state = rank_states_by_vol(
            self.model, self.n_regimes
        )

        label_map = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
        self.regime_labels = {
            i: label_map.get(i, f"Regime {i}") for i in range(self.n_regimes)
        }

    def predict(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict regime for each time point.

        Uses the fitted model to predict regimes for all data points.
        First min_train_size observations will be NaN as they were used for training.

        Args:
            prices: Price series
            returns: Optional pre-computed log returns

        Returns:
            Series with regime labels (0, 1, 2)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self._create_features(prices, returns)

        # Predict on all features using the fitted model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self.model
            regimes = self.model.predict(features.values)

        # Remap raw HMM states to volatility-ranked regime labels
        regimes = np.array([self._state_to_regime.get(s, s) for s in regimes])

        regimes_series = pd.Series(regimes, index=features.index)

        # Set first min_train_size observations to NaN (they were used for training)
        if len(regimes_series) >= self.min_train_size:
            regimes_series.iloc[: self.min_train_size] = np.nan

        return regimes_series

    def predict_proba(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Predict regime probabilities for each time point.

        Uses the fitted model to predict probabilities for all data points.

        Args:
            prices: Price series
            returns: Optional pre-computed log returns

        Returns:
            DataFrame with regime probabilities (columns: regime_0, regime_1, regime_2)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self._create_features(prices, returns)

        # Predict probabilities on all features using the fitted model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self.model
            probabilities = self.model.predict_proba(features.values)

        # Reorder probability columns to match volatility-ranked regime order
        remapped_probs = np.zeros_like(probabilities)
        for raw_state, ranked_regime in self._state_to_regime.items():
            remapped_probs[:, ranked_regime] = probabilities[:, raw_state]

        columns = pd.Index([f"regime_{i}" for i in range(self.n_regimes)])
        proba_df = pd.DataFrame(
            remapped_probs,
            index=features.index,
            columns=columns,
        )

        return proba_df

    def get_regime_statistics(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> RegimeStats:
        """
        Calculate statistics for each detected regime.

        Returns:
            Dictionary with regime statistics including:
            - mean_return: Average return per regime
            - volatility: Average volatility per regime
            - frequency: Percentage of time in each regime
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating statistics")

        features = self._create_features(prices, returns)
        regimes = self.predict(prices, returns)

        stats = RegimeStats(n_regimes=self.n_regimes)

        total_obs = len(regimes.dropna())

        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_features = features[mask]

            if len(regime_features) > 0:
                stats.mean_return[regime] = regime_features["returns"].mean() * 252
                stats.volatility[regime] = regime_features["volatility"].mean()
                stats.frequency[regime] = mask.sum() / total_obs
            else:
                stats.mean_return[regime] = np.nan
                stats.volatility[regime] = np.nan
                stats.frequency[regime] = 0.0

        return stats

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the regime transition matrix.

        Returns:
            DataFrame where element [i, j] is P(regime=j | regime=i)
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before getting transition matrix")

        transmat = self.model.transmat_

        return pd.DataFrame(
            transmat,
            index=pd.Index(
                [f"from_{self.regime_labels.get(i, i)}" for i in range(self.n_regimes)]
            ),
            columns=pd.Index(
                [f"to_{self.regime_labels.get(j, j)}" for j in range(self.n_regimes)]
            ),
        )

    def should_trade(self, features: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Determine if trading should occur based on current regime.

        Args:
            features: Feature vector [returns, volatility, momentum]
            threshold: Minimum probability threshold for confidence

        Returns:
            True if trading should occur (not in high volatility regime)
        """
        if not self.fitted:
            return False

        features_reshaped = features.reshape(1, -1)
        assert self.model
        raw_proba = self.model.predict_proba(features_reshaped)[0]
        raw_regime = self.model.predict(features_reshaped)[0]
        ranked_regime = self._state_to_regime.get(raw_regime, raw_regime)

        # Remap probabilities to ranked order
        remapped_proba = np.zeros(self.n_regimes)
        for raw_s, ranked_r in self._state_to_regime.items():
            remapped_proba[ranked_r] = raw_proba[raw_s]

        high_vol_regime = self.n_regimes - 1  # highest vol = last regime
        return (
            ranked_regime != high_vol_regime
            and remapped_proba[ranked_regime] >= threshold
        )

    def save(self, filepath: str) -> None:
        """
        Save fitted model to disk.

        Args:
            filepath: Path to save model (should end in .pkl)
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted model")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model and parameters
        model_data = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "vol_window": self.vol_window,
            "momentum_window": self.momentum_window,
            "min_train_size": self.min_train_size,
            "update_interval": self.update_interval,
            "random_state": self.random_state,
            "fitted": self.fitted,
            "last_train_idx": self.last_train_idx,
            "regime_labels": self.regime_labels,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)


def rank_states_by_vol(
    model, n_regimes: int, vol_feature_idx: int = 0
) -> tuple[dict[int, int], dict[int, int]]:
    """Canonicalise HMM component order into a deterministic vol ranking.

    HMM components are unidentifiable: refitting can permute the raw state
    indices without changing the fitted distribution. The training-window
    practice of sorting states by the empirical mean ``volatility`` over
    ``predict()`` assignments is unstable — a dominant state grabs most points
    so the others get tiny, noisy vol estimates, and their order can flip
    between refits, making ``current_vol``/``current_regime`` jump spuriously.

    Instead we anchor the rank to the *fitted* emission variance of the return
    feature (``covars_[s, vol_feature_idx]``), a fitted model parameter that is
    deterministic per fit and does not depend on assignment counts. Sorting
    states by it fixes the component permutation gauge, so ``raw_state → rank``
    (0 = lowest return-var, N-1 = highest) stays semantically stable across
    refits whenever the underlying fit is stable.

    Parameters
    ----------
    model : hmmlearn GaussianHMM
        Fitted model (diagonal covariance).
    n_regimes : int
        Number of hidden states.
    vol_feature_idx : int
        Column index of the return feature in the emission covariance. The
        feature vector is [returns, volatility, momentum], so index 0. Return
        variance is the canonical volatility anchor (a parameter, not a sample
        statistic).

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        (state_to_regime, regime_to_state) — inverse bijective remaps.
    """
    covars = np.asarray(model.covars_, dtype=float)
    var_by_state: dict[int, float] = {}
    for s in range(n_regimes):
        if covars.ndim == 3:
            # hmmlearn stores `diag` covariances as an (n_components,
            # n_features, n_features) stack of diagonal matrices — the
            # variance of feature *j* for state s is covars[s, j, j].
            var_by_state[s] = float(covars[s, vol_feature_idx, vol_feature_idx])
        elif covars.ndim == 2:
            # Dot-product style: (n_components, n_features) row profile.
            var_by_state[s] = float(covars[s, vol_feature_idx])
        else:
            # Scalar emission variance per component.
            var_by_state[s] = float(covars[s])

    # Lowest return-variance → regime 0, highest → regime N-1. Ties broken by
    # state index for a fully deterministic result.
    sorted_states = sorted(var_by_state, key=lambda s: (var_by_state[s], s))
    state_to_regime = {old: new for new, old in enumerate(sorted_states)}
    regime_to_state = {new: old for new, old in enumerate(sorted_states)}
    return state_to_regime, regime_to_state


def create_regime_features(
    prices: pd.Series,
    vol_window: int = 20,
    momentum_window: int = 10,
) -> pd.DataFrame:
    """
    Standalone function to create regime features from price data.

    Useful for feature generation without fitting HMM.

    Args:
        prices: Price series
        vol_window: Window for volatility calculation
        momentum_window: Window for momentum calculation

    Returns:
        DataFrame with returns, volatility, and momentum columns
    """
    returns = np.log(prices / prices.shift(1))
    volatility = returns.rolling(window=vol_window).std() * np.sqrt(252)
    momentum = returns.rolling(window=momentum_window).mean() * 252

    features = pd.DataFrame(
        {
            "returns": returns,
            "volatility": volatility,
            "momentum": momentum,
        }
    )

    return features.dropna()
