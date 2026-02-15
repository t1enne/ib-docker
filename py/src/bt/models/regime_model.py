"""
Regime Model wrapper for HMM-based market regime detection.

Provides a convenient interface for loading and using saved HMM models
in backtesting strategies.
"""

from typing import Optional
import numpy as np
import pandas as pd

from src.hmm.hmm import MarketRegimeHMM, create_regime_features


class RegimeModel:
    """
    Wrapper for MarketRegimeHMM to simplify usage in trading strategies.

    Example:
        >>> hmm = MarketRegimeHMM(min_train_size=252)
        >>> hmm.fit(prices)
        >>> model = RegimeModel(hmm)
        >>> regime = model.get_current_regime(prices)
    """

    def __init__(self, hmm_model: MarketRegimeHMM):
        """
        Initialize with a fitted MarketRegimeHMM instance.

        Args:
            hmm_model: Fitted HMM model
        """
        if not hmm_model.fitted:
            raise ValueError("HMM model must be fitted before use")

        self.hmm = hmm_model
        self.vol_window = hmm_model.vol_window
        self.momentum_window = hmm_model.momentum_window

    def get_current_regime(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> int:
        """
        Get the current regime for given price data.

        Args:
            prices: Recent price series (at least vol_window + momentum_window points)
            returns: Optional pre-computed log returns

        Returns:
            Regime label (0=Low Vol, 1=Med Vol, 2=High Vol)
        """
        regimes = self.hmm.predict(prices, returns)
        return int(regimes.dropna().iloc[-1])

    def get_regime_probability(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Get regime probabilities for the most recent point.

        Args:
            prices: Recent price series
            returns: Optional pre-computed log returns

        Returns:
            Array of probabilities for each regime
        """
        probabilities = self.hmm.predict_proba(prices, returns)
        return probabilities.iloc[-1].values

    def should_trade(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        confidence_threshold: float = 0.7,
        avoid_regimes: Optional[list] = None,
    ) -> bool:
        """
        Determine if trading should occur based on current regime.

        By default, avoids trading in high volatility regime (regime 2).

        Args:
            prices: Recent price series
            returns: Optional pre-computed log returns
            confidence_threshold: Minimum probability for regime confidence
            avoid_regimes: List of regime IDs to avoid (default: [2])

        Returns:
            True if trading is allowed
        """
        if avoid_regimes is None:
            avoid_regimes = [2]

        try:
            regime = self.get_current_regime(prices, returns)
            proba = self.get_regime_probability(prices, returns)
        except IndexError, ValueError:
            return False

        if regime in avoid_regimes:
            return False

        if proba[regime] < confidence_threshold:
            return False

        return True

    def get_regime_features(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculate features used by the model for given prices.

        Args:
            prices: Price series

        Returns:
            DataFrame with returns, volatility, momentum columns
        """
        return create_regime_features(
            prices,
            vol_window=self.vol_window,
            momentum_window=self.momentum_window,
        )

    @property
    def n_regimes(self) -> int:
        """Number of regimes in the model."""
        return self.hmm.n_regimes

    @property
    def regime_labels(self) -> dict:
        """Dictionary mapping regime IDs to labels."""
        return self.hmm.regime_labels

    def get_transition_matrix(self) -> pd.DataFrame:
        """Get the regime transition matrix."""
        return self.hmm.get_transition_matrix()
