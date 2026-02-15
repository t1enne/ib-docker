"""StrategyModel - Composite model facade for trading strategies.

Provides unified access to sub-models and historical data from within strategies.

Usage from strategy:
    - self.model.z_score         # Current z-score
    - self.model.market_data[-14:]  # Last 14 bars of OHLCV
    - self.model.current_regime  # Current HMM regime (if configured)
    - self.model.hmm            # HMM model (if configured)
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from src.bt.types import Tick
from src.bt.models.z_model import ZModel
from src.bt.models.market_data import MarketDataView
from src.bt.models.regime_model import RegimeModel
from src.hmm.hmm import MarketRegimeHMM


class StrategyModel:
    """Composite model that strategies access as self.model.

    Composes multiple sub-models (ZModel, RegimeModel) and provides
    unified access to computed features and historical market data.

    Usage:
        class MyStrategy:
            def __init__(self, symbols, params, model: StrategyModel):
                self.model = model

            def on_tick(self, tick, open_trade):
                z = self.model.z_score
                regime = self.model.current_regime
                ema_9 = ema(self.model.market_data[-14:].close, 9)
                # ...
    """

    def __init__(
        self,
        symbols: List[str],
        rolling_window_size: int,
        hmm_floating_window: Optional[int] = None,
        hmm_retrain_interval: Optional[int] = None,
    ):
        """Initialize the strategy model.

        Args:
            symbols: List of trading symbols
            rolling_window_size: Window size for z-score calculation
            hmm_floating_window: Lookback window for HMM training (None = HMM disabled)
            hmm_retrain_interval: Retrain HMM every N bars (default 50 if hmm enabled)
        """
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size

        # Sub-models
        self._z = ZModel(symbols, rolling_window_size)
        self._market_data = MarketDataView(symbols)

        # Z-score state
        self._current_z: float = 0.0
        self._price_buffers: List[dict[str, float]] = []

        # HMM regime model config
        self._hmm_enabled = hmm_floating_window is not None
        self._hmm_floating_window = hmm_floating_window or 252
        self._hmm_retrain_interval = hmm_retrain_interval or 50

        # HMM state
        self._hmm: Optional[MarketRegimeHMM] = None
        self._regime: Optional[RegimeModel] = None
        self._bars_since_last_hmm_train: int = 0
        self._current_regime: Optional[int] = None

    @property
    def z(self) -> ZModel:
        """Access to the ZModel (z-score calculator)."""
        return self._z

    @property
    def hmm(self) -> Optional[RegimeModel]:
        """Access to the HMM regime model (None until first fit)."""
        return self._regime

    @property
    def z_score(self) -> float:
        """Current z-score value (convenience property)."""
        return self._current_z

    @property
    def current_regime(self) -> Optional[int]:
        """Current regime label (0=Low Vol, 1=Med Vol, 2=High Vol).

        Returns None if:
        - HMM not enabled (hmm_floating_window not set)
        - Insufficient data to fit/predict regime
        """
        return self._current_regime

    @property
    def market_data(self) -> MarketDataView:
        """Historical OHLCV data view."""
        return self._market_data

    def update(self, timestamp: pd.Timestamp, tick_group: dict[str, Tick]) -> None:
        """Update the model with new market data.

        Called by the engine each time all symbols have ticked for a timestamp.
        Handles z-score update and HMM training/prediction.

        Args:
            timestamp: Current timestamp
            tick_group: Dict mapping symbol -> Tick for this timestamp
        """
        # Update market data view
        self._market_data.append(timestamp, tick_group)

        # Build price dict for z-score calculation
        prices = {symbol: tick.close for symbol, tick in tick_group.items()}
        self._price_buffers.append(prices)

        # Maintain rolling buffer for z-score
        if len(self._price_buffers) > self.rolling_window_size:
            self._price_buffers = self._price_buffers[-self.rolling_window_size :]

        # Update z-score if we have enough data
        if len(self._price_buffers) >= 2:
            self._current_z = self._z.calculate_z(self._price_buffers)

        # Handle HMM training and prediction
        if self._hmm_enabled:
            self._update_hmm()

    def _update_hmm(self) -> None:
        """Update HMM: train/retrain if needed, then predict."""
        self._bars_since_last_hmm_train += 1
        n_bars = len(self._market_data)

        needs_initial_fit = self._hmm is None and n_bars >= self._hmm_floating_window
        needs_retrain = (
            self._hmm is not None
            and self._bars_since_last_hmm_train >= self._hmm_retrain_interval
            and n_bars >= self._hmm_floating_window
        )

        if needs_initial_fit or needs_retrain:
            self._fit_hmm()

        if self._regime is not None:
            self._predict_regime()

    def _fit_hmm(self) -> None:
        """Train or retrain HMM on trailing hmm_floating_window bars."""
        symbol = self.symbols[0]
        prices = self._market_data[-self._hmm_floating_window :].for_symbol(symbol)[
            "close"
        ]

        hmm_model = MarketRegimeHMM(
            min_train_size=min(self._hmm_floating_window, len(prices)),
        )
        hmm_model.fit(pd.Series(prices))

        self._hmm = hmm_model
        self._regime = RegimeModel(hmm_model)
        self._bars_since_last_hmm_train = 0

    def _predict_regime(self) -> None:
        """Update current regime prediction using fitted HMM."""
        if not self._regime:
            self._current_regime = None
            return None

        try:
            symbol = self.symbols[0]
            prices = self._market_data[-self._hmm_floating_window :].for_symbol(symbol)[
                "close"
            ]
            self._current_regime = self._regime.get_current_regime(pd.Series(prices))
        except IndexError, ValueError:
            self._current_regime = None

    def get_price_buffers(self) -> List[dict[str, float]]:
        """Get current price buffers (for internal use/debugging).

        Returns:
            List of price dicts in the rolling window
        """
        return list(self._price_buffers)

    def get_regime_probability(self) -> Optional[np.ndarray]:
        """Get regime probabilities for the current state.

        Returns:
            Array of probabilities for each regime, or None if:
            - HMM not enabled or not yet fitted
            - Insufficient data
        """
        if self._regime is None:
            return None

        try:
            symbol = self.symbols[0]
            prices = self._market_data[-self._hmm_floating_window :].for_symbol(symbol)[
                "close"
            ]
            return self._regime.get_regime_probability(pd.Series(prices))
        except IndexError, ValueError:
            return None

    def should_trade(
        self,
        confidence_threshold: float = 0.7,
        avoid_regimes: Optional[List[int]] = None,
    ) -> bool:
        """Check if trading should be allowed based on current regime.

        By default avoids trading in high volatility regime (regime 2).

        Args:
            confidence_threshold: Minimum probability for regime confidence
            avoid_regimes: List of regime IDs to avoid (default: [2])

        Returns:
            True if trading is allowed
        """
        if self._regime is None:
            return True  # No HMM, allow all trading

        if avoid_regimes is None:
            avoid_regimes = [2]

        try:
            symbol = self.symbols[0]
            prices = self._market_data[-self._hmm_floating_window :].for_symbol(symbol)[
                "close"
            ]

            return self._regime.should_trade(
                pd.Series(prices),
                confidence_threshold=confidence_threshold,
                avoid_regimes=avoid_regimes,
            )
        except IndexError, ValueError:
            return True  # On error, be permissive
