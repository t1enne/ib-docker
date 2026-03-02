"""StrategyModel - Composite model facade for trading strategies."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.bt.models.market_data import MarketDataView
from src.bt.models.regime_model import RegimeModel
from src.bt.models.z_model import ZModel
from src.bt.types import Tick
from src.hmm.hmm import MarketRegimeHMM


@dataclass
class ZState:
    price_buffers: List[dict[str, float]]
    current_z: float = 0.0


@dataclass
class HMMState:
    enabled: bool
    floating_window: int
    retrain_interval: int
    model: Optional[MarketRegimeHMM] = None
    regime_model: Optional[RegimeModel] = None
    bars_since_last_train: int = 0
    current_regime: Optional[int] = None


class StrategyModel:
    """Composite model that strategies access as self.model."""

    def __init__(
        self,
        symbols: List[str],
        rolling_window_size: int,
        hmm_floating_window: Optional[int] = None,
        hmm_retrain_interval: Optional[int] = None,
    ):
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size

        self._z = ZModel(symbols, rolling_window_size)
        self._market_data = MarketDataView(symbols)

        self._z_state = ZState(price_buffers=[])

        hmm_enabled = hmm_floating_window is not None
        self._hmm_state = HMMState(
            enabled=hmm_enabled,
            floating_window=hmm_floating_window or 252,
            retrain_interval=hmm_retrain_interval or 50,
        )

    @property
    def z(self) -> ZModel:
        return self._z

    @property
    def hmm(self) -> Optional[RegimeModel]:
        return self._hmm_state.regime_model

    @property
    def z_score(self) -> float:
        return self._z_state.current_z

    @property
    def hedge_beta(self) -> float:
        return self._z.beta

    @property
    def current_regime(self) -> Optional[int]:
        return self._hmm_state.current_regime

    @property
    def market_data(self) -> MarketDataView:
        return self._market_data

    def update(self, tick: Tick) -> None:
        self._market_data.append(tick)

        prices = self._prices_from_latest()
        if prices is not None:
            self._update_z_state(prices)

        if self._hmm_state.enabled:
            self._update_hmm_state()

    def _prices_from_latest(self) -> Optional[dict[str, float]]:
        latest = self._market_data.get_latest()
        if len(latest) != len(self.symbols):
            return None
        prices = {symbol: latest[symbol].close for symbol in self.symbols}
        if len(prices) != len(self.symbols):
            return None
        return prices

    def _update_z_state(self, prices: dict[str, float]) -> None:
        buffers = self._z_state.price_buffers
        buffers.append(prices)

        if len(buffers) > self.rolling_window_size:
            del buffers[: len(buffers) - self.rolling_window_size]

        if len(buffers) >= 2:
            self._z_state.current_z = self._z.calculate_z(buffers)

    def _update_hmm_state(self) -> None:
        state = self._hmm_state
        state.bars_since_last_train += 1
        n_bars = len(self._market_data)

        if self._needs_hmm_fit(state, n_bars):
            self._fit_hmm(state)

        if state.regime_model is not None:
            self._predict_regime(state)

    def _needs_hmm_fit(self, state: HMMState, n_bars: int) -> bool:
        needs_initial_fit = state.model is None and n_bars >= state.floating_window
        needs_retrain = (
            state.model is not None
            and state.bars_since_last_train >= state.retrain_interval
            and n_bars >= state.floating_window
        )
        return needs_initial_fit or needs_retrain

    def _fit_hmm(self, state: HMMState) -> None:
        prices = self._get_hmm_prices(state.floating_window)
        hmm_model = MarketRegimeHMM(
            min_train_size=min(state.floating_window, len(prices)),
        )
        hmm_model.fit(prices)

        state.model = hmm_model
        state.regime_model = RegimeModel(hmm_model)
        state.bars_since_last_train = 0

    def _predict_regime(self, state: HMMState) -> None:
        assert state.regime_model
        try:
            prices = self._get_hmm_prices(state.floating_window)
            state.current_regime = state.regime_model.get_current_regime(prices)
        except IndexError, ValueError:
            state.current_regime = None

    def _get_hmm_prices(self, window: int) -> pd.Series:
        assert self.symbols, "At least one symbol is required for HMM"
        symbol = self.symbols[0]
        closes = self._market_data[-window:].for_symbol(symbol)["close"]
        return pd.Series(closes)

    def get_price_buffers(self) -> List[dict[str, float]]:
        return list(self._z_state.price_buffers)

    def get_regime_probability(self) -> Optional[np.ndarray]:
        state = self._hmm_state
        if state.regime_model is None:
            return None

        try:
            prices = self._get_hmm_prices(state.floating_window)
            return state.regime_model.get_regime_probability(prices)
        except IndexError, ValueError:
            return None

    def should_trade(
        self,
        confidence_threshold: float = 0.7,
        avoid_regimes: Optional[List[int]] = None,
    ) -> bool:
        state = self._hmm_state
        if state.regime_model is None:
            return True

        if avoid_regimes is None:
            avoid_regimes = [2]

        try:
            prices = self._get_hmm_prices(state.floating_window)
            return state.regime_model.should_trade(
                prices,
                confidence_threshold=confidence_threshold,
                avoid_regimes=avoid_regimes,
            )
        except IndexError, ValueError:
            return True
