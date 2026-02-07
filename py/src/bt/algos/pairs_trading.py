from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.algos.z_model import ZModel, TrainedZModel
from src.bt.types import Tick, TradeSignal, StrategyProtocol


@dataclass
class StrategyParams:
    entry_z: float
    exit_z: float


@dataclass
class PositionState:
    entry_timestamp: pd.Timestamp
    entry_z: float
    side: str
    symbols: Tuple[str, str]
    bars_held: int = 0


class PairsTradingStrategy(StrategyProtocol):
    """Stateless pairs trading strategy that generates signals from trained model."""

    bps: BasePairsStrategy

    def __init__(
        self,
        symbols: List[str],
        # hdata: dict[str, pd.DataFrame],
        strategy_params: StrategyParams,
        rolling_window_size: int = 20,
    ):
        self.bps = BasePairsStrategy(symbols)
        self.params = strategy_params
        self.position: Optional[PositionState] = None
        self.current_model: Optional[TrainedZModel] = None
        self.z_model = ZModel(symbols, rolling_window_size=rolling_window_size)

    def set_model(self, model: TrainedZModel) -> None:
        """Set the trained model from the engine."""
        self.current_model = model

    def get_z_scores(self) -> pd.DataFrame:
        return self.bps.z_scores

    def on_tick(self, tick: Tick) -> List[TradeSignal]:
        """Process tick and generate signals using trained model."""
        symbol = tick.symbol
        timestamp = tick.timestamp
        close = tick.close

        # Add to historical data
        # self.bps.hdata[symbol].loc[timestamp, "Close"] = close
        self.bps.pending_ticks[timestamp][symbol] = close

        # Need both symbols to process
        if len(self.bps.pending_ticks[timestamp]) != 2:
            return []

        # Get current prices
        prices = dict(self.bps.pending_ticks[timestamp])

        # Calculate z-score using trained model
        if self.current_model is None:
            del self.bps.pending_ticks[timestamp]
            return []

        z_score = self.z_model.calculate_z(prices, self.current_model)
        self.bps.z_scores.loc[timestamp, "z"] = z_score

        signals = self._calculate_signal(timestamp, z_score, prices)
        del self.bps.pending_ticks[timestamp]

        return signals

    def _calculate_signal(
        self, timestamp: pd.Timestamp, z_score: float, prices: dict[str, float]
    ) -> List[TradeSignal]:
        """Generate signals based on z-score."""
        sym1, sym2 = self.bps.symbols

        if self.position is not None:
            self.position.bars_held += 1

            exit_signals = self._check_exit_conditions(timestamp, z_score)
            if exit_signals:
                return exit_signals

        if abs(z_score) > self.params.entry_z:
            if z_score < -self.params.entry_z:
                self.position = PositionState(
                    entry_timestamp=timestamp,
                    entry_z=z_score,
                    side="long_spread",
                    symbols=(sym1, sym2),
                    bars_held=0,
                )
                return [
                    self.bps._long(sym1, timestamp),
                    self.bps._short(sym2, timestamp),
                ]
            elif z_score > self.params.entry_z:
                self.position = PositionState(
                    entry_timestamp=timestamp,
                    entry_z=z_score,
                    side="short_spread",
                    symbols=(sym1, sym2),
                    bars_held=0,
                )
                return [
                    self.bps._long(sym2, timestamp),
                    self.bps._short(sym1, timestamp),
                ]

        return []

    def _check_exit_conditions(
        self, timestamp: pd.Timestamp, z_score: float
    ) -> List[TradeSignal]:
        """Check exit conditions based on z-score reversion or time decay."""
        if self.position is None:
            return []

        if self.position.side == "long_spread" and z_score > -self.params.exit_z:
            return self._close_position(timestamp)
        elif self.position.side == "short_spread" and z_score < self.params.exit_z:
            return self._close_position(timestamp)

        # if self.position.bars_held >= self.params.time_decay_bars:
        #     return self._close_position(timestamp)

        return []

    def _close_position(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """Generate close signals for both legs."""
        if self.position is None:
            return []

        sym1, sym2 = self.position.symbols
        self.position = None
        return [
            self.bps._close(sym1, timestamp),
            self.bps._close(sym2, timestamp),
        ]
