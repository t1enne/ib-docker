from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import Tick, TradeSignal, StrategyProtocol, Trade


@dataclass
class StrategyParams:
    entry_z: float
    exit_z: float


class PairsTradingStrategy(StrategyProtocol):
    """Stateless pairs trading strategy that generates signals from z-score."""

    bps: BasePairsStrategy

    def __init__(self, symbols: List[str], strategy_params: StrategyParams):
        self.bps = BasePairsStrategy(symbols)
        self.params = strategy_params

    def on_tick(
        self, tick: Tick, z_score: float, open_trade: Optional[Trade]
    ) -> List[TradeSignal]:
        """Process tick with z-score and generate signals."""
        symbol = tick.symbol
        timestamp = tick.timestamp
        close = tick.close

        if open_trade and abs(z_score) < self.params.exit_z:
            # z regression
            return self._calculate_exit_signal(tick, open_trade, z_score)

        self.bps.pending_ticks[timestamp][symbol] = close

        if len(self.bps.pending_ticks[timestamp]) != 2:
            return []

        prices = dict(self.bps.pending_ticks[timestamp])
        del self.bps.pending_ticks[timestamp]

        signals = self._calculate_signal(timestamp, z_score, prices)
        return signals

    def _calculate_signal(
        self, timestamp: pd.Timestamp, z_score: float, prices: dict[str, float]
    ) -> List[TradeSignal]:
        """Generate signals based on z-score."""
        sym1, sym2 = self.bps.symbols

        if z_score < -self.params.entry_z:
            return [
                self.bps._long(sym1, timestamp, prices[sym1], z_score),
                self.bps._short(sym2, timestamp, prices[sym2], z_score),
            ]

        if z_score > self.params.entry_z:
            return [
                self.bps._long(sym2, timestamp, prices[sym2], z_score),
                self.bps._short(sym1, timestamp, prices[sym1], z_score),
            ]

        return []

    def _calculate_exit_signal(self, tick: Tick, trade: Trade, z_score: int | float):
        return [self.bps._close(trade.symbol, tick.timestamp, tick.close, z_score)]
