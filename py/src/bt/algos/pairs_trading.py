import pandas as pd
from typing import List
from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import StrategyProtocol, Tick, TradeSignal
from src.utils import calculate_zscore_spread


class PairsTradingStrategy(StrategyProtocol):
    """Pairs trading strategy with z-score recalc at intervals."""

    bps: BasePairsStrategy

    def __init__(
        self,
        symbols: List[str],
        hdata: dict[str, pd.DataFrame],
        entry_z: float,
        rolling_window_size: int,
    ):
        self.bps = BasePairsStrategy(
            symbols,
            hdata,
            entry_threshold=entry_z,
            rolling_window_size=rolling_window_size,
        )
        # Tick counter for retraining (if needed)

    def on_tick(self, tick: Tick) -> List[TradeSignal]:
        """Process data from signal_queue and put signals into order_queue."""
        symbol = tick.symbol
        timestamp = tick.timestamp
        close = tick.close

        # Add to historical data
        self.bps.hdata[symbol].loc[timestamp, "Close"] = close
        # Add to pending ticks
        self.bps.pending_ticks[timestamp][symbol] = close

        # If we have data for both symbols at this timestamp, process
        if len(self.bps.pending_ticks[timestamp]) != 2:
            return []
        # Calculate z-score
        z_score = self._calculate_zscore()
        if z_score is None:
            # Clear pending for this timestamp
            del self.bps.pending_ticks[timestamp]
            return []
        # Generate signal
        self.bps.z_scores[timestamp] = z_score
        signals = self._calculate_signal(timestamp)
        if len(signals) > 0:
            return signals

        return []

    def _calculate_zscore(self) -> float | None:
        """Calculate z-score for the current timestamp using OLS on recent data, consistent with spread command."""
        # Use recent closes for z-score calculation, limited to rolling window
        s1_closes = self.bps.hdata[self.bps.symbols[0]]["Close"]
        s2_closes = self.bps.hdata[self.bps.symbols[1]]["Close"]
        [tail1, tail2] = [
            s.tail(self.bps.rolling_window_size).dropna()
            for s in [s1_closes, s2_closes]
        ]
        if len(tail1) < 2 or len(tail2) < 2:
            return None
        # Calculate z-score series using same method as spread command
        z_scores = calculate_zscore_spread(tail1, tail2)
        if z_scores.empty:
            return None
        # Return the latest z-score
        return round(z_scores.iloc[-1], 2)

    def _calculate_signal(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """Generate buy/sell/close signal based on z-score."""
        z_score = self.bps._get_z(timestamp)

        # Check for opening position if no position and z-score extreme
        if abs(z_score) > self.bps.entry_threshold:
            if z_score < -self.bps.entry_threshold:
                return [
                    self.bps._long(self.bps.symbols[0], timestamp),
                    self.bps._short(self.bps.symbols[1], timestamp),
                ]
            elif z_score > self.bps.entry_threshold:
                return [
                    self.bps._long(self.bps.symbols[1], timestamp),
                    self.bps._short(self.bps.symbols[0], timestamp),
                ]

        return []
