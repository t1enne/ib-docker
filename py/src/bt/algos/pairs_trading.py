import asyncio
import pandas as pd
from typing import List
from collections import defaultdict
from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import Tick, TradeSignal
from src.utils import calculate_zscore_spread


class PairsTradingStrategy(BasePairsStrategy):
    """Pairs trading strategy with z-score recalc at intervals."""

    def __init__(
        self,
        symbols: List[str],
        entry_z: float,
        stop_loss: float,
        take_profit: float,
        rolling_window_size: int,
    ):
        super().__init__(
            symbols,
            entry_threshold=entry_z,
            exit_threshold=stop_loss,
            rolling_window_size=rolling_window_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        # Tick counter for retraining (if needed)
        self.tick_count = 0

    async def process_data(
        self, ticks_queue: asyncio.Queue[Tick], order_queue: asyncio.Queue
    ):
        """Process data from signal_queue and put signals into order_queue."""
        while True:
            tick = await ticks_queue.get()
            if tick is None:
                await order_queue.put(None)
                break

            symbol = tick.symbol
            timestamp = tick.timestamp
            close = tick.close

            # Add to historical data
            self.historical_data[symbol].loc[timestamp, "close"] = close
            # Add to pending ticks
            self.pending_ticks[timestamp][symbol] = close

            # If we have data for both symbols at this timestamp, process
            if len(self.pending_ticks[timestamp]) != 2:
                continue
            # Increment tick count (per timestamp with both symbols)
            self.tick_count += 1
            # Calculate z-score
            z_score = self._calculate_zscore(timestamp)
            if z_score is None:
                # Clear pending for this timestamp
                del self.pending_ticks[timestamp]
                continue
                # Generate signal
            self.z_scores[timestamp] = z_score
            signals = self._calculate_signal(timestamp)
            if len(signals) > 0:
                for s in signals:
                    await order_queue.put(s)

        # Process any remaining pending ticks (e.g., last incomplete timestamp)
        for timestamp, symbols in list(self.pending_ticks.items()):
            if len(symbols) == 2:
                # Increment tick count
                self.tick_count += 1
                # Calculate z-score
                z_score = self._calculate_zscore(timestamp)
                if z_score is not None:
                    # Generate signal
                    signals = self._calculate_signal(timestamp)
                    if len(signals) > 0:
                        for s in signals:
                            await order_queue.put(s)
                # Clear
                del self.pending_ticks[timestamp]

    def _calculate_zscore(self, timestamp: pd.Timestamp) -> float | None:
        """Calculate z-score for the current timestamp using OLS on recent data, consistent with spread command."""
        # Use recent closes for z-score calculation, limited to rolling window
        s1_closes = self.historical_data[self.symbols[0]]["close"].tail(self.rolling_window_size)
        s2_closes = self.historical_data[self.symbols[1]]["close"].tail(self.rolling_window_size)
        if len(s1_closes) < 2 or len(s2_closes) < 2:
            return None
        # Calculate z-score series using same method as spread command
        z_scores = calculate_zscore_spread(s1_closes, s2_closes)
        if z_scores.empty:
            return None
        # Return the latest z-score
        return round(z_scores.iloc[-1], 2)

    def _calculate_signal(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """Generate buy/sell/close signal based on z-score."""
        z_score = self._get_z(timestamp)

        # Check for closing position if z-score reverts to neutral or hits take_profit
        has_position = any(pos != 0 for pos in self.positions.values())
        if has_position and (abs(z_score) < self.exit_threshold or
                            (self.take_profit is not None and abs(z_score) > self.take_profit)):
            self.positions = {symbol: 0.0 for symbol in self.symbols}  # Reset positions
            return [
                self._close(self.symbols[0], timestamp),
                self._close(self.symbols[1], timestamp),
            ]

        # Check for opening position if no position and z-score extreme
        if not has_position and abs(z_score) > self.entry_threshold:
            if z_score < -self.entry_threshold:
                self.positions[self.symbols[0]] = 1.0  # Long
                self.positions[self.symbols[1]] = -1.0  # Short
                return [
                    self._long(self.symbols[0], timestamp),
                    self._short(self.symbols[1], timestamp),
                ]
            elif z_score > self.entry_threshold:
                self.positions[self.symbols[0]] = -1.0  # Short
                self.positions[self.symbols[1]] = 1.0  # Long
                return [
                    self._long(self.symbols[1], timestamp),
                    self._short(self.symbols[0], timestamp),
                ]

        return []
