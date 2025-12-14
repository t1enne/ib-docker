import asyncio
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from src.bt.types import ActionType, Tick, TradeSignal
from src.utils import calculate_zscore_spread


class PairsTradingStrategy:
    """Pairs trading strategy with z-score recalc at intervals."""

    historical_data: Dict[str, pd.DataFrame]

    def __init__(
        self,
        symbols: List[str],
        entry_z: float,
        stop_loss: float,
        take_profit: float,
        retrain_tick_interval: int,
    ):
        self.symbols = symbols
        self.entry_z = entry_z
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.retrain_tick_interval = retrain_tick_interval
        self.alpha = None
        self.beta = None
        # Buffers for historical data (for retraining)
        self.historical_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        # Pending ticks for current timestamp
        self.pending_ticks = defaultdict(dict)
        # Tick counter for retraining
        self.tick_count = 0
        # Position tracking
        self.has_position = False

    def populate_historical_data(self, data: Dict[str, pd.DataFrame]):
        for symbol in data:
            df = data[symbol]
            self.historical_data[symbol] = pd.DataFrame(
                {"timestamp": df.index, "close": df["Close"]}
            )

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
            self.historical_data[symbol].loc[timestamp] = close
            # Add to pending ticks
            self.pending_ticks[timestamp][symbol] = close

            # If we have data for both symbols at this timestamp, process
            if len(self.pending_ticks[timestamp]) == 2:
                # Increment tick count (per timestamp with both symbols)
                self.tick_count += 1
                # Calculate z-score
                z_score = self._calculate_zscore(timestamp)
                if z_score is not None:
                    # Generate signal
                    signals = self._generate_signal(z_score, timestamp)
                    if len(signals) > 0:
                        for s in signals:
                            await order_queue.put(s)

                # Clear pending for this timestamp
                del self.pending_ticks[timestamp]

        # Process any remaining pending ticks (e.g., last incomplete timestamp)
        for timestamp, symbols in list(self.pending_ticks.items()):
            if len(symbols) == 2:
                # Increment tick count
                self.tick_count += 1
                # Calculate z-score
                z_score = self._calculate_zscore(timestamp)
                if z_score is not None:
                    # Generate signal
                    signals = self._generate_signal(z_score, timestamp)
                    if len(signals) > 0:
                        for s in signals:
                            await order_queue.put(s)
                # Clear
                del self.pending_ticks[timestamp]

    def _calculate_zscore(self, timestamp: pd.Timestamp) -> float | None:
        """Calculate z-score for the current timestamp using OLS on recent data, consistent with spread command."""
        # Use recent closes for z-score calculation
        s1_closes = self.historical_data[self.symbols[0]]["close"]
        s2_closes = self.historical_data[self.symbols[1]]["close"]
        # Calculate z-score series using same method as spread command
        z_scores = calculate_zscore_spread(s1_closes, s2_closes)
        if z_scores.empty:
            return None
        # Return the latest z-score
        return round(z_scores.iloc[-1], 2)

    def _generate_signal(
        self, z_score: float, timestamp: pd.Timestamp
    ) -> List[TradeSignal]:
        """Generate buy/sell/close signal based on z-score."""
        # Check for closing position if z-score reverts to neutral
        if self.has_position and abs(z_score) < 0.5:
            self.has_position = False
            return [
                self._close(self.symbols[0], z_score, timestamp),
                self._close(self.symbols[1], z_score, timestamp),
            ]

        # Check for opening position if no position and z-score extreme
        if not self.has_position and abs(z_score) > self.entry_z:
            self.has_position = True
            if z_score < -self.entry_z:
                return [
                    self._long(self.symbols[0], z_score, timestamp),
                    self._short(self.symbols[1], z_score, timestamp),
                ]
            elif z_score > self.entry_z:
                return [
                    self._long(self.symbols[1], z_score, timestamp),
                    self._short(self.symbols[0], z_score, timestamp),
                ]

        return []

    def _short(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        return TradeSignal(
            action=ActionType.short,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _long(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        return TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _close(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        return TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )
