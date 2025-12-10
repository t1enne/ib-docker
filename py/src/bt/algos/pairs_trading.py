import asyncio
import pandas as pd
from typing import List
from collections import defaultdict, deque
from src.bt.types import ActionType, TradeSignal
from src.utils import get_ols_fit_model


class PairsTradingStrategy:
    """Pairs trading strategy with z-score recalc at intervals."""

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
        self.historical_data = {symbol: [] for symbol in symbols}
        # Buffer for recent prices (for z-score calc, last 100 points)
        self.price_buffer = {symbol: deque(maxlen=100) for symbol in symbols}
        # Pending ticks for current timestamp
        self.pending_ticks = defaultdict(dict)
        # Rolling buffer for spreads to compute dynamic z-score
        self.spread_buffer = deque(maxlen=100)  # Lookback window for rolling stats
        # Tick counter for retraining
        self.tick_count = 0

    async def process_data(
        self, signal_queue: asyncio.Queue, order_queue: asyncio.Queue
    ):
        """Process data from signal_queue and put signals into order_queue."""
        while True:
            tick = await signal_queue.get()
            if tick is None:  # End of data
                await order_queue.put(None)
                break

            symbol = tick.symbol
            timestamp = tick.timestamp
            close = tick.close

            # Add to historical data
            self.historical_data[symbol].append((timestamp, close))
            # Add to price buffer
            self.price_buffer[symbol].append(close)

            # Add to pending ticks
            self.pending_ticks[timestamp][symbol] = close

            # If we have data for both symbols at this timestamp, process
            if len(self.pending_ticks[timestamp]) == 2:
                # Increment tick count (per timestamp with both symbols)
                self.tick_count += 1

                # Check if retraining is needed (every N ticks)
                if self.tick_count % self.retrain_tick_interval == 0:
                    self._retrain_model(timestamp)

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

    def _retrain_model(self, current_timestamp: pd.Timestamp):
        """Retrain the OLS model using recent historical data."""
        # Use last 12 months or 250 trading days for retraining
        s1_data = self.historical_data[self.symbols[0]]
        s2_data = self.historical_data[self.symbols[1]]
        if len(s1_data) < 30 or len(s2_data) < 30:  # Minimum data
            return
        # Extract closes
        s1_closes = [x[1] for x in s1_data]
        s2_closes = [x[1] for x in s2_data]
        if len(s1_closes) != len(s2_closes):
            return  # Mismatched data
        s1_series = pd.Series(s1_closes)
        s2_series = pd.Series(s2_closes)
        if s1_series.empty or s2_series.empty:
            return
        # Fit model
        model = get_ols_fit_model(s1_series, s2_series)
        self.alpha, self.beta = model.params
        # Model is refit, rolling stats will update dynamically

    def _calculate_zscore(self, timestamp: pd.Timestamp) -> float | None:
        """Calculate z-score for the current timestamp using rolling spread statistics."""
        if self.alpha is None:
            return None
        # Use the closes from pending ticks
        s1_close = self.pending_ticks[timestamp][self.symbols[0]]
        s2_close = self.pending_ticks[timestamp][self.symbols[1]]
        # Calculate spread using current model
        spread = s1_close - (self.alpha + self.beta * s2_close)
        # Add to rolling buffer
        self.spread_buffer.append(spread)
        # Need minimum data for rolling stats
        if len(self.spread_buffer) < 30:
            return None
        # Compute rolling mean and std
        spreads = list(self.spread_buffer)
        rolling_mean = sum(spreads) / len(spreads)
        rolling_std = (
            sum((x - rolling_mean) ** 2 for x in spreads) / len(spreads)
        ) ** 0.5
        if rolling_std == 0:
            return None
        z_score = (spread - rolling_mean) / rolling_std
        return round(z_score, 2)

    def _generate_signal(
        self, z_score: float, timestamp: pd.Timestamp
    ) -> List[TradeSignal]:
        """Generate buy/sell signal based on z-score."""
        if abs(z_score) > self.entry_z:
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
