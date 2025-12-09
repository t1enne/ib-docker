import asyncio
import pandas as pd
from typing import List, Dict


class PairsTradingStrategy:
    """Pairs trading strategy with z-score recalc at intervals."""

    def __init__(
        self,
        symbols: List[str],
        entry_z: float,
        stop_loss: float,
        take_profit: float,
        retrain_interval_months: int,
        training_start: str,
        training_end: str,
    ):
        self.symbols = symbols
        self.entry_z = entry_z
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.retrain_interval_months = retrain_interval_months
        self.training_start = pd.Timestamp(training_start)
        self.training_end = pd.Timestamp(training_end)
        self.last_retrain = self.training_end
        self.alpha = None
        self.beta = None
        self.mean_spread = None
        self.std_spread = None

    async def process_data(
        self, signal_queue: asyncio.Queue, order_queue: asyncio.Queue
    ):
        """Process data from signal_queue and put signals into order_queue."""
        while True:
            tick = await signal_queue.get()
            if tick is None:  # End of data
                await order_queue.put(None)
                break

            # Check if retraining is needed
            if tick["timestamp"] >= self.last_retrain:
                self._retrain()

            # Calculate z-score
            z_score = self._calculate_zscore(tick)
            if z_score is not None:
                # Generate signal
                signal = self._generate_signal(z_score)
                if signal["action"] != "HOLD":
                    await order_queue.put(signal)

    def _retrain(self):
        """Retrain the model at intervals."""
        # Collect historical data up to last_retrain
        # Note: In a full implementation, buffer data or use a stateful approach
        # For simplicity, assume data is available; in practice, accumulate data
        # Here, we'll simulate with a fixed model for now
        # TODO: Implement proper data accumulation for retraining
        self.alpha = 0.0  # Placeholder
        self.beta = 1.0
        self.mean_spread = 0.0
        self.std_spread = 1.0
        self.last_retrain += pd.offsets.MonthEnd(self.retrain_interval_months)

    def _calculate_zscore(self, tick: Dict) -> float | None:
        """Calculate z-score for the current tick."""
        if self.alpha is None:
            return None
        if tick["symbol"] not in self.symbols[:2]:  # Assume pairs
            return None
        # Simplified: Assume we have close prices
        # In practice, need to track prices for both symbols
        # Placeholder
        spread = tick["close"] - (self.alpha + self.beta * tick["close"])  # Simplified
        z_score = (spread - self.mean_spread) / self.std_spread
        return z_score

    def _generate_signal(self, z_score: float) -> Dict:
        """Generate buy/sell signal based on z-score."""
        if abs(z_score) > self.entry_z:
            if z_score < -self.entry_z:
                return {
                    "action": "BUY",
                    "symbol1": self.symbols[0],
                    "symbol2": self.symbols[1],
                    "z_score": z_score,
                }
            elif z_score > self.entry_z:
                return {
                    "action": "SELL",
                    "symbol1": self.symbols[0],
                    "symbol2": self.symbols[1],
                    "z_score": z_score,
                }
        return {"action": "HOLD"}
