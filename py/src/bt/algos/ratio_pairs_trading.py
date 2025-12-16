import numpy as np
from typing import List
from collections import deque
import pandas as pd

from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import TradeSignal


class RatioPairsTradingStrategy(BasePairsStrategy):
    """
    Pair trading strategy based on price ratio deviations.

    This strategy trades when the ratio of two prices deviates significantly from its mean,
    without requiring regression for hedge ratios. Simpler than spread-based methods.

    Implementation Notes:
    - Compute ratio = log(price1 / price2) for stationarity
    - Maintain buffer of recent ratios
    - Calculate z-score on ratio series
    - Enter when |z| > entry_threshold, exit when |z| < exit_threshold
    - No complex model retraining needed
    """

    def __init__(self, symbols: List[str], **kwargs):
        super().__init__(symbols, **kwargs)
        self.ratio_buffer = deque(maxlen=kwargs.get("buffer_size", 100))

    async def process_data(self, ticks_queue, order_queue):
        """
        Process tick data and compute ratio-based signals.
        """
        while True:
            tick = await ticks_queue.get()
            if tick is None:
                await order_queue.put(None)
                break

            # Implementation: Add tick, compute ratio if both available, buffer it
            pass

    def _calculate_signal(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """
        Generate signals based on ratio z-score.
        """
        # Implementation: Compute z-score on ratio_buffer, generate entry/exit signals
        return []

    def _compute_ratio_zscore(self) -> float:
        """
        Compute z-score of current ratio against historical ratios.

        Implementation:
        - ratios = list(self.ratio_buffer)
        - mean = np.mean(ratios), std = np.std(ratios)
        - current_ratio = ratios[-1]
        - return (current_ratio - mean) / std if std > 0 else 0
        """
        return 0.0
