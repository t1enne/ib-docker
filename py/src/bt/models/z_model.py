from typing import List, Tuple
from src.bt.zscore import calculate_rolling_z


class ZModel:
    def __init__(self, symbols: List[str], rolling_window_size: int):
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size
        self._current_beta: float = 1.0

    @property
    def beta(self) -> float:
        return self._current_beta

    def calculate_z(self, buffers: List[dict[str, float]]) -> float:
        """Compute rolling z-score from price buffers using shared calculation."""
        if len(buffers) < 2:
            return 0.0

        sym1, sym2 = self.symbols
        prices1 = [b[sym1] for b in buffers]
        prices2 = [b[sym2] for b in buffers]

        z, _, beta = calculate_rolling_z(prices1, prices2, self.rolling_window_size)
        self._current_beta = beta
        return z

    def calculate_z_by_index(
        self, prices1: List[float], prices2: List[float], window: int
    ) -> float:
        """Calculate z-score for given price lists (matches spread module behavior).

        Returns NaN if insufficient data (less than window points).
        """
        if len(prices1) < window or len(prices2) < window:
            return float("nan")

        prices1_arr = prices1[-window:]
        prices2_arr = prices2[-window:]

        z, _, beta = calculate_rolling_z(prices1_arr, prices2_arr, window)
        self._current_beta = beta
        return z
