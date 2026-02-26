from typing import List, Optional
import pandas as pd
import numpy as np


class CorrelationModel:
    def __init__(self, symbols: List[str], rolling_window_size: int):
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def calculate_correlation_matrix(
        self, buffers: List[dict]
    ) -> Optional[pd.DataFrame]:
        """Compute rolling correlation matrix from price buffers.

        Args:
            buffers: List of {symbol: price} dicts

        Returns:
            DataFrame with correlation matrix
        """
        if len(buffers) < self.rolling_window_size:
            self._correlation_matrix = None
            return self._correlation_matrix

        price_df = pd.DataFrame(buffers)

        available_symbols = [s for s in self.symbols if s in price_df.columns]
        if len(available_symbols) < 2:
            self._correlation_matrix = None
            return self._correlation_matrix

        price_df = price_df[available_symbols]

        returns = price_df.pct_change().dropna()

        if len(returns) < 20:
            self._correlation_matrix = None
            return self._correlation_matrix

        self._correlation_matrix = returns.corr()
        return self._correlation_matrix

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols.

        Returns 0.0 if correlation matrix is not available.
        """
        if self._correlation_matrix is None:
            return 0.0

        if symbol1 not in self._correlation_matrix.index:
            return 0.0
        if symbol2 not in self._correlation_matrix.columns:
            return 0.0

        return float(self._correlation_matrix.loc[symbol1, symbol2])
