"""Market data view for strategies.

Provides accumulated OHLCV history with sliceable access for indicator calculations.
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from src.bt.types import Candle


class MarketDataView:
    """Accumulated OHLCV history, sliceable from strategies.

    Usage:
        # Get last 14 bars of close prices
        closes = self.model.market_data[-14:].close

        # Get OHLCV for a specific symbol
        aapl_data = self.model.market_data.for_symbol("AAPL")

        # Apply indicator
        from src.bt.indicators import ema
        ema_9 = ema(self.model.market_data[-14:].close["AAPL"], 9)
    """

    def __init__(self, symbols: List[str]):
        self._symbols = symbols
        self._bars: List[Dict[str, Candle]] = []  # [{sym1: Tick, sym2: Tick}, ...]
        self._timestamps: List[pd.Timestamp] = []

    def append(self, tick: Candle) -> None:
        """Add a new tick to the view.

        Args:
            tick: Tick for a single symbol
        """
        if self._timestamps and self._timestamps[-1] == tick.timestamp:
            self._bars[-1][tick.symbol] = tick
            return

        self._bars.append({tick.symbol: tick})
        self._timestamps.append(tick.timestamp)

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, key: int | slice) -> "MarketDataView":
        """Slice the market data view.

        Args:
            key: Index or slice (supports negative indexing)

        Returns:
            A new MarketDataView containing only the sliced data
        """
        if isinstance(key, int):
            if key < 0:
                key = len(self._bars) + key
            if key < 0 or key >= len(self._bars):
                raise IndexError(f"Index {key} out of range for {len(self._bars)} bars")
            # Return a view with just this single bar
            new_view = MarketDataView(self._symbols)
            new_view._bars = [self._bars[key]]
            new_view._timestamps = [self._timestamps[key]]
            return new_view
        elif isinstance(key, slice):
            # Handle slice (supports negative indices)
            indices = range(len(self._bars))[key]
            new_view = MarketDataView(self._symbols)
            new_view._bars = [self._bars[i] for i in indices]
            new_view._timestamps = [self._timestamps[i] for i in indices]
            return new_view
        else:
            raise TypeError(f"Key must be int or slice, got {type(key)}")

    @property
    def open(self) -> pd.DataFrame:
        """Open prices for all symbols (columns=symbols, index=timestamps)."""
        return self._get_ohlcv_field("open")

    @property
    def high(self) -> pd.DataFrame:
        """High prices for all symbols (columns=symbols, index=timestamps)."""
        return self._get_ohlcv_field("high")

    @property
    def low(self) -> pd.DataFrame:
        """Low prices for all symbols (columns=symbols, index=timestamps)."""
        return self._get_ohlcv_field("low")

    @property
    def close(self) -> pd.DataFrame:
        """Close prices for all symbols (columns=symbols, index=timestamps)."""
        return self._get_ohlcv_field("close")

    @property
    def volume(self) -> pd.DataFrame:
        """Volume for all symbols (columns=symbols, index=timestamps)."""
        return self._get_ohlcv_field("volume")

    def _get_ohlcv_field(self, field: str) -> pd.DataFrame:
        """Build DataFrame for a specific OHLCV field."""
        if not self._bars:
            return pd.DataFrame(columns=pd.Index(self._symbols))

        data = {symbol: [] for symbol in self._symbols}
        for bar in self._bars:
            for symbol in self._symbols:
                tick = bar.get(symbol)
                if tick:
                    data[symbol].append(getattr(tick, field))
                else:
                    data[symbol].append(np.nan)

        return pd.DataFrame(data, index=pd.Index(self._timestamps))

    def for_symbol(self, symbol: str) -> pd.DataFrame:
        """Get full OHLCV DataFrame for a specific symbol.

        Args:
            symbol: Symbol to get data for

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if not self._bars:
            return pd.DataFrame(
                columns=pd.Index(["open", "high", "low", "close", "volume"])
            )

        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        for bar in self._bars:
            tick = bar.get(symbol)
            if tick:
                data["open"].append(tick.open)
                data["high"].append(tick.high)
                data["low"].append(tick.low)
                data["close"].append(tick.close)
                data["volume"].append(tick.volume)
            else:
                for key in data:
                    data[key].append(np.nan)

        return pd.DataFrame(data, index=pd.Index(self._timestamps))

    def get_timestamps(self) -> List[pd.Timestamp]:
        """Get list of all timestamps in the view."""
        return list(self._timestamps)

    def get_latest(self) -> Dict[str, Candle]:
        """Get the most recent tick map."""
        if not self._bars:
            return {}
        return dict(self._bars[-1])
