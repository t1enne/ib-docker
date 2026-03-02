# Data feed module
import asyncio
from src.db.models import ISymbol
from src.syncm import get_symbols, sync_data

import pandas as pd
from pandas import Timestamp

from src.bt.types import StrategyConfig, EngineWindow
from src.bt.state import Tick
from src import get_local_candles
from typing import AsyncGenerator, Dict, List, Optional, Tuple, cast


class DataFeed:
    """Async data feed for n symbols from SQLite."""

    def __init__(self, config: StrategyConfig, window: EngineWindow):
        self.symbols = config.symbols
        self.bar = config.bar

    async def load(self, config: StrategyConfig, window: EngineWindow):
        from_date = window.train_start.date()
        # await sync_data(
        #     config.symbols,
        # )
        self.time_series = [
            get_local_candles(x, window.train_start, window.test_end, self.bar)
            for x in config.symbols
        ]
        self.candles_df = pd.concat(self.time_series, axis=1, keys=self.symbols)

    async def get_data_stream(self) -> AsyncGenerator[Optional[Tick]]:
        """Returns an async generator that yields market data ticks for all symbols."""
        data = self.time_series
        if not data or not len(data[0].index):
            raise ValueError("no timestamps")

        has_equal_ts = all(df.index.equals(data[0].index) for df in data[1:])
        if not has_equal_ts:
            print("timestamps not in sync")

        # Get union of all timestamps
        all_timestamps = sorted(set().union(*[set(df.index) for df in data]))

        # Create a lookup for faster existence checking
        # This is more memory efficient than filtering all dataframes
        for timestamp in all_timestamps:
            for j, df in enumerate(data):
                try:
                    row = df.loc[timestamp]
                    yield Tick(
                        timestamp=cast(Timestamp, timestamp),
                        symbol=self.symbols[j],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                except KeyError:
                    # This symbol doesn't have data for this timestamp, skip
                    continue
