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
        await sync_data(config.symbols, window.train_start)
        self.time_series = [
            get_local_candles(x, window.train_start, window.test_end, self.bar)
            for x in config.symbols
        ]
        self.candles_df = pd.concat(self.time_series, axis=1, keys=self.symbols)

    async def get_data_stream(self) -> AsyncGenerator[Optional[Tick]]:
        """Returns an async generator that yields market data ticks for all symbols."""
        data = self.time_series
        ticks = []
        for i in range(len(data)):
            df = data[i]
            symbol = self.symbols[i]
            for idx, row in df.iterrows():
                ticks.append(
                    Tick(
                        timestamp=cast(Timestamp, idx),
                        symbol=symbol,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                )

        ticks.sort(key=lambda x: x.timestamp)

        for tick in ticks:
            yield tick
