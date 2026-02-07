import asyncio
import pandas as pd

from typing import List, AsyncGenerator, Tuple, Dict
from src.bt.portfolio.portfolio import Portfolio
from src.utils import read_candles
from src.bt.types import Tick, Trade, StrategyProtocol, PortfolioResult, TradeSignal


class DataFeed:
    """Async data feed for n symbols from SQLite."""

    def __init__(
        self,
        symbols: List[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    async def get_data_stream(self) -> AsyncGenerator[Tick]:
        """Returns an async generator that yields market data ticks for all symbols."""

        # Load data for all symbols

        candles_list = [
            read_candles(x, self.start_date, self.end_date) for x in self.symbols
        ]

        # Combine into a single stream of ticks
        ticks = []
        for i in range(len(candles_list)):
            df = candles_list[i]
            symbol = self.symbols[i]
            for idx, row in df.iterrows():
                ticks.append(
                    Tick(
                        timestamp=idx,
                        symbol=symbol,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                )

        # Sort ticks by timestamp to interleave symbols
        ticks.sort(key=lambda x: x.timestamp)

        # Yield ticks asynchronously
        for tick in ticks:
            yield tick
            # await asyncio.sleep(0.001)


class BTEngine:
    """Main backtesting engine using asyncio."""
    pending_signals:List[TradeSignal] = []
    
    def __init__(
        self, strategy: StrategyProtocol, portfolio: Portfolio, data_feed: DataFeed
    ):
        self.strategy = strategy
        self.portfolio = portfolio
        self.data_feed = data_feed
        # Load data for plotting
        print(
            f"datafeed start: {data_feed.start_date}, datafeed end: {data_feed.end_date}"
        )
        self.data = {
            symbol: read_candles(symbol, data_feed.start_date, data_feed.end_date)
            for symbol in data_feed.symbols
        }

    async def run(self) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame]]:
        """Run the backtest simulation asynchronously."""

        # Create task for data feed
        # data_task = asyncio.create_task(self._run_data_feed(ticks_queue))

        async for tick in self.data_feed.get_data_stream():
            if tick is None:
                break
            
            # Send signals to portfolio
            for signal in self.pending_signals:
                self.portfolio.on_signal(signal)
            # Process tick through strategy
            self.pending_signals = self.strategy.on_tick(tick)
            # Send tick to portfolio for SL/TP
            self.portfolio.on_tick(tick)

        # Finalize results
        return self._finalize_results()

    async def _run_data_feed(self, signal_queue: asyncio.Queue):
        """Run the data feed and put ticks into the signal queue."""
        async for tick in self.data_feed.get_data_stream():
            await signal_queue.put(tick)
        # Signal end of data
        await signal_queue.put(None)

    def _finalize_results(self) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame]]:
        """Finalize and return results."""
        # Close any open positions at the last prices
        last_timestamp = max(df.index[-1] for df in self.data.values())
        last_prices = {symbol: df["Close"].iloc[-1] for symbol, df in self.data.items()}
        self.portfolio.close_all_positions(last_timestamp, last_prices)

        pf_results = self.portfolio.get_results()
        # Trades are already Trade objects from portfolio
        trades: List[Trade] = pf_results.trades
        # Create equity curve as pd.Series
        equity_curve = pd.Series(pf_results.equity_curve)
        return self.portfolio.get_results(), self.data
