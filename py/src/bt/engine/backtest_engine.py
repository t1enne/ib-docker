import asyncio
from typing import List, AsyncGenerator
import pandas as pd

from src.utils import read_candles
from src.bt.types import Tick, BacktestResult, Trade


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

    async def get_data_stream(self) -> AsyncGenerator[Tick, None]:
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


class BacktestEngine:
    """Main backtesting engine using asyncio."""

    def __init__(self, strategy, portfolio, data_feed: DataFeed):
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

    async def run(self):
        """Run the backtest simulation asynchronously."""
        # Create queues for communication
        signal_queue = asyncio.Queue()
        order_queue = asyncio.Queue()

        # Create tasks for each component
        data_task = asyncio.create_task(self._run_data_feed(signal_queue))
        strategy_task = asyncio.create_task(
            self.strategy.process_data(signal_queue, order_queue)
        )
        portfolio_task = asyncio.create_task(
            self.portfolio.process_signals(order_queue)
        )

        # Wait for all tasks to complete
        await asyncio.gather(data_task, strategy_task, portfolio_task)

        # Finalize results
        return self._finalize_results()

    async def _run_data_feed(self, signal_queue: asyncio.Queue):
        """Run the data feed and put ticks into the signal queue."""
        async for tick in self.data_feed.get_data_stream():
            await signal_queue.put(tick)
        # Signal end of data
        await signal_queue.put(None)

    def _finalize_results(self):
        """Finalize and return results."""

        pf_results = self.portfolio.get_results()
        # Trades are already Trade objects from portfolio
        trades: List[Trade] = pf_results.trades
        # Create equity curve as pd.Series
        equity_curve = pd.Series(pf_results.equity_curve)
        results = BacktestResult(
            total_return=pf_results.total_return,
            sharpe_ratio=pf_results.sharpe_ratio,
            max_drawdown=0.0,  # Placeholder
            win_rate=0.0,  # Placeholder
            total_trades=len(trades),
            profitable_trades=len(list(filter(lambda t: t.pnl > 0, trades))),
            trades=trades,
            equity_curve=equity_curve,
        )
        return results, self.data
