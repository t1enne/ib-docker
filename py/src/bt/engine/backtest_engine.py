import asyncio
from typing import List, AsyncGenerator, Dict
from src.utils import read_candles


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

    async def get_data_stream(self) -> AsyncGenerator[Dict, None]:
        """Returns an async generator that yields market data ticks for all symbols."""
        # Load data for all symbols
        data_dict = {}
        for symbol in self.symbols:
            df = read_candles(symbol, self.start_date, self.end_date)
            data_dict[symbol] = df

        # Combine into a single stream of ticks
        ticks = []
        for symbol, df in data_dict.items():
            for idx, row in df.iterrows():
                ticks.append(
                    {
                        "timestamp": idx,
                        "symbol": symbol,
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"],
                    }
                )

        # Sort by timestamp
        ticks.sort(key=lambda x: x["timestamp"])

        # Yield ticks asynchronously
        for tick in ticks:
            yield tick
            # Simulate real-time by sleeping for a small amount
            await asyncio.sleep(0.001)


class BacktestEngine:
    """Main backtesting engine using asyncio."""

    def __init__(self, strategy, portfolio, data_feed: DataFeed):
        self.strategy = strategy
        self.portfolio = portfolio
        self.data_feed = data_feed

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
        self._finalize_results()

    async def _run_data_feed(self, signal_queue: asyncio.Queue):
        """Run the data feed and put ticks into the signal queue."""
        async for tick in self.data_feed.get_data_stream():
            await signal_queue.put(tick)
        # Signal end of data
        await signal_queue.put(None)

    def _finalize_results(self):
        """Finalize and display results."""
        results = self.portfolio.get_results()
        print("Backtest completed.")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        # Add more metrics
