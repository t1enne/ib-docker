from collections import defaultdict
from src.bt.algos.z_model import ZModel
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, AsyncGenerator, Tuple, Dict, Optional
from src.bt.portfolio import Portfolio, PortfolioProps
from src.utils import read_candles
from src.bt.types import (
    Tick,
    StrategyProtocol,
    PortfolioResult,
    TradeSignal,
    ExecutionParams,
)
from src.bt.execution import ExecutionHandler


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
        candles_list = [
            read_candles(x, self.start_date, self.end_date) for x in self.symbols
        ]

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

        ticks.sort(key=lambda x: x.timestamp)

        for tick in ticks:
            yield tick


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: Dict[str, pd.DataFrame]
    test_data: Dict[str, pd.DataFrame]


class BacktestEngine:
    """Unified backtesting engine with rolling z-score calculation."""

    def __init__(
        self,
        strategy: StrategyProtocol,
        symbols: List[str],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        rolling_window_size: int = 20,
        initial_capital: float = 10000,
        position_size: float = 0.1,
        commission: float = 0.001,
        stop_loss: float = 0.10,
        take_profit: float = 1.0,
        execution_params: Optional[ExecutionParams] = None,
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size
        self.execution_handler = (
            ExecutionHandler(execution_params) if execution_params else None
        )

        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)

        self.portfolio = Portfolio(
            PortfolioProps(
                stop_loss=stop_loss,
                take_profit=take_profit,
                initial_capital=initial_capital,
                position_size=position_size,
                commission=commission,
                start_date=self.test_start,
            )
        )

        self.data = {
            symbol: read_candles(
                symbol,
                self.test_start.strftime("%Y-%m-%d"),
                self.test_end.strftime("%Y-%m-%d"),
            )
            for symbol in symbols
        }

        self.pending_signals: List[TradeSignal] = []

        self.z_model = ZModel(symbols, rolling_window_size)
        self.price_buffers: List[dict[str, float]] = []
        self.pending_prices: dict[pd.Timestamp, dict[str, float]] = defaultdict(dict)
        self.z_scores: List[float] = []
        self.z_timestamps: List[pd.Timestamp] = []
        self.z_scores_synchronized: List[float] = []
        self.z_timestamps_synchronized: List[pd.Timestamp] = []

    def calculate_zscores_synchronized(self) -> Tuple[pd.Series, List[Dict]]:
        """Calculate z-scores using synchronized DataFrames (matches spread module).

        Returns:
            z_scores: pd.Series with datetime index
            raw_values: List of dicts with timestamp, z, s1, s2 for comparison
        """
        prices1 = self.data[self.symbols[0]]["Close"].tolist()
        prices2 = self.data[self.symbols[1]]["Close"].tolist()
        dates = self.data[self.symbols[0]].index

        z_scores = []
        raw_values = []

        for i in range(len(prices1)):
            s1 = prices1[: i + 1]
            s2 = prices2[: i + 1]
            z = self.z_model.calculate_z_by_index(s1, s2, self.rolling_window_size)
            z_scores.append(z)
            raw_values.append(
                {
                    "timestamp": dates[i],
                    "z": z,
                    "s1": prices1[i],
                    "s2": prices2[i],
                    "data_points": i + 1,
                }
            )

        return pd.Series(z_scores, index=dates), raw_values

    def _compute_windows(self) -> List[WalkForwardWindow]:
        """Compute walk-forward windows."""
        train_data = {
            symbol: read_candles(
                symbol,
                self.train_start.strftime("%Y-%m-%d"),
                self.train_end.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
        }
        test_data = {
            symbol: read_candles(
                symbol,
                self.test_start.strftime("%Y-%m-%d"),
                self.test_end.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
        }

        return [
            WalkForwardWindow(
                train_start=self.train_start,
                train_end=self.train_end,
                test_start=self.test_start,
                test_end=self.test_end,
                train_data=train_data,
                test_data=test_data,
            )
        ]

    async def run(
        self,
    ) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame], pd.DataFrame]:
        """Run the backtest with rolling z-score calculation."""
        windows = self._compute_windows()

        for window in windows:
            await self._run_backtest(window.test_data)

        results, data, _ = self._finalize_results()
        z_scores_df = pd.DataFrame(
            {"z": self.z_scores},
            index=pd.Index(self.z_timestamps),
        )
        return results, data, z_scores_df

    async def _run_backtest(self, test_data: Dict[str, pd.DataFrame]):
        """Run backtest on test data with rolling z-score."""
        feed = DataFeed(
            self.symbols,
            str(test_data[self.symbols[0]].index[0]),
            str(test_data[self.symbols[0]].index[-1]),
        )
        current_z = 0.0
        tick_groups = defaultdict(list)
        async for tick in feed.get_data_stream():
            if tick is None:
                break

            for signal in self.pending_signals:
                if self.execution_handler:
                    fill = self.execution_handler.execute(signal, tick)
                    self.portfolio.on_fill(fill)
                else:
                    self.portfolio.on_signal(signal)

            tick_groups[tick.timestamp].append(tick)

            if len(tick_groups[tick.timestamp]) == len(self.symbols):
                for t in tick_groups[tick.timestamp]:
                    self.pending_prices[tick.timestamp][t.symbol] = t.close

                prices = dict(self.pending_prices[tick.timestamp])
                self.price_buffers.append(prices)

                if len(self.price_buffers) >= self.rolling_window_size:
                    if len(self.price_buffers) > self.rolling_window_size:
                        self.price_buffers = self.price_buffers[
                            -self.rolling_window_size :
                        ]
                    current_z = self.z_model.calculate_z(self.price_buffers)
                    self.z_scores.append(current_z)
                    self.z_timestamps.append(tick.timestamp)

                del tick_groups[tick.timestamp]

            self.pending_signals = self.strategy.on_tick(tick, current_z)
            self.portfolio.on_tick(tick)

    def _finalize_results(
        self,
    ) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame], pd.DataFrame]:
        """Finalize and return results."""
        last_timestamp = max(df.index[-1] for df in self.data.values())
        last_prices = {symbol: df["Close"].iloc[-1] for symbol, df in self.data.items()}
        self.portfolio.close_all_trades(last_timestamp, last_prices)

        z_scores_df = pd.DataFrame(
            {"z": self.z_scores},
            index=pd.Index(self.z_timestamps),
        )

        return self.portfolio.get_results(), self.data, z_scores_df
