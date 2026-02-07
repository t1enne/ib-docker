from src.bt.algos.z_model import ZModel, TrainedZModel
import pandas as pd
from dataclasses import dataclass
from typing import List, AsyncGenerator, Tuple, Dict, Optional, cast
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
    """Unified backtesting engine with walk-forward and execution support."""

    def __init__(
        self,
        strategy: StrategyProtocol,
        z_model: ZModel,
        symbols: List[str],
        # Walk-forward parameters
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        # Portfolio parameters
        initial_capital: float = 10000,
        position_size: float = 0.1,
        commission: float = 0.001,
        stop_loss: float = 0.10,
        take_profit: float = 1.0,
        # Execution parameters
        execution_params: Optional[ExecutionParams] = None,
    ):
        self.strategy = strategy
        self.z_model = z_model
        self.symbols = symbols
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

    def _train_model(self, train_data: Dict[str, pd.DataFrame]) -> TrainedZModel:
        """Train the model on training data."""
        return self.z_model.train(train_data)

    async def run(
        self,
    ) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame], pd.DataFrame]:
        """Run the backtest with walk-forward windows."""
        windows = self._compute_windows()
        print(windows)

        all_z_scores: List[pd.DataFrame] = []
        z_scores_df = pd.DataFrame()

        for window in windows:
            trained_model = self._train_model(window.train_data)
            self.strategy.set_model(trained_model)

            await self._run_backtest(window.test_data)

            strat = cast(StrategyProtocol, self.strategy)
            z = strat.get_z_scores()
            if isinstance(z, pd.DataFrame):
                if not z.empty:
                    all_z_scores.append(z.copy())

        if all_z_scores:
            z_scores_df = pd.concat(all_z_scores)
            z_scores_df = z_scores_df[~z_scores_df.index.duplicated(keep="first")]
            z_scores_df = pd.DataFrame({"z": z_scores_df["z"]})

        results, data = self._finalize_results()
        return results, data, z_scores_df

    async def _run_backtest(self, test_data: Dict[str, pd.DataFrame]):
        """Run backtest on test data."""
        strat = cast(StrategyProtocol, self.strategy)
        # for symbol in self.symbols:
        #     strat.bps.hdata[symbol] = test_data[symbol].copy()

        feed = DataFeed(
            self.symbols,
            str(test_data[self.symbols[0]].index[0]),
            str(test_data[self.symbols[0]].index[-1]),
        )

        async for tick in feed.get_data_stream():
            if tick is None:
                break
            
            for signal in self.pending_signals:
                if self.execution_handler:
                    fill = self.execution_handler.execute(signal, tick)
                    self.portfolio.on_fill(fill)
                else:
                    self.portfolio.on_signal(signal)

            self.pending_signals = self.strategy.on_tick(tick)

            self.portfolio.on_tick(tick)

    def _finalize_results(self) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame]]:
        """Finalize and return results."""
        last_timestamp = max(df.index[-1] for df in self.data.values())
        last_prices = {symbol: df["Close"].iloc[-1] for symbol, df in self.data.items()}
        self.portfolio.close_all_positions(last_timestamp, last_prices)

        return self.portfolio.get_results(), self.data
