from typing import List
import pandas as pd

from src.bt.algos.pairs_trading import PairsTradingStrategy
from src.bt.engine.backtest_engine import BacktestEngine, DataFeed
from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
from src.utils import pick, read_candles
from src.bt.plotting.plotting import plot_backtest_results


class WalkForwardEngine:
    """
    Production-grade walk-forward testing engine.

    Implements rolling window analysis where:
    - Train on historical data (training window)
    - Test on future data (testing/trading window)
    - Roll forward and repeat
    """

    def __init__(
        self,
        strategy_class,
        symbols: List[str],
        initial_train_start: str,
        initial_train_end: str,
        trading_start: str,
        trading_end: str,
        **kwargs,
    ):
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.initial_train_start = pd.Timestamp(initial_train_start)
        self.initial_train_end = pd.Timestamp(initial_train_end)
        self.trading_start = trading_start
        self.trading_end = trading_end
        self.plot = kwargs.get("plot")
        self.historical_data = {
            symbol: read_candles(symbol, initial_train_start, initial_train_end)
            for symbol in self.symbols
        }
        # Separate strategy and portfolio parameters
        self.strategy_params = pick(
            kwargs, ["entry_z", "stop_loss", "take_profit", "rolling_window_size"]
        )
        self.pf_params = pick(
            kwargs, ["initial_capital", "position_size", "commission"]
        )

    async def run(self):
        portfolio = Portfolio(
            PortfolioProps(
                initial_capital=self.pf_params.get("initial_capital", 10000),
                position_size=self.pf_params.get("position_size", 0.1),
                commission=self.pf_params.get("commission", 0.001),
                start_date=pd.Timestamp(self.trading_start),
            ),
        )
        strat = PairsTradingStrategy(symbols=self.symbols, **self.strategy_params)
        strat.populate_historical_data(self.historical_data)
        # self._initial_train(strat)
        # Initial training to fit the model
        feed = DataFeed(
            self.symbols,
            self.trading_start,  # Start testing from end of training
            self.trading_end,  # Run to end of data
        )
        ngn = BacktestEngine(strat, portfolio, feed)
        results, data = await ngn.run()
        # Merge training and trading data for plotting
        for symbol in self.symbols:
            data[symbol] = pd.concat([self.historical_data[symbol], data[symbol]])

        # Plot if requested
        if self.plot:
            plot_backtest_results(results, self.symbols, "Pairs Trading", data)
        return results
