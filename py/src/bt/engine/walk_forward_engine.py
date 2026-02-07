from typing import List, Optional

from src.bt.algos.pairs_trading import PairsTradingStrategy, StrategyParams
from src.bt.algos.z_model import ZModel
from src.bt.engine.backtest_engine import BacktestEngine
from src.bt.types import ExecutionParams
from src.utils import pick


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
        self.symbols = symbols
        self.initial_train_start = initial_train_start
        self.initial_train_end = initial_train_end
        self.trading_start = trading_start
        self.trading_end = trading_end
        self.plot = kwargs.get("plot")

        # Strategy parameters
        self.strategy_params = StrategyParams(
            entry_z=kwargs.get("entry_z", 2.0),
            exit_z=kwargs.get("exit_threshold", 0.5),
        )

        # ZModel parameters
        self.rolling_window_size = kwargs.get("rolling_window_size", 20)

        # Portfolio parameters
        self.pf_params = pick(
            kwargs,
            [
                "initial_capital",
                "position_size",
                "commission",
                "stop_loss",
                "take_profit",
            ],
        )

        # Execution parameters
        if "spread_bps" in kwargs or "slippage_bps" in kwargs:
            self.execution_params = ExecutionParams(
                spread_bps=kwargs.get("spread_bps", 5.0),
                slippage_bps=kwargs.get("slippage_bps", 2.0),
            )
        else:
            self.execution_params = None

    async def run(self):
        # Create strategy and model
        z_model = ZModel(self.symbols, self.rolling_window_size)
        strategy = PairsTradingStrategy(
            symbols=self.symbols,
            strategy_params=self.strategy_params,
        )

        # Create unified engine
        engine = BacktestEngine(
            strategy=strategy,
            z_model=z_model,
            symbols=self.symbols,
            train_start=self.initial_train_start,
            train_end=self.initial_train_end,
            test_start=self.trading_start,
            test_end=self.trading_end,
            initial_capital=self.pf_params.get("initial_capital", 10000),
            position_size=self.pf_params.get("position_size", 0.1),
            commission=self.pf_params.get("commission", 0.001),
            stop_loss=self.pf_params.get("stop_loss", 0.10),
            take_profit=self.pf_params.get("take_profit", 1.0),
            execution_params=self.execution_params,
        )

        results, data, z_scores = await engine.run()

        # Plot if requested
        if self.plot:
            from src.bt.plotting.plotting import plot_backtest_results

            plot_backtest_results(
                results,
                self.symbols,
                "Pairs Trading",
                data,
                z_scores=z_scores,
                entry_z=self.strategy_params.entry_z,
                exit_z=self.strategy_params.exit_z,
            )

        return results
