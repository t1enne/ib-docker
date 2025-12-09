import asyncio
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.bt.engine.backtest_engine import BacktestEngine, DataFeed
from src.bt.portfolio.portfolio import Portfolio


@dataclass
class WalkForwardWindow:
    """Represents a training/testing window in walk-forward analysis."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int

    def __post_init__(self):
        # Ensure no NaT values
        if (
            pd.isna(self.train_start)
            or pd.isna(self.train_end)
            or pd.isna(self.test_start)
            or pd.isna(self.test_end)
        ):
            raise ValueError("Timestamp cannot be NaT")


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""

    window: WalkForwardWindow
    portfolio_results: Dict
    strategy_params: Dict[str, Any]
    performance_metrics: Dict[str, float]


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
        walk_forward_end: str,
        train_window_months: int = 24,  # 2 years
        test_window_months: int = 1,  # 1 month
        step_months: int = 1,  # Move forward 1 month each step
        **kwargs,
    ):
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.initial_train_start = pd.Timestamp(initial_train_start.split('T')[0] if 'T' in initial_train_start else initial_train_start)
        self.initial_train_end = pd.Timestamp(initial_train_end.split('T')[0] if 'T' in initial_train_end else initial_train_end)
        self.walk_forward_end = pd.Timestamp(walk_forward_end.split('T')[0] if 'T' in walk_forward_end else walk_forward_end)
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months

        # Separate strategy and portfolio parameters
        self.strategy_params = {
            k: v for k, v in kwargs.items()
            if k in ['entry_z', 'stop_loss', 'take_profit', 'retrain_interval_months']
        }
        self.portfolio_params = {
            k: v for k, v in kwargs.items()
            if k in ['initial_capital', 'position_size', 'commission']
        }

        self.windows: List[WalkForwardWindow] = []
        self.results: List[WalkForwardResult] = []

    def generate_windows(self) -> List[WalkForwardWindow]:
        """Generate all training/testing windows for walk-forward analysis."""
        windows = []
        window_id = 0

        current_train_start = self.initial_train_start
        current_train_end = self.initial_train_end

        while current_train_end < self.walk_forward_end:  # type: ignore
            # Define test window (immediately after training window)
            test_start = current_train_end

            # Calculate test_end by adding months (simple approximation)
            test_end_calc = test_start + pd.DateOffset(months=self.test_window_months)
            test_end = min(test_end_calc, self.walk_forward_end)

            # Ensure we have valid timestamps
            if (
                pd.isna(test_end)
                or pd.isna(current_train_start)
                or pd.isna(current_train_end)
                or pd.isna(test_start)
            ):  # type: ignore
                break

            window = WalkForwardWindow(
                train_start=current_train_start,  # type: ignore
                train_end=current_train_end,  # type: ignore
                test_start=test_start,  # type: ignore
                test_end=test_end,  # type: ignore
                window_id=window_id,
            )

            windows.append(window)
            window_id += 1

            # Move forward by step_months
            current_train_start = current_train_start + pd.DateOffset(
                months=self.step_months
            )
            current_train_end = current_train_end + pd.DateOffset(
                months=self.step_months
            )

            # Ensure we don't go beyond walk_forward_end
            if current_train_end >= self.walk_forward_end:  # type: ignore
                break

        self.windows = windows
        return windows

    async def run_window(self, window: WalkForwardWindow) -> WalkForwardResult:
        """Run backtest for a single walk-forward window."""
        # Create fresh strategy instance for this window
        strategy = self.strategy_class(
            symbols=self.symbols,
            training_start=window.train_start.strftime('%Y-%m-%d'),
            training_end=window.train_end.strftime('%Y-%m-%d'),
            **self.strategy_params,
        )

        # Create data feed for testing period only
        data_feed = DataFeed(
            symbols=self.symbols,
            start_date=window.test_start.strftime('%Y-%m-%d'),
            end_date=window.test_end.strftime('%Y-%m-%d'),
        )

        # Create fresh portfolio
        portfolio = Portfolio(
            initial_capital=self.portfolio_params.get("initial_capital", 100000),
            position_size=self.portfolio_params.get("position_size", 0.1),
            commission=self.portfolio_params.get("commission", 0.001),
        )

        # Create and run backtest engine
        engine = BacktestEngine(strategy, portfolio, data_feed)
        await engine.run()

        # Get results
        portfolio_results = portfolio.get_results()

        # Calculate additional metrics for this window
        performance_metrics = self._calculate_window_metrics(portfolio_results, window)

        # Extract strategy parameters (could be optimized in future)
        strategy_params = {
            "entry_z": self.strategy_params.get("entry_z", 2.0),
            "stop_loss": self.strategy_params.get("stop_loss", 0.05),
            "take_profit": self.strategy_params.get("take_profit", 0.10),
        }

        return WalkForwardResult(
            window=window,
            portfolio_results=portfolio_results,
            strategy_params=strategy_params,
            performance_metrics=performance_metrics,
        )

    async def run_walk_forward(self) -> List[WalkForwardResult]:
        """Run complete walk-forward analysis."""
        if not self.windows:
            self.generate_windows()

        print(f"Running walk-forward analysis with {len(self.windows)} windows...")

        # Run all windows concurrently for efficiency
        tasks = [self.run_window(window) for window in self.windows]
        self.results = await asyncio.gather(*tasks)

        return self.results

    def _calculate_window_metrics(
        self, portfolio_results: Dict, window: WalkForwardWindow
    ) -> Dict[str, float]:
        """Calculate additional metrics for a walk-forward window."""
        equity_curve = portfolio_results["equity_curve"]
        trades = portfolio_results["trades"]

        # Basic metrics
        total_return = portfolio_results["total_return"]
        sharpe_ratio = portfolio_results["sharpe_ratio"]

        # Window-specific metrics
        window_days = (window.test_end - window.test_start).days
        annualized_return = (
            (1 + total_return) ** (365 / window_days) - 1 if window_days > 0 else 0
        )

        # Trading frequency
        trades_per_day = len(trades) / window_days if window_days > 0 else 0

        # Maximum drawdown for this window
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "trades_per_day": trades_per_day,
            "total_trades": len(trades),
            "window_days": window_days,
        }

    def get_aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all walk-forward windows."""
        if not self.results:
            raise ValueError(
                "No walk-forward results available. Run run_walk_forward() first."
            )

        # Combine all equity curves
        all_equity_curves = []
        all_trades = []

        for result in self.results:
            all_equity_curves.extend(result.portfolio_results["equity_curve"])
            all_trades.extend(result.portfolio_results["trades"])

        # Calculate aggregate metrics
        if all_equity_curves:
            total_return = (
                all_equity_curves[-1] - all_equity_curves[0]
            ) / all_equity_curves[0]

            # Sharpe ratio across all windows
            returns = pd.Series(all_equity_curves).pct_change().dropna()
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            )

            # Maximum drawdown across all windows
            equity_series = pd.Series(all_equity_curves)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
        else:
            total_return = 0.0
            sharpe = 0.0
            max_drawdown = 0.0

        # Window-by-window performance
        window_returns = [r.performance_metrics["total_return"] for r in self.results]
        window_sharpes = [r.performance_metrics["sharpe_ratio"] for r in self.results]

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": len(all_trades),
            "equity_curve": all_equity_curves,
            "all_trades": all_trades,
            "window_results": self.results,
            "window_returns": window_returns,
            "window_sharpes": window_sharpes,
            "num_windows": len(self.results),
            "avg_window_return": np.mean(window_returns) if window_returns else 0,
            "std_window_return": np.std(window_returns) if window_returns else 0,
            "avg_window_sharpe": np.mean(window_sharpes) if window_sharpes else 0,
        }

    def print_summary(self):
        """Print a summary of walk-forward results."""
        if not self.results:
            print("No walk-forward results available.")
            return

        aggregate = self.get_aggregate_results()

        print("\n" + "=" * 60)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Number of windows: {aggregate['num_windows']}")
        print(f"Total Return: {aggregate['total_return']:.2%}")
        print(f"Sharpe Ratio: {aggregate['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {aggregate['max_drawdown']:.2%}")
        print(f"Total Trades: {aggregate['total_trades']}")
        print(f"Average Window Return: {aggregate['avg_window_return']:.2%}")
        print(f"Window Return Std Dev: {aggregate['std_window_return']:.2%}")
        print(f"Average Window Sharpe: {aggregate['avg_window_sharpe']:.2f}")
        print("=" * 60)

        # Print window-by-window results
        print("\nWindow-by-Window Results:")
        print("Window | Return | Sharpe | Max DD | Trades")
        print("-" * 45)
        for result in self.results:
            metrics = result.performance_metrics
            print(
                f"{result.window.window_id:6} | {metrics['total_return']:6.1%} | {metrics['sharpe_ratio']:6.2f} | {metrics['max_drawdown']:6.1%} | {metrics['total_trades']:6}"
            )
