import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List


class Portfolio:
    """Portfolio management for positions and P&L."""

    def __init__(self, initial_capital: float, position_size: float, commission: float):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol: quantity
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]

    async def process_signals(self, order_queue: asyncio.Queue):
        """Process signals from order_queue and execute orders."""
        while True:
            signal = await order_queue.get()
            if signal is None:  # End of signals
                break

            # Execute order
            trade = self._execute_order(signal)
            print(f"Executed: {trade}")

    def _execute_order(self, signal: Dict) -> Dict:
        """Execute order based on signal."""
        qty = 0
        if signal["action"] == "BUY":
            # Long symbol1, short symbol2
            qty = (
                self.position_size * self.cash / signal.get("price", 100)
            )  # Placeholder price
            self.positions[signal["symbol1"]] = (
                self.positions.get(signal["symbol1"], 0) + qty
            )
            self.positions[signal["symbol2"]] = (
                self.positions.get(signal["symbol2"], 0) - qty
            )
            self.cash -= qty * signal.get("price", 100) * (1 + self.commission)
        elif signal["action"] == "SELL":
            # Short symbol1, long symbol2
            qty = self.position_size * self.cash / signal.get("price", 100)
            self.positions[signal["symbol1"]] = (
                self.positions.get(signal["symbol1"], 0) - qty
            )
            self.positions[signal["symbol2"]] = (
                self.positions.get(signal["symbol2"], 0) + qty
            )
            self.cash -= qty * signal.get("price", 100) * (1 + self.commission)

        # Record trade
        trade = {
            "timestamp": pd.Timestamp.now(),
            "action": signal["action"],
            "symbol1": signal["symbol1"],
            "symbol2": signal["symbol2"],
            "z_score": signal["z_score"],
            "qty": qty,
            "price": signal.get("price", 100),
        }
        self.trades.append(trade)
        self.equity_curve.append(
            self.cash + sum(pos * 100 for pos in self.positions.values())
        )  # Placeholder valuation

        return trade

    def get_results(self) -> Dict:
        """Get backtest results."""
        total_return = (
            self.equity_curve[-1] - self.initial_capital
        ) / self.initial_capital
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        )
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }
