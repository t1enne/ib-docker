import asyncio
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List

from src.bt.types import ActionType, Trade, TradeSignal, PortfolioResult


@dataclass
class PortfolioProps:
    initial_capital: float
    position_size: float
    commission: float


class Portfolio:
    """Portfolio management for positions and P&L."""

    def __init__(self, props: PortfolioProps):
        self.initial_capital = props.initial_capital
        self.position_size = props.position_size
        self.commission = props.commission
        self.cash = props.initial_capital
        self.positions: Dict[str, float] = {}  # symbol: quantity
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [props.initial_capital]

    async def process_signals(self, order_queue: asyncio.Queue):
        """Process signals from order_queue and execute orders."""
        while True:
            signal = await order_queue.get()
            if signal is None:  # End of signals
                break

            # Execute order
            trade = self._execute_order(signal)
            print(f"Executed: {str(trade.entry_time.date())}")

    def _execute_order(self, signal: TradeSignal) -> Trade:
        """Execute order based on signal."""
        qty = 0
        if signal.action == ActionType.long:
            qty = round(self.position_size * self.cash / signal.price, 4)
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + qty
            self.cash -= qty * signal.price * (1 + self.commission)
        elif signal.action == ActionType.short:
            # Short symbol1, long symbol2
            qty = round(self.position_size * self.cash / signal.price, 4)
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) - qty
            self.cash -= qty * signal.price * (1 + self.commission)

        # Record trade
        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=signal.price,
            exit_time=None,
            exit_price=None,
            z_score=signal.z_score,
            symbol=signal.symbol,
            position=signal.action,
        )
        self.trades.append(trade)
        self.equity_curve.append(
            self.cash + sum(pos * 100 for pos in self.positions.values())
        )  # Placeholder valuation

        return trade

    def get_results(self) -> PortfolioResult:
        """Get backtest results."""
        total_return = (
            self.equity_curve[-1] - self.initial_capital
        ) / self.initial_capital
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        )
        return PortfolioResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )
