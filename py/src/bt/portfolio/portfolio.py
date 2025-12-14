import asyncio
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from pandas._libs import NaTType

from src.bt.types import ActionType, Trade, TradeSignal, PortfolioResult, TradeStatus


@dataclass
class PortfolioProps:
    initial_capital: float
    position_size: float
    commission: float
    start_date: pd.Timestamp | NaTType


class Portfolio:
    """Portfolio management for positions and P&L."""

    def __init__(self, props: PortfolioProps):
        self.initial_capital = props.initial_capital
        self.position_size = props.position_size
        self.commission = props.commission
        self.cash = props.initial_capital
        self.positions: Dict[str, float] = {}  # symbol: quantity
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}  # symbol: open trade
        self.equity_curve: Dict[pd.Timestamp, float] = {}
        self.equity_curve[cast(pd.Timestamp, props.start_date)] = props.initial_capital

    async def process_signals(self, order_queue: asyncio.Queue):
        """Process signals from order_queue and execute orders."""
        while True:
            signal = await order_queue.get()
            if signal is None:  # End of signals
                break

            # Execute order
            trade = self._execute_order(signal)
            if trade:
                print("Executed:", trade)

    def _execute_order(self, signal: TradeSignal) -> Optional[Trade]:
        """Execute order based on signal."""
        if signal.action == ActionType.close:
            # Close existing position
            if signal.symbol not in self.open_trades:
                return None  # No open trade to close

            open_trade = self.open_trades[signal.symbol]
            qty = abs(self.positions.get(signal.symbol, 0))
            if qty <= 0:
                return None
            # Calculate P&L
            is_long = open_trade.position == ActionType.long
            pnl = (
                (open_trade.entry_price - signal.price) * qty
                if is_long
                else (signal.price - open_trade.entry_price) * qty
            )
            self.cash += pnl - self.commission
            self.positions[signal.symbol] = 0
            # Update trade
            open_trade.exit_time = signal.timestamp
            open_trade.exit_price = signal.price
            open_trade.pnl = pnl
            open_trade.status = TradeStatus.closed
            del self.open_trades[signal.symbol]
            self._update_equity(signal.timestamp)
            return open_trade

        # Open position
        qty = round(self.position_size * self.cash / signal.price, 4)
        if signal.action == ActionType.long:
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + qty
            self.cash -= qty * signal.price * (1 + self.commission)
        elif signal.action == ActionType.short:
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
        self.open_trades[signal.symbol] = trade
        return trade

    def _update_equity(self, timestamp: pd.Timestamp):
        """Update equity curve at given timestamp."""
        current_equity = self.cash + sum(pos * 100 for pos in self.positions.values())
        self.equity_curve[timestamp] = current_equity

    def close_all_positions(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """Close all open positions at the given prices."""
        for symbol, trade in list(self.open_trades.items()):
            signal = TradeSignal(
                action=ActionType.close,
                symbol=symbol,
                z_score=0.0,  # Neutral
                timestamp=timestamp,
                price=prices[symbol],
            )
            self._execute_order(signal)

    def get_results(self) -> PortfolioResult:
        """Get backtest results."""
        equity_series = pd.Series(self.equity_curve).sort_index()
        total_return = (
            equity_series.iloc[-1] - self.initial_capital
        ) / self.initial_capital
        returns = equity_series.pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        )
        return PortfolioResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            trades=self.trades,
            equity_curve=equity_series,
        )
