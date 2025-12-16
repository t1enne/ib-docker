import asyncio
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from pandas._libs import NaTType

from src.bt.types import (
    ActionType,
    Tick,
    Trade,
    TradeSignal,
    PortfolioResult,
    TradeStatus,
)


@dataclass
class PortfolioProps:
    stop_loss: float
    take_profit: float
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
        self.take_profit = props.take_profit
        self.stop_loss = props.stop_loss
        self.positions: Dict[str, float] = {}  # symbol: quantity
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}  # symbol: open trade
        self.equity_curve: Dict[pd.Timestamp, float] = {}
        self.equity_curve[cast(pd.Timestamp, props.start_date)] = props.initial_capital

    def on_tick(self, tick: Tick):
        if tick.symbol not in self.open_trades:
            return None  # No open trade to close
        sym = tick.symbol
        open_pos = self.open_trades[sym]
        is_long = open_pos.position == ActionType.long
        should_close_long = is_long and (
            tick.close <= open_pos.stop_loss or open_pos.take_profit <= tick.close
        )
        should_close_short = not is_long and (
            open_pos.stop_loss <= tick.close and tick.close <= open_pos.take_profit
        )
        should_close = should_close_long or should_close_short
        if should_close:
            self._close_pos(tick.symbol, tick.close, tick.timestamp)

    def on_signal(self, signal: TradeSignal) -> Optional[Trade]:
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
            print(
                f"Closing position for {open_trade.symbol:>4} at $ {open_trade.pnl:>6} on {str(signal.timestamp)}"
            )
            del self.open_trades[signal.symbol]
            self._update_equity(signal.timestamp)
            return open_trade

        if signal.symbol in self.open_trades:
            return None  # Already have position

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
            qty=qty,
            z_score=signal.z_score,
            symbol=signal.symbol,
            stop_loss=signal.price * (1 - self.stop_loss),
            take_profit=signal.price * self.take_profit,
            position=signal.action,
            exit_time=None,
            exit_price=None,
        )
        self.trades.append(trade)
        self.open_trades[signal.symbol] = trade
        msg = f"Executed trade at  {trade.entry_price:>6} / {trade.entry_time}. {trade.status} {trade.pnl:>6} sym:{trade.symbol:>4}"
        print(msg)
        return trade

    def close_all_positions(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """Close all open positions at the given prices."""
        for symbol, _ in list(self.open_trades.items()):
            p = prices[symbol]
            self._close_pos(symbol, p, timestamp)

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

    def _update_equity(self, timestamp: pd.Timestamp):
        """Update equity curve at given timestamp."""
        current_equity = self.cash + sum(pos * 100 for pos in self.positions.values())
        self.equity_curve[timestamp] = current_equity

    def _close_pos(self, symbol: str, price: float, timestamp: pd.Timestamp):
        signal = TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=0.0,  # Neutral
            timestamp=timestamp,
            price=price,
        )
        self.on_signal(signal)
