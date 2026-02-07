from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from pandas._libs import NaTType
import src.bt.metrics as metrics

from src.bt.types import (
    ActionType,
    Tick,
    Trade,
    TradeSignal,
    PortfolioResult,
    TradeStatus,
    FillEvent,
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
        # udpate equity for the position
        if open_pos:
            self._update_equity_on_tick(open_pos, tick)

        is_long = open_pos.position == ActionType.long
        should_close_long = is_long and (
            tick.close < open_pos.stop_loss or open_pos.take_profit < tick.close
        )
        should_close_short = not is_long and (
            open_pos.stop_loss < tick.close or tick.close < open_pos.take_profit
        )

        should_close = should_close_long or should_close_short
        if should_close:
            reason = (
                "stop_loss"
                if (should_close_long and tick.close <= open_pos.stop_loss)
                or (should_close_short and tick.close >= open_pos.stop_loss)
                else "take_profit"
            )
            self._send_close_signal(tick.symbol, tick.close, tick.timestamp, reason)

        self._update_sl(open_pos, tick)

    def on_signal(self, signal: TradeSignal) -> Optional[Trade]:
        """Execute order based on signal."""
        # CLOSE
        if signal.action == ActionType.close:
            if signal.symbol not in self.open_trades:
                return None  # No open trade to close

            open_trade = self.open_trades[signal.symbol]
            return self._close_trade(open_trade, signal)
        if signal.symbol in self.open_trades:
            return None  # Already have position

        # OPEN
        qty = round(self.position_size * self.cash / signal.price, 4)
        return self._open_trade(signal, qty)

    def on_fill(self, fill: FillEvent) -> Optional[Trade]:
        """Execute order based on fill event from execution handler."""
        signal = fill.signal
        if signal.action == ActionType.close:
            if signal.symbol not in self.open_trades:
                return None
            open_trade = self.open_trades[signal.symbol]
            return self._close_trade_from_fill(open_trade, fill)
        if signal.symbol in self.open_trades:
            return None
        return self._open_trade_from_fill(signal, fill)

    def close_all_positions(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """Close all open positions at the given prices."""
        for symbol, _ in list(self.open_trades.items()):
            p = prices[symbol]
            self._send_close_signal(symbol, p, timestamp)

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

        annual_return = metrics.annual_return(equity_series)
        annual_volatility = metrics.annual_volatility(equity_series)
        max_drawdown = metrics.max_drawdown(equity_series)
        calmar_ratio = metrics.calmar_ratio(equity_series)
        sortino_ratio = metrics.sortino_ratio(equity_series)
        omega_ratio = metrics.omega_ratio(equity_series)
        skewness = metrics.skewness(equity_series)
        kurtosis = metrics.kurtosis(equity_series)
        stability = metrics.stability(equity_series)
        alpha, beta = metrics.alpha_beta(equity_series)

        return PortfolioResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            trades=self.trades,
            equity_curve=equity_series,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            alpha=alpha,
            beta=beta,
            skewness=skewness,
            kurtosis=kurtosis,
            stability=stability,
            omega_ratio=omega_ratio,
        )

    def _update_equity_on_tick(self, pos: Trade, tick: Tick):
        assert pos and tick
        self.open_trades[pos.symbol].last_price = tick.close
        self.equity_curve[tick.timestamp] = self._calc_current_equity()

    def _calc_current_equity(self):
        return self.cash + sum(
            [t.qty * t.last_price for t in self.open_trades.values()]
        )

    def _send_close_signal(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
        reason: str = "unknown",
    ):
        signal = TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=0.0,  # Neutral
            timestamp=timestamp,
            price=price,
            reason=reason,
        )
        self.on_signal(signal)

    def _open_trade(self, signal: TradeSignal, qty: float):
        """Open position, deduct fees from cash and updates equity curve"""
        is_long = signal.action == ActionType.long
        sl = 0.0
        tp = 0.0
        if is_long:
            sl = round(signal.price * (1 - self.stop_loss), 2)
            tp = round(signal.price * (1 + self.take_profit), 2)
        else:
            sl = signal.price * (1 + self.stop_loss)
            tp = signal.price * (1 - self.take_profit)
        # Record trade
        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=signal.price,
            last_price=signal.price,
            qty=qty,
            z_score=signal.z_score,
            symbol=signal.symbol,
            stop_loss=sl,
            take_profit=tp,
            position=signal.action,
            exit_time=None,
            exit_price=None,
        )
        self.trades.append(trade)
        self.open_trades[signal.symbol] = trade

        direction = 1 if is_long else -1
        self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + (
            qty * direction
        )
        self.cash -= qty * signal.price + self.commission
        self.equity_curve[signal.timestamp] = self._calc_current_equity()
        # update equity curve
        return trade

    def _close_trade(self, trade: Trade, signal: TradeSignal):
        """Closes position, adds pnl to cash, updates equity curve"""
        qty = abs(self.positions.get(signal.symbol, 0))
        assert qty > 0
        is_long = trade.position == ActionType.long
        # Calculate P&L
        pnl = (
            (signal.price - trade.entry_price) * qty
            if is_long
            else (trade.entry_price - signal.price) * qty
        )

        self.positions[signal.symbol] = 0
        # Update trade
        trade.exit_time = signal.timestamp
        trade.exit_price = signal.price
        trade.pnl = pnl
        trade.status = TradeStatus.closed
        trade.close_reason = signal.reason

        # Update cash and equity
        self.cash += pnl - self.commission
        self.equity_curve[signal.timestamp] = self._calc_current_equity()

        # print(f"Closing {trade.position} trade with {round(open_trade.pnl, 2):>6} on {str(signal.timestamp)} (reason: {open_trade.close_reason}) sym: {open_trade.symbol:>4}")
        del self.open_trades[signal.symbol]
        return trade

    def _update_sl(self, trade: Trade, tick: Tick):
        is_long = trade.position == ActionType.long

        if is_long:
            max_price = max(trade.entry_price, tick.close)
            trade.stop_loss = max_price * (1 - self.stop_loss)
            return
        # short
        min_price = min(trade.entry_price, tick.close)
        trade.stop_loss = min_price * (1 - self.stop_loss)

    def _open_trade_from_fill(self, signal: TradeSignal, fill: FillEvent):
        """Open position from fill event with execution pricing."""
        is_long = signal.action == ActionType.long
        qty = round(self.position_size * self.cash / fill.executed_price, 4)
        sl = 0.0
        tp = 0.0
        if is_long:
            sl = round(fill.executed_price * (1 - self.stop_loss), 2)
            tp = round(fill.executed_price * (1 + self.take_profit), 2)
        else:
            sl = fill.executed_price * (1 + self.stop_loss)
            tp = fill.executed_price * (1 - self.take_profit)

        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=fill.executed_price,
            last_price=fill.executed_price,
            qty=qty,
            z_score=signal.z_score,
            symbol=signal.symbol,
            stop_loss=sl,
            take_profit=tp,
            position=signal.action,
            exit_time=None,
            exit_price=None,
        )
        self.trades.append(trade)
        self.open_trades[signal.symbol] = trade

        direction = 1 if is_long else -1
        self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + (
            qty * direction
        )
        self.cash -= (qty * fill.executed_price) + fill.commission
        self.equity_curve[signal.timestamp] = self._calc_current_equity()
        return trade

    def _close_trade_from_fill(self, trade: Trade, fill: FillEvent):
        """Close position from fill event with execution pricing."""
        qty = abs(self.positions.get(fill.signal.symbol, 0))
        is_long = trade.position == ActionType.long
        pnl = (
            (fill.executed_price - trade.entry_price) * qty
            if is_long
            else (trade.entry_price - fill.executed_price) * qty
        )

        self.positions[fill.signal.symbol] = 0
        trade.exit_time = fill.signal.timestamp
        trade.exit_price = fill.executed_price
        trade.pnl = pnl
        trade.status = TradeStatus.closed
        trade.close_reason = fill.signal.reason

        self.cash += pnl - fill.commission
        self.equity_curve[fill.signal.timestamp] = self._calc_current_equity()

        del self.open_trades[fill.signal.symbol]
        return trade

__all__ = ["PortfolioProps", "Portfolio"]
