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
    TradeExitReason,
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
        self.positions: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}
        self.equity_curve: pd.DataFrame
        start_date = cast(pd.Timestamp, props.start_date)
        self.equity_curve = pd.DataFrame(
            {
                "equity": [float(props.initial_capital)],
                "cash": [float(props.initial_capital)],
                "positions_value": [0.0],
            },
            index=pd.DatetimeIndex([start_date]),
        )
        self.equity_curve.index.name = "timestamp"

    def _record_equity(self, timestamp: pd.Timestamp):
        positions_value = sum(t.qty * t.last_price for t in self.open_trades.values())
        equity = float(self.cash) + float(positions_value)

        if timestamp in self.equity_curve.index:
            self.equity_curve.loc[timestamp, "equity"] = float(equity)
            self.equity_curve.loc[timestamp, "cash"] = float(self.cash)
            self.equity_curve.loc[timestamp, "positions_value"] = float(positions_value)
        else:
            new_row = pd.DataFrame(
                {
                    "equity": [float(equity)],
                    "cash": [float(self.cash)],
                    "positions_value": [float(positions_value)],
                },
                index=pd.DatetimeIndex([timestamp]),
            )
            self.equity_curve = pd.concat([self.equity_curve, new_row])

    def update_market_value(self, tick: Tick):
        """Update market value of positions based on tick price."""
        if tick.symbol in self.open_trades:
            self.open_trades[tick.symbol].last_price = tick.close
        self._record_equity(tick.timestamp)

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

    def get_results(self) -> PortfolioResult:
        """Get backtest results."""
        equity_series = self.equity_curve["equity"].sort_index()
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
        self._record_equity(fill.signal.timestamp)
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
        trade.last_price = fill.executed_price
        trade.pnl = pnl
        trade.status = TradeStatus.closed
        trade.close_reason = fill.signal.reason

        del self.open_trades[fill.signal.symbol]
        self.cash += qty * trade.exit_price - fill.commission
        self._record_equity(fill.signal.timestamp)

        return trade


__all__ = ["PortfolioProps", "Portfolio"]
