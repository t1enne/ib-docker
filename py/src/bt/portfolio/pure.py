"""Pure portfolio functions.

All functions are pure: they take state and inputs, return new state.
No mutations, no side effects.
"""

from typing import Dict, Optional
import pandas as pd

from src.bt.state.types import (
    PortfolioState,
    Position,
    Trade,
    FillEvent,
    EquityPoint,
    ActionType,
    TradeStatus,
    TradeExitReason,
)


def apply_fill(
    portfolio: PortfolioState,
    fill: FillEvent,
    position_size_pct: float = 0.2,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.1,
) -> PortfolioState:
    """Apply a fill to portfolio, return new state.

    This is a pure function - input portfolio is not modified.

    Args:
        portfolio: Current portfolio state
        fill: The fill event to apply
        position_size_pct: % of capital to use per position (default 20%)
        stop_loss_pct: % stop loss (default 5%)
        take_profit_pct: % take profit (default 10%)
    """
    signal = fill.signal

    if signal.action == ActionType.close:
        return _close_position(portfolio, fill)

    return _open_position(
        portfolio,
        fill,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


def _open_position(
    portfolio: PortfolioState,
    fill: FillEvent,
    position_size_pct: float = 0.2,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.1,
) -> PortfolioState:
    """Open a new position from fill."""
    signal = fill.signal

    # Calculate position quantity using config
    is_long = signal.action == ActionType.long
    base_qty = portfolio.cash * position_size_pct / fill.executed_price
    qty = round(base_qty * (signal.hedge_beta or 1.0), 4)
    if qty <= 0:
        return portfolio

    # Calculate SL/TP using config
    if is_long:
        stop_loss = round(fill.executed_price * (1 - stop_loss_pct), 2)
        take_profit = round(fill.executed_price * (1 + take_profit_pct), 2)
    else:
        stop_loss = round(fill.executed_price * (1 + stop_loss_pct), 2)
        take_profit = round(fill.executed_price * (1 - take_profit_pct), 2)

    # TODO: Partial take profit support - for strategies like vol_extension_pullback
    # that want to take 50% profit at 2R and trail the remainder.
    # This would require modifying Position to track partial fills and remaining qty.

    # Create position
    position = Position(
        symbol=signal.symbol,
        qty=qty,
        entry_price=fill.executed_price,
        entry_time=fill.timestamp,
        stop_loss=stop_loss,
        take_profit=take_profit,
        last_price=fill.executed_price,
    )

    # Calculate new cash
    direction = 1 if is_long else -1
    cash_used = qty * fill.executed_price + fill.commission
    new_cash = portfolio.cash - cash_used

    # Create trade record
    trade = Trade(
        entry_time=signal.timestamp,
        entry_price=fill.executed_price,
        exit_time=None,
        exit_price=None,
        last_price=fill.executed_price,
        z_score=signal.z_score,
        symbol=signal.symbol,
        position=signal.action,
        qty=qty,
        stop_loss=stop_loss,
        take_profit=take_profit,
        pnl=0.0,
        status=TradeStatus.open,
        close_reason=None,
    )

    # Build new state
    new_positions = dict(portfolio.positions)
    new_positions[signal.symbol] = position

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=portfolio.trades + (trade,),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _close_position(portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
    """Close a position from fill."""
    symbol = fill.signal.symbol
    position = portfolio.positions.get(symbol)

    # No position to close
    if not position:
        return portfolio

    # Calculate PnL
    is_long = position.qty > 0
    qty = abs(position.qty)

    if is_long:
        pnl = (fill.executed_price - position.entry_price) * qty
        cash_change = qty * fill.executed_price - fill.commission
    else:
        pnl = (position.entry_price - fill.executed_price) * qty
        cash_change = (qty * position.entry_price) + pnl - fill.commission

    # Update trade record
    # Find the matching open trade
    updated_trades = list(portfolio.trades)
    for i, trade in enumerate(updated_trades):
        if trade.symbol == symbol and trade.status == TradeStatus.open:
            updated_trades[i] = Trade(
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=fill.timestamp,
                exit_price=fill.executed_price,
                last_price=fill.executed_price,
                z_score=trade.z_score,
                symbol=trade.symbol,
                position=trade.position,
                qty=trade.qty,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                pnl=pnl,
                status=TradeStatus.closed,
                close_reason=fill.signal.reason or TradeExitReason.none,
            )
            break

    # Build new state
    new_positions = {k: v for k, v in portfolio.positions.items() if k != symbol}
    new_cash = portfolio.cash + cash_change

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=tuple(updated_trades),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def update_prices(portfolio: PortfolioState, tick) -> PortfolioState:
    """Update position prices and equity curve with new tick."""
    symbol = tick.symbol
    new_positions = dict(portfolio.positions)

    # Update position price if it exists
    if symbol in portfolio.positions:
        position = portfolio.positions[symbol]
        updated_position = Position(
            symbol=position.symbol,
            qty=position.qty,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            last_price=tick.close,
        )
        new_positions[symbol] = updated_position

    # Calculate equity (always do this, even if no positions)
    positions_value = calculate_positions_value(new_positions)
    equity = portfolio.cash + positions_value

    # Add equity point
    equity_point = EquityPoint(
        timestamp=tick.timestamp,
        equity=equity,
        cash=portfolio.cash,
        positions_value=positions_value,
    )

    return PortfolioState(
        cash=portfolio.cash,
        positions=new_positions,
        trades=portfolio.trades,
        equity_curve=portfolio.equity_curve + (equity_point,),
        initial_capital=portfolio.initial_capital,
    )


def calculate_positions_value(positions: Dict[str, Position]) -> float:
    """Calculate total value of all positions."""
    value = 0.0
    for position in positions.values():
        if position.qty > 0:
            value += position.qty * position.last_price
        else:
            # Short: collateral + unrealized pnl
            value += abs(position.qty) * (
                2 * position.entry_price - position.last_price
            )
    return value


def calculate_equity(portfolio: PortfolioState) -> float:
    """Calculate total equity."""
    return portfolio.cash + calculate_positions_value(portfolio.positions)


def get_open_position(portfolio: PortfolioState, symbol: str) -> Optional[Position]:
    """Get open position for symbol if exists."""
    return portfolio.positions.get(symbol)
