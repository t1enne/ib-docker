"""Pure portfolio functions.

All functions are pure: they take state and inputs, return new state.
No mutations, no side effects.

positions dict: Dict[str, Tuple[Position, ...]] — symbol → tuple of Positions.
Multiple positions per symbol are supported (e.g. partial entries, net rebalancing).
"""

from typing import Dict, Optional, Tuple as TupleT

from src.bt.state.types import (
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    FillEvent,
    EquityPoint,
    ActionType,
    TradeStatus,
    TradeExitReason,
)


def apply_fill(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Apply a fill to portfolio, return new state.

    Pure function - input portfolio is not modified.

    signal.qty must be set explicitly (>0) for open/rebalance signals.
    signal.stop_loss / signal.take_profit are used directly for new positions.
    No fallback sizing or SL/TP levels - strategies provide everything.
    """
    signal = fill.signal

    if signal.action == ActionType.close:
        return _close_position(portfolio, fill)

    if signal.action == ActionType.rebalance:
        return _rebalance_position(portfolio, fill)

    return _open_position(portfolio, fill)


def _open_position(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Open a new position from fill - appends to symbol's position tuple.

    signal.qty and signal.sl/tp must be set explicitly by the strategy.
    No fallback sizing or SL/TP levels.
    """
    signal = fill.signal

    qty = round(signal.qty, 4)
    if qty <= 0:
        return portfolio

    # Generate position_id from signal if provided, else auto-generate
    pid = signal.position_id or f"{signal.symbol}_{fill.timestamp.timestamp()}"

    # Create position — SL/TP from signal, None if not set
    position = Position(
        symbol=signal.symbol,
        qty=qty,
        entry_price=fill.executed_price,
        entry_time=fill.timestamp,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        last_price=fill.executed_price,
        type=signal.action,
        position_id=pid,
    )

    # Calculate new cash
    cash_used = qty * fill.executed_price + fill.commission
    new_cash = portfolio.cash - cash_used

    # Create trade record
    trade = Trade(
        entry_time=signal.timestamp,
        entry_price=fill.executed_price,
        exit_time=None,
        exit_price=None,
        last_price=fill.executed_price,
        symbol=signal.symbol,
        position=signal.action,
        qty=qty,
        stop_loss=signal.stop_loss or 0.0,
        take_profit=signal.take_profit or 0.0,
        pnl=0.0,
        reason=signal.reason,
        status=TradeStatus.open,
        close_reason=None,
        position_id=pid,
    )

    # Build new state — append position to symbol's tuple
    new_positions = dict(portfolio.positions)
    existing = new_positions.get(signal.symbol, ())
    new_positions[signal.symbol] = existing + (position,)

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=portfolio.trades + (trade,),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _close_position(portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
    """Close a position from fill.

    Requires position_id on the fill signal. Matches the exact position
    by position_id and removes it from the symbol's tuple. If no
    position_id is provided, raises ValueError.
    """
    symbol = fill.signal.symbol
    pid = fill.signal.position_id

    if not pid:
        raise ValueError(
            f"_close_position requires position_id on TradeSignal. "
            f"Signal for {symbol} has no position_id set."
        )

    positions_tuple = portfolio.positions.get(symbol)
    if not positions_tuple:
        return portfolio

    # Find target position by position_id
    target_idx: Optional[int] = None
    target: Optional[Position] = None
    for i, pos in enumerate(positions_tuple):
        if pos.position_id == pid:
            target_idx = i
            target = pos
            break

    if target is None:
        raise ValueError(
            f"Position {pid} not found for symbol {symbol}. "
            f"Available: {[p.position_id for p in positions_tuple]}"
        )

    # Calculate PnL
    is_long = target.type == ActionType.long
    qty = abs(target.qty)

    if is_long:
        pnl = (fill.executed_price - target.entry_price) * qty
        cash_change = qty * fill.executed_price - fill.commission
    else:
        pnl = (target.entry_price - fill.executed_price) * qty
        cash_change = (qty * target.entry_price) + pnl - fill.commission

    # Update trade record — match by position_id
    updated_trades = list(portfolio.trades)
    for i, trade in enumerate(updated_trades):
        if (
            trade.status == TradeStatus.open
            and trade.position_id == pid
            and trade.symbol == symbol
        ):
            updated_trades[i] = _close_trade(trade, fill, pnl)
            break

    # Build new positions — remove only the target position from the tuple
    assert target_idx is not None  # guaranteed by target-is-None check above
    new_positions = dict(portfolio.positions)
    remaining = positions_tuple[:target_idx] + positions_tuple[target_idx + 1 :]
    if remaining:
        new_positions[symbol] = remaining
    else:
        del new_positions[symbol]

    new_cash = portfolio.cash + cash_change

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=tuple(updated_trades),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _rebalance_position(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Adjust position quantity by delta (positive = add, negative = reduce).

    Requires signal.position_id to target the exact position.
    Signal.qty is the DELTA, not the absolute quantity:
      - +qty: add to position (buy more)
      - -qty: reduce position (sell some or all)

    When the resulting qty <= 0, the position is fully closed.
    When the resulting qty > 0, the position is adjusted — entry_price
    is NOT updated (simple average, not time-weighted). Trade record
    is closed and a new one opened to track the adjusted position.
    """
    symbol = fill.signal.symbol
    pid = fill.signal.position_id

    positions_tuple = portfolio.positions.get(symbol)

    if not positions_tuple:
        # No existing position — treat as open if delta positive
        if fill.signal.qty > 0:
            return _open_position(portfolio, fill)
        return portfolio

    if not pid:
        raise ValueError(
            f"_rebalance_position requires position_id on TradeSignal. "
            f"Signal for {symbol} has no position_id set "
            f"but positions exist: {[p.position_id for p in positions_tuple]}"
        )

    # Find target position by position_id
    target_idx: Optional[int] = None
    target: Optional[Position] = None
    for i, pos in enumerate(positions_tuple):
        if pos.position_id == pid:
            target_idx = i
            target = pos
            break

    if target is None:
        raise ValueError(
            f"Position {pid} not found for symbol {symbol}. "
            f"Available: {[p.position_id for p in positions_tuple]}"
        )

    delta = round(fill.signal.qty, 4)
    if delta == 0:
        return portfolio

    new_qty = round(target.qty + delta, 4)

    if new_qty <= 0:
        # Full close — build a close fill for the remaining position
        close_fill = FillEvent(
            signal=TradeSignal(
                action=ActionType.close,
                symbol=fill.signal.symbol,
                timestamp=fill.signal.timestamp,
                price=fill.executed_price,
                qty=abs(target.qty),
                reason=fill.signal.reason,
                position_id=pid,
            ),
            filled_qty=abs(target.qty),
            executed_price=fill.executed_price,
            commission=fill.commission,
            slippage=fill.slippage,
            timestamp=fill.timestamp,
        )
        return _close_position(portfolio, close_fill)

    # Partial adjustment: close old trade, open new trade with adjusted qty
    is_long = target.type == ActionType.long

    if delta > 0:
        # Adding: cash goes out
        cash_change = -(delta * fill.executed_price + fill.commission)
        pnl_on_closed = 0.0  # no PnL realized on add — old position continues
    else:
        # Reducing: cash comes in, realize partial PnL
        reduce_qty = abs(delta)
        if is_long:
            partial_pnl = (fill.executed_price - target.entry_price) * reduce_qty
            cash_change = reduce_qty * fill.executed_price - fill.commission
        else:
            partial_pnl = (target.entry_price - fill.executed_price) * reduce_qty
            cash_change = (
                (reduce_qty * target.entry_price) + partial_pnl - fill.commission
            )
        pnl_on_closed = partial_pnl

    new_cash = portfolio.cash + cash_change

    # Close the old trade with its partial PnL
    updated_trades = list(portfolio.trades)
    for i, trade in enumerate(updated_trades):
        if (
            trade.status == TradeStatus.open
            and trade.position_id == pid
            and trade.symbol == symbol
        ):
            updated_trades[i] = _close_trade(trade, fill, pnl_on_closed)
            break

    # Open a new trade/position for the remaining qty
    new_pid = f"{symbol}_{fill.timestamp.timestamp()}"

    new_position = Position(
        symbol=symbol,
        qty=new_qty,
        entry_price=target.entry_price,  # preserve original entry price
        entry_time=target.entry_time,  # preserve original entry time
        stop_loss=target.stop_loss,
        take_profit=target.take_profit,
        last_price=fill.executed_price,
        type=target.type,
        position_id=new_pid,
    )

    new_trade = Trade(
        entry_time=target.entry_time,
        entry_price=target.entry_price,
        exit_time=None,
        exit_price=None,
        last_price=fill.executed_price,
        symbol=symbol,
        position=target.type,
        qty=new_qty,
        stop_loss=target.stop_loss or 0.0,
        take_profit=target.take_profit or 0.0,
        pnl=0.0,
        reason=fill.signal.reason,
        status=TradeStatus.open,
        close_reason=None,
        position_id=new_pid,
    )

    # Replace old position with new in the tuple
    assert target_idx is not None
    new_positions = dict(portfolio.positions)
    new_tup = (
        positions_tuple[:target_idx]
        + (new_position,)
        + positions_tuple[target_idx + 1 :]
    )
    new_positions[symbol] = new_tup

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=tuple(updated_trades + [new_trade]),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _close_trade(trade: Trade, fill: FillEvent, pnl: float) -> Trade:
    """Return trade closed by fill."""
    return Trade(
        entry_time=trade.entry_time,
        entry_price=trade.entry_price,
        exit_time=fill.timestamp,
        exit_price=fill.executed_price,
        last_price=fill.executed_price,
        reason=trade.reason,
        symbol=trade.symbol,
        position=trade.position,
        qty=trade.qty,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        pnl=pnl,
        status=TradeStatus.closed,
        close_reason=fill.signal.reason or TradeExitReason.none,
        position_id=trade.position_id,
    )


def update_prices(portfolio: PortfolioState, tick) -> PortfolioState:
    """Update position prices and equity curve with new tick."""
    symbol = tick.symbol
    new_positions = dict(portfolio.positions)

    # Update all positions for the tick's symbol
    symbol_positions = portfolio.positions.get(symbol)
    if symbol_positions:
        updated = tuple(
            Position(
                symbol=pos.symbol,
                qty=pos.qty,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                last_price=tick.close,
                type=pos.type,
                position_id=pos.position_id,
            )
            for pos in symbol_positions
        )
        new_positions[symbol] = updated

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


def calculate_positions_value(positions: Dict[str, TupleT[Position, ...]]) -> float:
    """Calculate total value of all positions across all symbols."""
    value = 0.0
    for positions_tuple in positions.values():
        for position in positions_tuple:
            if position.type == ActionType.long:
                value += position.qty * position.last_price
            else:
                upnl = position.qty * (position.entry_price - position.last_price)
                value += (position.qty * position.entry_price) + upnl
    return value


def calculate_equity(portfolio: PortfolioState) -> float:
    """Calculate total equity."""
    return portfolio.cash + calculate_positions_value(portfolio.positions)


# ---------------------------------------------------------------------------
# Multi-position helpers — iterate all positions across all symbols
# ---------------------------------------------------------------------------


def iter_positions(
    portfolio: PortfolioState,
) -> Dict[str, TupleT[Position, ...]]:
    """Return positions dict directly — iterate via .items(), .values(), .keys().

    Each value is Tuple[Position, ...].
    """
    return portfolio.positions


def count_positions(portfolio: PortfolioState) -> int:
    """Total number of individual positions across all symbols."""
    return sum(len(tup) for tup in portfolio.positions.values())


def get_symbol_positions(
    portfolio: PortfolioState, symbol: str
) -> TupleT[Position, ...]:
    """Return all positions for a symbol. Empty tuple if none."""
    return portfolio.positions.get(symbol, ())
