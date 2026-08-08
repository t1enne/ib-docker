"""Pure risk management functions."""

from typing import Tuple, Optional, Union

from src.bt.state.types import (
    PortfolioState,
    Position,
    Candle,
    StopLossEvent,
    TakeProfitEvent,
    RiskConfig,
    ActionType,
)


def check_risk(
    portfolio: PortfolioState, tick: Candle, config: RiskConfig
) -> Tuple[Tuple, PortfolioState]:
    """Check positions for SL/TP triggers and update trailing stops.

    Returns (risk_events, updated_portfolio). The updated portfolio
    carries persisted SL/TP levels (initialised or trailed) even when
    no risk event fires.
    """
    positions_tuple = portfolio.positions.get(tick.symbol, ())
    if not positions_tuple:
        return (), portfolio

    events: list = []
    updated_positions: list[Position] = []
    dirty = False

    for position in positions_tuple:
        new_pos, event = check_position_risk(position, tick, config)
        updated_positions.append(new_pos)
        if new_pos is not position:
            dirty = True
        if event:
            events.append(event)

    new_portfolio = portfolio
    if dirty:
        new_positions = dict(portfolio.positions)
        new_positions[tick.symbol] = tuple(updated_positions)
        new_portfolio = PortfolioState(
            cash=portfolio.cash,
            positions=new_positions,
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            initial_capital=portfolio.initial_capital,
        )

    return tuple(events), new_portfolio


def check_position_risk(
    position: Position, tick: Candle, config: RiskConfig
) -> Tuple[Position, Optional[Union[StopLossEvent, TakeProfitEvent]]]:
    """Check single position for SL/TP triggers.

    SL/TP levels are strategy-owned (set per-trade on ``TradeSignal`` and stored
    on the ``Position``). There is no config-level fallback. Returns
    ``(updated_position, event_or_none)``.
    """
    # Trail the stop if configured. Levels are strategy-owned and fixed; if
    # trailing_stop is enabled it moves the stop with price. Default is off.
    if config.trailing_stop:
        position = _trail_stop(position, tick, config)

    pid = position.position_id
    is_long = position.type == ActionType.long

    # Check stop loss
    if is_long and position.stop_loss and tick.low <= position.stop_loss:
        return position, StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=position.stop_loss,
            reason="sl",
            position_id=pid,
            position_qty=abs(position.qty),
            position_type=ActionType.long,
        )

    if not is_long and position.stop_loss and tick.high >= position.stop_loss:
        return position, StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=position.stop_loss,
            reason="sl",
            position_id=pid,
            position_qty=abs(position.qty),
            position_type=ActionType.short,
        )

    # Check take profit
    if is_long and position.take_profit and tick.high >= position.take_profit:
        return position, TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=position.take_profit,
            reason="tp",
            position_id=pid,
            position_qty=abs(position.qty),
            position_type=ActionType.long,
        )

    if not is_long and position.take_profit and tick.low <= position.take_profit:
        return position, TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=position.take_profit,
            reason="tp",
            position_id=pid,
            position_qty=abs(position.qty),
            position_type=ActionType.short,
        )

    return position, None


def _trail_stop(position: Position, tick: Candle, config: RiskConfig) -> Position:
    """Move trailing stop if favourable price movement warrants it."""
    is_long = position.type == ActionType.long

    if config.stop_loss_pct <= 0:
        return position  # trailing stop disabled

    if is_long:
        new_stop = tick.high * (1 - config.stop_loss_pct)
        if position.stop_loss is None or new_stop > position.stop_loss:
            return Position(
                symbol=position.symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                stop_loss=new_stop,
                take_profit=position.take_profit,
                last_price=position.last_price,
                type=position.type,
                position_id=position.position_id,
            )
    else:
        new_stop = tick.low * (1 + config.stop_loss_pct)
        if position.stop_loss is None or new_stop < position.stop_loss:
            return Position(
                symbol=position.symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                stop_loss=new_stop,
                take_profit=position.take_profit,
                last_price=position.last_price,
                type=position.type,
                position_id=position.position_id,
            )

    return position


def update_trailing_stop(
    position: Position, tick: Candle, config: RiskConfig
) -> Position:
    """Legacy — update trailing stop, return new position.

    Prefer check_position_risk() for the full risk-check flow.
    """
    if not config.trailing_stop:
        return position
    return _trail_stop(position, tick, config)
