"""Pure risk management functions."""

from typing import Tuple, Optional, Union

from src.bt.state.types import (
    PortfolioState,
    Position,
    Tick,
    StopLossEvent,
    TakeProfitEvent,
    RiskConfig,
    ActionType,
)


def check_risk(portfolio: PortfolioState, tick: Tick, config: RiskConfig) -> Tuple:
    """Check if any positions hit risk limits.

    Returns tuple of risk events (StopLossEvent, TakeProfitEvent, etc.)
    """
    events = []

    position = portfolio.positions.get(tick.symbol)
    if position:
        event = check_position_risk(position, tick, config)
        if event:
            events.append(event)

    return tuple(events)


def check_position_risk(
    position: Position, tick: Tick, config: RiskConfig
) -> Optional[Union[StopLossEvent, TakeProfitEvent]]:
    """Check single position for SL/TP triggers."""
    is_long = position.qty > 0

    # Check stop loss
    if is_long and position.stop_loss and tick.close <= position.stop_loss:
        return StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close,
            reason="sl",
        )

    if not is_long and position.stop_loss and tick.close >= position.stop_loss:
        return StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close,
            reason="sl",
        )

    # Check take profit
    if is_long and position.take_profit and tick.close >= position.take_profit:
        return TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close,
            reason="tp",
        )

    if not is_long and position.take_profit and tick.close <= position.take_profit:
        return TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close,
            reason="tp",
        )

    return None


def update_trailing_stop(
    position: Position, tick: Tick, config: RiskConfig
) -> Position:
    """Update trailing stop, return new position."""
    if not config.trailing_stop:
        return position

    is_long = position.qty > 0

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
            )

    return position
