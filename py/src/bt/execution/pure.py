"""Pure execution functions."""

from src.bt.state.types import (
    TradeSignal,
    Candle,
    FillEvent,
    ExecutionParams,
    ActionType,
)


def execute_signal(
    signal: TradeSignal, tick: Candle, params: ExecutionParams
) -> FillEvent:
    """Convert signal to fill with slippage/spread.

    Pure function - no side effects.

    When signal.fill_at_next_open is True, uses tick.open as the base price
    (realistic: signals generated at close fill at next bar's open).
    Otherwise uses signal.price (same-bar fill at signal generation price).
    """
    base_price = tick.open if signal.fill_at_next_open else signal.price
    spread_bps = params.spread_bps or 0.01
    base_spread = base_price * (spread_bps / 10000)

    # Calculate base price with spread
    if signal.action == ActionType.long:
        fill_base = base_price + base_spread
    elif signal.action == ActionType.short:
        fill_base = base_price - base_spread
    else:
        fill_base = base_price

    # Calculate slippage
    adverse = calculate_adverse_selection(signal, tick)
    slippage_bps = params.slippage_bps * (1.5 if adverse else 1.0)
    slippage = base_price * (slippage_bps / 10000)

    executed_price = fill_base + slippage
    commission = params.fixed_commission

    return FillEvent(
        signal=signal,
        filled_qty=signal.qty if signal.qty > 0 else 1.0,
        executed_price=executed_price,
        commission=commission,
        slippage=slippage,
        timestamp=tick.timestamp,
    )


def execute_risk_event(event, tick: Candle, params: ExecutionParams) -> FillEvent:
    """Convert risk event (SL/TP) into fill event."""
    base_spread = event.trigger_price * (params.spread_bps / 10000)
    base_price = event.trigger_price - base_spread

    # Risk events often have worse slippage
    slippage_bps = params.slippage_bps * 2.0
    slippage = event.trigger_price * (slippage_bps / 10000)
    executed_price = base_price - slippage

    signal = TradeSignal(
        action=ActionType.close,
        symbol=event.symbol,
        timestamp=event.timestamp,
        price=event.trigger_price,
        reason=event.reason,
    )

    return FillEvent(
        signal=signal,
        filled_qty=1.0,
        executed_price=executed_price,
        commission=params.fixed_commission,
        slippage=slippage,
        timestamp=tick.timestamp,
    )


def calculate_adverse_selection(signal: TradeSignal, tick: Candle) -> bool:
    """Determine if slippage should be adverse."""
    price_move = tick.close - tick.open
    percent_move = price_move / tick.open if tick.open != 0 else 0

    if signal.action == ActionType.long:
        return percent_move < -0.001
    elif signal.action == ActionType.short:
        return percent_move > 0.001
    return False
