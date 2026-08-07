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
    """Convert risk event (SL/TP) into fill event, modeling intra-bar gaps.

    A stop-loss/take-profit event fires because the bar's high/low crossed
    the trigger level. When the bar *gaps through* the level, the real fill
    is not the trigger price but the worse-of-trigger-and-open price:

    - Long stop  (trigger when price falls to ``trigger``): a bar that opens
      below the stop has already gapped past it downward — fill at the open.
    - Short stop (trigger when price rises to ``trigger``): a bar that opens
      above the stop has gapped past it upward — fill at the open.
    - Take-profit gaps go *favorably* through the level: a conservative fill
      takes at least the trigger, and at the gap-open when the open is already
      beyond it (e.g. a long TP gapped open above the target is filled at the
      higher open).

    Without this, a stop that is gapped through is happily filled at the
    trigger every time, which systematically overstates P&L on gap days.
    """
    pid = getattr(event, "position_id", "")
    qty = getattr(event, "position_qty", 0.0)
    position_type = getattr(event, "position_type", None)

    trigger = event.trigger_price
    is_stop = getattr(event, "reason", "") == "sl"
    is_long = position_type == ActionType.long

    # Determine the fill base price, accounting for a gap through the level.
    if is_stop:
        # Adverse direction: the loss-worse side of the trigger.
        if is_long:
            # Long stop: an open below the stop means we filled at the gap-open
            # (worse than trigger). Guard against tick.open falling below trigger.
            fill_base = min(trigger, tick.open)
        else:
            # Short stop: an open above the stop means we filled at the gap-open
            # (worse than trigger, i.e. higher buy-back price = bigger loss).
            fill_base = max(trigger, tick.open)
    else:
        # Take-profit: favorable direction. Take at least the trigger; if the
        # open already gapped past it favorably, capture the better open.
        if is_long:
            # Long TP: open above target is a favorable gap.
            fill_base = max(trigger, tick.open)
        else:
            # Short TP: open below target is a favorable gap.
            fill_base = min(trigger, tick.open)

    # Apply spread + adverse slippage on top of the resolved base price.
    # Direction matters: a long closes *by selling* (receive less), a short
    # closes *by buying to cover* (pay more).
    base_spread = fill_base * (params.spread_bps / 10000)
    slippage_bps = params.slippage_bps * 2.0
    slippage = fill_base * (slippage_bps / 10000)
    if is_long:
        executed_price = fill_base - base_spread - slippage
    else:
        executed_price = fill_base + base_spread + slippage

    signal = TradeSignal(
        action=ActionType.close,
        symbol=event.symbol,
        timestamp=event.timestamp,
        price=event.trigger_price,
        reason=event.reason,
        position_id=pid,
    )

    return FillEvent(
        signal=signal,
        filled_qty=qty,
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
