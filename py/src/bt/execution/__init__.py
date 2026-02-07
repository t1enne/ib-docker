from typing import Optional

from src.bt.types import TradeSignal, Tick, ActionType, FillEvent, ExecutionParams


class ExecutionHandler:
    """Models realistic order execution with spread and adverse selection slippage."""

    def __init__(self, params: Optional[ExecutionParams] = None):
        self.params = params or ExecutionParams()

    def execute(self, signal: TradeSignal, tick: Tick) -> FillEvent:
        """
        Convert a trade signal into a fill event with realistic pricing.

        Args:
            signal: The trade signal to execute
            tick: The current market tick

        Returns:
            FillEvent with executed price including spread and slippage
        """
        signal_price = signal.price
        base_spread = signal_price * (self.params.spread_bps / 10000)

        if signal.action == ActionType.long:
            base_price = signal_price + base_spread
        elif signal.action == ActionType.short:
            base_price = signal_price - base_spread
        else:
            base_price = signal_price

        adverse_selection = self._calculate_adverse_selection(signal, tick)

        slippage_bps = self.params.slippage_bps * (1.5 if adverse_selection else 1.0)
        slippage_price = signal_price * (slippage_bps / 10000)

        executed_price = base_price + slippage_price

        commission = self._calculate_commission(signal, executed_price)

        return FillEvent(
            signal=signal,
            filled_qty=1.0,
            executed_price=executed_price,
            commission=commission,
            slippage=slippage_price,
        )

    def _calculate_adverse_selection(self, signal: TradeSignal, tick: Tick) -> bool:
        """
        Determine if slippage should be adverse.

        Adverse selection occurs when price moves unfavorably:
        - Long entry: price went down (close < open)
        - Short entry: price went up (close > open)
        """
        price_move = tick.close - tick.open
        percent_move = price_move / tick.open if tick.open != 0 else 0

        if signal.action == ActionType.long:
            return percent_move < -0.001
        elif signal.action == ActionType.short:
            return percent_move > 0.001
        return False

    def _calculate_commission(
        self, signal: TradeSignal, executed_price: float
    ) -> float:
        """Calculate commission based on notional value."""
        return executed_price * 0.0001


__all__ = ["ExecutionHandler"]
