from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Optional

from src.bt.types import (
    Tick,
    Trade,
    ActionType,
    StopLossEvent,
    TakeProfitEvent,
    RiskEvent,
)


@dataclass
class RiskManagerProps:
    stop_loss_pct: float
    take_profit_pct: float


class RiskManager:
    """Monitors positions for SL/TP triggers and generates risk events."""

    def __init__(self, props: RiskManagerProps):
        self.stop_loss_pct = props.stop_loss_pct
        self.take_profit_pct = props.take_profit_pct
        self._open_trades: Dict[str, Trade] = {}

    def update_trades(self, open_trades: Dict[str, Trade]):
        """Update reference to open trades from portfolio."""
        self._open_trades = open_trades

    def on_tick(self, tick: Tick) -> List[RiskEvent]:
        """Check if any open position hit SL/TP. Returns all triggered events."""
        events: List[RiskEvent] = []

        if tick.symbol not in self._open_trades:
            return events

        trade = self._open_trades[tick.symbol]

        is_long = trade.position == ActionType.long
        has_sl = trade.stop_loss > 0
        has_tp = trade.take_profit > 0

        long_sl_triggered = is_long and has_sl and tick.close <= trade.stop_loss
        long_tp_triggered = is_long and has_tp and tick.close >= trade.take_profit
        short_sl_triggered = not is_long and has_sl and tick.close >= trade.stop_loss
        short_tp_triggered = not is_long and has_tp and tick.close <= trade.take_profit

        if long_sl_triggered or short_sl_triggered:
            events.append(
                StopLossEvent(
                    symbol=tick.symbol,
                    timestamp=tick.timestamp,
                    trigger_price=tick.close,
                    reason="sl",
                )
            )

        if long_tp_triggered or short_tp_triggered:
            events.append(
                TakeProfitEvent(
                    symbol=tick.symbol,
                    timestamp=tick.timestamp,
                    trigger_price=tick.close,
                    reason="tp",
                )
            )

        if not events:
            self._update_trailing_sl(trade, tick)

        return events

    def _update_trailing_sl(self, trade: Trade, tick: Tick):
        """Update trailing stop loss based on tick prices."""
        is_long = trade.position == ActionType.long

        if is_long:
            highest_high = max(trade.entry_price, tick.high)
            trade.stop_loss = round(highest_high * (1 - self.stop_loss_pct), 2)
        else:
            lowest_low = min(trade.entry_price, tick.low)
            trade.stop_loss = round(lowest_low * (1 + self.stop_loss_pct), 2)


__all__ = [
    "RiskManager",
    "RiskManagerProps",
    "RiskEvent",
    "StopLossEvent",
    "TakeProfitEvent",
]
