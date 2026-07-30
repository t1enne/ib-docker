from typing import List, Optional, Any

from src.bt.state import (
    TradeSignal,
    ActionType,
    Candle,
    Position,
)


def close(
    tick: Candle,
    position: Position,
    reason: Any,
    z: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=ActionType.close,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            qty=abs(position.qty),
            position_id=position.position_id,
            reason=reason,
        )
    ]


def open(
    tick: Candle,
    dir: ActionType,
    qty: float,
    reason: Optional[str] = "",
    hedge: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=dir,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            qty=qty,
            reason=reason,
            hedge_beta=hedge,
        ),
    ]
