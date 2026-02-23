from typing import List, Optional, Literal, Union, Any
from src.bt.state import (
    BacktestState,
    TradeSignal,
    ActionType,
    Tick,
    Position,
)


def close(
    tick: Tick,
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
            reason=reason,
        )
    ]


def open(
    tick: Tick, dir: ActionType, z: Optional[float], hedge: Optional[float] = None
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=dir,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            hedge_beta=hedge,
        ),
    ]
