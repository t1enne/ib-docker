from typing import List, Optional, Any, Tuple

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
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
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
            stop_loss=stop_loss,
            take_profit=take_profit,
        ),
    ]


# ---------------------------------------------------------------------------
# shared sizing + SL/TP helpers — per-trade sizing is strategy-owned and
# driven by ``strategy_params`` (position_size / stop_loss / take_profit).
# ---------------------------------------------------------------------------


def sized_qty(cash: float, position_size: float, price: float) -> float:
    """Share quantity for ``position_size`` fraction of ``cash`` at ``price``.

    ``position_size`` is a 0-1 fraction of cash deployed per position/trade.
    """
    if cash <= 0 or price <= 0 or position_size <= 0:
        return 0.0
    return round(cash * position_size / price, 4)


def sl_tp_from_pct(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    is_long: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute absolute per-trade SL/TP prices from percentage params.

    ``stop_loss`` / ``take_profit`` are fractional pcts (e.g. 0.05 = 5%). A
    pct <= 0 disables that leg (returns None so the position has no SL/TP).
    Returns ``(stop_loss_price, take_profit_price)``.
    """
    if is_long:
        sl = entry_price * (1 - stop_loss) if stop_loss > 0 else None
        tp = entry_price * (1 + take_profit) if take_profit > 0 else None
    else:
        sl = entry_price * (1 + stop_loss) if stop_loss > 0 else None
        tp = entry_price * (1 - take_profit) if take_profit > 0 else None
    return sl, tp
