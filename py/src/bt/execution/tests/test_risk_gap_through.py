"""Gap-through fills for stop-loss / take-profit risk events.

A rule-based single-price executor must not fill a stop that the bar gapped
*through* at the trigger price every time — that is systematically optimistic.
When the bar opens beyond the trigger in the adverse/favorable direction, the
real fill is the open. These tests lock in that behavior.
"""

from src.bt.execution import execute_risk_event, create_execution_params
from src.bt.state import (
    Candle,
    StopLossEvent,
    TakeProfitEvent,
    ActionType,
)
from src.utils import get_ts


def _candle(open_, high, low, close):
    return Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000,
    )


def _params(spread=5.0, slip=2.0, commission=0.5):
    return create_execution_params(
        spread_bps=spread, slippage_bps=slip, fixed_commission=commission
    )


# ── Gap through a long stop ────────────────────────────────────────────
def test_long_stop_gaps_down_to_open():
    """Long stop at 90, bar opens at 85 (below stop) — fill at the open (worse)."""
    event = StopLossEvent(
        symbol="AAPL",
        timestamp=get_ts("2025-01-02"),
        trigger_price=90.0,
        reason="sl",
        position_id="p1",
        position_qty=10.0,
        position_type=ActionType.long,
    )
    fill = execute_risk_event(
        event, _candle(open_=85.0, high=88.0, low=84.0, close=86.0), _params()
    )
    # Long closes by selling → fills below the open (spread+slippage).
    assert fill.executed_price < 85.0
    assert fill.executed_price < event.trigger_price  # worse than stop, not equal


def test_long_stop_no_gap_fills_at_trigger():
    """Long stop at 90, bar opens at 95 but lows cross 90 — fill at the stop."""
    event = StopLossEvent(
        symbol="AAPL",
        timestamp=get_ts("2025-01-02"),
        trigger_price=90.0,
        reason="sl",
        position_id="p1",
        position_qty=10.0,
        position_type=ActionType.long,
    )
    fill = execute_risk_event(
        event, _candle(open_=95.0, high=96.0, low=89.0, close=91.0), _params()
    )
    assert fill.executed_price < 90.0  # near trigger, spread+slippage off it


# ── Gap through a short stop ───────────────────────────────────────────
def test_short_stop_gaps_up_to_open():
    """Short stop at 110, bar opens at 115 (above stop) — fill at the open (worse)."""
    event = StopLossEvent(
        symbol="AAPL",
        timestamp=get_ts("2025-01-02"),
        trigger_price=110.0,
        reason="sl",
        position_id="p1",
        position_qty=10.0,
        position_type=ActionType.short,
    )
    fill = execute_risk_event(
        event, _candle(open_=115.0, high=118.0, low=114.0, close=117.0), _params()
    )
    # Short closes by buying to cover → pays above the open (spread+slippage).
    assert fill.executed_price > 115.0
    assert fill.executed_price > event.trigger_price  # worse than stop


# ── Gap through take-profit (favorable) ────────────────────────────────
def test_long_tp_gaps_up_to_open():
    """Long TP at 120, bar opens at 125 (above target) — capture the better open."""
    event = TakeProfitEvent(
        symbol="AAPL",
        timestamp=get_ts("2025-01-02"),
        trigger_price=120.0,
        reason="tp",
        position_id="p1",
        position_qty=10.0,
        position_type=ActionType.long,
    )
    fill = execute_risk_event(
        event, _candle(open_=125.0, high=127.0, low=124.0, close=126.0), _params()
    )
    assert fill.executed_price < 125.0  # sells beneath the favorable open
    assert fill.executed_price > 120.0  # but better than the trigger


def test_short_tp_gaps_down_to_open():
    """Short TP at 80, bar opens at 75 (below target) — capture the better open."""
    event = TakeProfitEvent(
        symbol="AAPL",
        timestamp=get_ts("2025-01-02"),
        trigger_price=80.0,
        reason="tp",
        position_id="p1",
        position_qty=10.0,
        position_type=ActionType.short,
    )
    fill = execute_risk_event(
        event, _candle(open_=75.0, high=77.0, low=73.0, close=74.0), _params()
    )
    assert fill.executed_price > 75.0  # short covers by buying → above the open
    assert fill.executed_price < 80.0  # but better than the trigger


# ── End-to-end through check_position_risk + execute_risk_event ────────
def test_gap_stop_through_full_risk_flow():
    """Long position stopped out on a gap-down open — fill reflects the open."""
    from src.bt.risk import check_position_risk, create_risk_config
    from src.bt.state import Position

    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
        sl_explicit=True,
        tp_explicit=True,
    )
    tick = _candle(open_=85.0, high=88.0, low=84.0, close=86.0)
    cfg = create_risk_config(stop_loss_pct=0.1, take_profit_pct=0.2)

    new_pos, event = check_position_risk(position, tick, cfg)
    assert isinstance(event, StopLossEvent)
    assert event.position_type == ActionType.long  # direction threaded through

    fill = execute_risk_event(event, tick, _params())
    assert fill.executed_price < event.trigger_price  # gap-down, not at stop
