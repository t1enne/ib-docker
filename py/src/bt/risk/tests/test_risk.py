"""Tests for risk checks — stop loss, take profit, trailing stop."""

import pytest
from src.bt.risk import (
    check_risk,
    check_position_risk,
    update_trailing_stop,
    create_risk_config,
)
from src.bt.state import (
    Position,
    Candle,
    ActionType,
    StopLossEvent,
    TakeProfitEvent,
    RiskConfig,
)
from src.utils import get_ts


@pytest.fixture
def risk_config():
    return create_risk_config(stop_loss_pct=0.1, take_profit_pct=0.2)


def test_stop_loss_long(risk_config):
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=98.0,
        low=85.0,
        close=88.0,
        volume=1000,
    )
    event = check_position_risk(position, tick, risk_config)
    assert isinstance(event, StopLossEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 90.0


def test_take_profit_short(risk_config):
    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=110.0,
        take_profit=80.0,
        last_price=100.0,
        type=ActionType.short,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=85.0,
        high=90.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )
    event = check_position_risk(position, tick, risk_config)
    assert isinstance(event, TakeProfitEvent)
    assert event.trigger_price == 80.0


def test_trailing_sl_moves_up_long():
    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=True)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    new_pos = update_trailing_stop(position, tick, config)
    assert new_pos.stop_loss
    assert position.stop_loss
    assert new_pos.stop_loss > position.stop_loss


def test_no_risk_on_different_symbol(risk_config):
    from src.bt.state import PortfolioState, EquityPoint

    position = Position(
        symbol="GOOGL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=10000,
        positions={"GOOGL": (position,)},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=get_ts("2025-01-01"),
                equity=10000,
                cash=10000,
                positions_value=0.0,
            ),
        ),
        initial_capital=10000,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=80.0,
        high=85.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )
    assert check_risk(portfolio, tick, risk_config) == ()
