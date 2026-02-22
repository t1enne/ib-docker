import pytest
from src.bt.risk import (
    check_risk,
    check_position_risk,
    update_trailing_stop,
    create_risk_config,
)
from src.bt.state import Position, Tick, ActionType, StopLossEvent, TakeProfitEvent
from src.utils import get_ts


@pytest.fixture
def risk_config():
    return create_risk_config(
        stop_loss_pct=0.1,
        take_profit_pct=0.2,
    )


@pytest.fixture
def portfolio_state():
    """Helper to create a portfolio state with positions."""
    from src.bt.state import PortfolioState, EquityPoint

    return PortfolioState(
        cash=10000,
        positions={},
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


def test_no_trades_returns_empty(risk_config, portfolio_state):
    """No positions should return no risk events."""
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        volume=1000,
    )
    events = check_risk(portfolio_state, tick, risk_config)
    assert events == ()


def test_stop_loss_triggered_long(risk_config):
    """Long position hitting stop loss should trigger event."""
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
    )

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=98.0,
        low=85.0,
        close=88.0,
        volume=1000,
    )

    event = check_position_risk(position, tick, risk_config)

    assert event is not None
    assert isinstance(event, StopLossEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 88.0


def test_take_profit_triggered_long(risk_config):
    """Long position hitting take profit should trigger event."""
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
    )

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=115.0,
        high=125.0,
        low=110.0,
        close=122.0,
        volume=1000,
    )

    event = check_position_risk(position, tick, risk_config)

    assert event is not None
    assert isinstance(event, TakeProfitEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 122.0


def test_stop_loss_triggered_short(risk_config):
    """Short position hitting stop loss should trigger event."""
    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=110.0,
        take_profit=80.0,
        last_price=100.0,
    )

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=108.0,
        close=112.0,
        volume=1000,
    )

    event = check_position_risk(position, tick, risk_config)

    assert event is not None
    assert isinstance(event, StopLossEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 112.0


def test_take_profit_triggered_short(risk_config):
    """Short position hitting take profit should trigger event."""
    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=110.0,
        take_profit=80.0,
        last_price=100.0,
    )

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=85.0,
        high=90.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )

    event = check_position_risk(position, tick, risk_config)

    assert event is not None
    assert isinstance(event, TakeProfitEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 78.0


def test_no_trailing_sl_update_when_sl_triggered(risk_config):
    """Trailing SL should not update when SL is already triggered."""
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
    )

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=98.0,
        low=85.0,
        close=88.0,  # Below SL
        volume=1000,
    )

    # Check if SL is triggered first
    event = check_position_risk(position, tick, risk_config)
    assert event is not None
    assert isinstance(event, StopLossEvent)


def test_trailing_sl_updates_for_long(risk_config):
    """Trailing stop loss should move up for long positions."""
    from src.bt.state import RiskConfig

    # Create config with trailing stop enabled
    config_with_trailing = RiskConfig(
        stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=True
    )

    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
    )

    # Price goes up
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )

    new_position = update_trailing_stop(position, tick, config_with_trailing)

    # Trailing SL should have moved up
    assert new_position.stop_loss
    assert position.stop_loss
    assert new_position.stop_loss > position.stop_loss


def test_trailing_sl_updates_for_short(risk_config):
    """Trailing stop loss should move down for short positions."""
    from src.bt.state import RiskConfig

    # Create config with trailing stop enabled
    config_with_trailing = RiskConfig(
        stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=True
    )

    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=110.0,
        take_profit=80.0,
        last_price=100.0,
    )

    # Price goes down
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=96.0,
        low=85.0,
        close=90.0,
        volume=1000,
    )

    new_position = update_trailing_stop(position, tick, config_with_trailing)

    # Trailing SL should have moved down
    assert new_position.stop_loss
    assert position.stop_loss
    assert new_position.stop_loss < position.stop_loss


def test_different_symbol_no_trigger(risk_config, portfolio_state):
    """Risk check for different symbol should not trigger."""
    position = Position(
        symbol="GOOGL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
    )

    # Create portfolio with GOOGL position
    from src.bt.state import PortfolioState, EquityPoint

    portfolio = PortfolioState(
        cash=10000,
        positions={"GOOGL": position},
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

    # Tick for different symbol
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=80.0,  # Would trigger if it was GOOGL
        high=85.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )

    events = check_risk(portfolio, tick, risk_config)

    # Should be empty since tick is for AAPL but position is GOOGL
    assert events == ()
