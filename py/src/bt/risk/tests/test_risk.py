import pytest
from src.bt.risk import RiskManager, RiskManagerProps
from src.bt.types import Trade, Tick, ActionType, StopLossEvent, TakeProfitEvent
from src.utils import get_ts


@pytest.fixture
def risk_manager():
    return RiskManager(
        RiskManagerProps(
            stop_loss_pct=0.1,
            take_profit_pct=0.2,
        )
    )


def test_no_trades_returns_empty(risk_manager: RiskManager):
    risk_manager.update_trades({})
    events = risk_manager.on_tick(
        Tick(
            timestamp=get_ts("2025-01-02"),
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=1000,
        )
    )
    assert events == []


def test_stop_loss_triggered_long(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=2.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=120.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=98.0,
        low=85.0,
        close=88.0,
        volume=1000,
    )
    events = risk_manager.on_tick(tick)

    assert len(events) == 1
    assert isinstance(events[0], StopLossEvent)
    assert events[0].symbol == "AAPL"
    assert events[0].trigger_price == 88.0


def test_take_profit_triggered_long(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=2.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=120.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=115.0,
        high=125.0,
        low=110.0,
        close=122.0,
        volume=1000,
    )
    events = risk_manager.on_tick(tick)

    assert len(events) == 1
    assert isinstance(events[0], TakeProfitEvent)
    assert events[0].symbol == "AAPL"
    assert events[0].trigger_price == 122.0


def test_stop_loss_triggered_short(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=-2.0,
        symbol="AAPL",
        position=ActionType.short,
        qty=10.0,
        stop_loss=110.0,
        take_profit=80.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=102.0,
        close=112.0,
        volume=1000,
    )
    events = risk_manager.on_tick(tick)

    assert len(events) == 1
    assert isinstance(events[0], StopLossEvent)
    assert events[0].symbol == "AAPL"
    assert events[0].trigger_price == 112.0


def test_take_profit_triggered_short(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=-2.0,
        symbol="AAPL",
        position=ActionType.short,
        qty=10.0,
        stop_loss=110.0,
        take_profit=80.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=90.0,
        high=95.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )
    events = risk_manager.on_tick(tick)

    assert len(events) == 1
    assert isinstance(events[0], TakeProfitEvent)
    assert events[0].symbol == "AAPL"
    assert events[0].trigger_price == 78.0


def test_no_trailing_sl_update_when_sl_triggered(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=2.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=150.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick1 = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=120.0,
        low=98.0,
        close=115.0,
        volume=1000,
    )
    events1 = risk_manager.on_tick(tick1)

    assert len(events1) == 0
    assert trade.stop_loss == 108.0

    tick2 = Tick(
        timestamp=get_ts("2025-01-03"),
        symbol="AAPL",
        open=115.0,
        high=118.0,
        low=85.0,
        close=88.0,
        volume=1000,
    )
    events2 = risk_manager.on_tick(tick2)

    assert len(events2) == 1
    assert isinstance(events2[0], StopLossEvent)
    assert events2[0].trigger_price == 88.0
    assert trade.stop_loss == 108.0


def test_trailing_sl_updates_for_long(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=2.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=150.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick1 = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=120.0,
        low=98.0,
        close=115.0,
        volume=1000,
    )
    risk_manager.on_tick(tick1)

    assert trade.stop_loss == 108.0

    tick2 = Tick(
        timestamp=get_ts("2025-01-03"),
        symbol="AAPL",
        open=115.0,
        high=130.0,
        low=110.0,
        close=125.0,
        volume=1000,
    )
    risk_manager.on_tick(tick2)

    assert trade.stop_loss == 117.0


def test_trailing_sl_updates_for_short(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=-2.0,
        symbol="AAPL",
        position=ActionType.short,
        qty=10.0,
        stop_loss=110.0,
        take_profit=70.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick1 = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=80.0,
        close=85.0,
        volume=1000,
    )
    risk_manager.on_tick(tick1)

    assert trade.stop_loss == 88.0

    tick2 = Tick(
        timestamp=get_ts("2025-01-03"),
        symbol="AAPL",
        open=85.0,
        high=90.0,
        low=70.0,
        close=75.0,
        volume=1000,
    )
    risk_manager.on_tick(tick2)

    assert trade.stop_loss == 77.0


def test_different_symbol_no_trigger(risk_manager: RiskManager):
    trade = Trade(
        entry_time=get_ts("2025-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        z_score=2.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=120.0,
    )
    risk_manager.update_trades({"AAPL": trade})

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="MSFT",
        open=200.0,
        high=210.0,
        low=190.0,
        close=180.0,
        volume=1000,
    )
    events = risk_manager.on_tick(tick)

    assert events == []
