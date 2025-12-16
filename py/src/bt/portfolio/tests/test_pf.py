import pytest
import pandas as pd
from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
from src.bt.types import TradeSignal, ActionType, Tick
from src.utils import get_ts


@pytest.fixture
def portfolio():
    return Portfolio(
        PortfolioProps(
            stop_loss=0.10,
            take_profit=1.5,
            initial_capital=10000,
            position_size=0.1,
            commission=0.0001,
            start_date=get_ts("2025-01-01"),
        )
    )


def test_on_signal(portfolio):
    # Test opening a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    trade = portfolio.on_signal(signal)
    assert trade is not None
    assert trade.symbol == "AAPL"
    assert trade.position == ActionType.long
    assert trade.entry_price == 100.0
    assert portfolio.positions["AAPL"] > 0
    assert portfolio.cash < 10000  # Cash decreased

    # Test closing the position
    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=110.0,
    )
    closed_trade = portfolio.on_signal(close_signal)
    assert closed_trade is not None
    assert closed_trade.exit_price == 110.0
    assert closed_trade.pnl > 0
    assert closed_trade.close_reason == "signal"
    assert portfolio.positions["AAPL"] == 0


def test_sl(portfolio):
    # Open a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)

    # Simulate tick with price below stop loss (100 * 0.9 = 90)
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=105.0,
        low=85.0,  # Below SL
        close=85.0,
        volume=1000,
    )
    portfolio.on_tick(tick)

    # Position should be closed
    assert portfolio.positions["AAPL"] == 0
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].exit_price == 85.0
    assert portfolio.trades[0].close_reason == "stop_loss"


def test_tp(portfolio):
    # Open a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)

    # Simulate tick with price above take profit (100 * 1.5 = 150)
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=160.0,  # Above TP
        low=95.0,
        close=155.0,
        volume=1000,
    )
    portfolio.on_tick(tick)

    # Position should be closed
    assert portfolio.positions["AAPL"] == 0
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].exit_price == 155.0
    assert portfolio.trades[0].close_reason == "take_profit"


def test_position_sizing(portfolio):
    # Initial cash 10000, position_size 0.1, price 100
    # Expected qty: 0.1 * 10000 / 100 = 10
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    expected_qty = round(0.1 * 10000 / 100.0, 4)
    assert portfolio.positions["AAPL"] == expected_qty
    assert portfolio.trades[0].qty == expected_qty


def test_commissions(portfolio):
    initial_cash = portfolio.cash
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    qty = portfolio.positions["AAPL"]
    expected_cost = qty * 100.0 * (1 + 0.0001)
    assert portfolio.cash == initial_cash - expected_cost

    # Close and check commission on exit
    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=120.0,  # Higher price for profit
    )
    portfolio.on_signal(close_signal)
    # Commission deducted again on close
    final_cash = portfolio.cash
    expected_pnl = (120 - 100) * qty
    expected_commission_exit = 0.0001 * qty * 120
    expected_final = (
        initial_cash - expected_cost + expected_pnl - expected_commission_exit
    )
    assert abs(final_cash - expected_final) < 0.01
