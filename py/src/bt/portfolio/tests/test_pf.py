import pytest
from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
from src.bt.types import TradeSignal, ActionType, Tick
from src.utils import get_ts


@pytest.fixture
def portfolio():
    return Portfolio(
        PortfolioProps(
            stop_loss=0.1,
            take_profit=0.5,
            initial_capital=10000,
            position_size=0.1,
            commission=1,
            start_date=get_ts("2024-12-31"),
        )
    )


def test_on_signal(portfolio: Portfolio):
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
    assert portfolio.positions["AAPL"] == 0


def test_sl(portfolio: Portfolio):
    print(portfolio.open_trades)
    # Open a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    assert portfolio.open_trades["AAPL"].stop_loss == 90

    # Simulate tick with price below stop loss (100 * (1 - 0.1) = 90)
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=105.0,
        low=85.0,
        close=85.0,  # Below SL
        volume=1000,
    )
    portfolio.on_tick(tick)

    # Position should be closed
    assert portfolio.positions["AAPL"] == 0
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].exit_price == 85.0
    assert portfolio.trades[0].close_reason == "stop_loss"


def test_trailing_sl(portfolio: Portfolio):
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=105.0,
        low=85.0,
        close=105.0,  # Below SL
        volume=1000,
    )
    portfolio.on_tick(tick)
    assert portfolio.trades[0].stop_loss == 105 * (1 - 0.1)


def test_tp(portfolio: Portfolio):
    # Open a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    assert portfolio.open_trades["AAPL"].take_profit == 150

    # Simulate tick with price above take profit (100 * 1.5 = 150)
    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=160.0,
        low=95.0,
        close=155.0,  # Above TP
        volume=1000,
    )
    portfolio.on_tick(tick)

    # Position should be closed
    assert portfolio.positions["AAPL"] == 0
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].exit_price == 155.0
    assert portfolio.trades[0].close_reason == "take_profit"


def test_position_sizing(portfolio: Portfolio):
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


def test_commissions(portfolio: Portfolio):
    entry_price = 100
    exit_price = 120
    initial_cash = portfolio.cash
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    portfolio.on_signal(signal)
    qty = portfolio.positions["AAPL"]
    cost = qty * entry_price
    assert portfolio.cash == initial_cash - cost - portfolio.commission

    updated_cash = portfolio.cash
    # Close and check commission on exit
    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    portfolio.on_signal(close_signal)
    # Commission deducted again on close
    pnl = (exit_price - entry_price) * qty
    assert portfolio.cash == updated_cash + pnl - portfolio.commission


def test_equity_updates_on_every_tick(portfolio: Portfolio):
    commission = portfolio.commission

    portfolio.on_signal(
        TradeSignal(
            timestamp=get_ts("2025-01-01 10:00"),
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            price=100.0,
        )
    )

    qty = portfolio.positions["AAPL"]
    expected_cash = portfolio.initial_capital - (qty * 100.0) - commission
    # assert abs(portfolio.cash - expected_cash) < 0.01

    ticks = [
        Tick(
            timestamp=get_ts("2025-01-01 10:00"),
            symbol="AAPL",
            open=102.0,
            high=103.0,
            low=101.0,
            close=102.0,
            volume=1000,
        ),
        Tick(
            timestamp=get_ts("2025-01-01 11:00"),
            symbol="AAPL",
            open=102.0,
            high=105.0,
            low=101.0,
            close=104.0,
            volume=1000,
        ),
        Tick(
            timestamp=get_ts("2025-01-01 12:00"),
            symbol="AAPL",
            open=104.0,
            high=106.0,
            low=103.0,
            close=103.0,
            volume=1000,
        ),
    ]

    for tick in ticks:
        portfolio.on_tick(tick)

    equity_curve = portfolio.get_results().equity_curve
    assert len(equity_curve) == 4

    expected_equities = [
        10000.0,
        expected_cash + qty * 102.0,
        expected_cash + qty * 104.0,
        expected_cash + qty * 103.0,
    ]
    for i, expected_eq in enumerate(expected_equities):
        assert abs(equity_curve.iloc[i] - expected_eq) < 1, f"Tick {i}"

    # portfolio.on_signal(
    #     TradeSignal(
    #         timestamp=get_ts("2025-01-01 13:00"),
    #         action=ActionType.close,
    #         symbol="AAPL",
    #         z_score=0.0,
    #         price=110.0,
    #     )
    # )
    # expected_final_cash = expected_cash + (qty * 110.0 - qty * 100.0)
    # assert abs(portfolio.cash - expected_final_cash) < 0.01
    # assert (
    #     abs(portfolio.get_results().equity_curve.iloc[-1] - expected_final_cash) < 0.01
    # )
