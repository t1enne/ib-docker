import pytest
from src.bt.portfolio import Portfolio, PortfolioProps
from src.bt.types import TradeSignal, ActionType, Tick, TradeExitReason, FillEvent
from src.utils import get_ts


@pytest.fixture
def pf():
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


def get_fill(s: TradeSignal, pf: Portfolio):
    return FillEvent(
        signal=s,
        filled_qty=1,
        executed_price=s.price,
        commission=pf.commission,
        slippage=0.0,
    )


def test_on_fill(pf: Portfolio):
    # Test opening a long position
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    trade = pf.on_fill(get_fill(signal, pf))

    assert trade is not None
    assert trade.symbol == "AAPL"
    assert trade.position == ActionType.long
    assert trade.entry_price == 100.0
    assert pf.positions["AAPL"] > 0
    assert pf.cash < 10000  # Cash decreased

    # Test closing the position
    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=110.0,
    )
    closed_trade = pf.on_fill(get_fill(close_signal, pf))
    assert closed_trade is not None
    assert closed_trade.exit_price == 110.0
    assert closed_trade.pnl > 0
    assert pf.positions["AAPL"] == 0


# def test_sl(pf: Portfolio):
#     # Open a long position
#     signal = TradeSignal(
#         action=ActionType.long,
#         symbol="AAPL",
#         z_score=2.0,
#         timestamp=get_ts("2025-01-01"),
#         price=100.0,
#     )
#
#     pf.on_fill(get_fill(signal, pf))
#     assert pf.open_trades["AAPL"].stop_loss == 90
#
#     # Simulate tick with price below stop loss (100 * (1 - 0.1) = 90)
#     tick = Tick(
#         timestamp=get_ts("2025-01-02"),
#         symbol="AAPL",
#         open=95.0,
#         high=105.0,
#         low=85.0,
#         close=85.0,  # Below SL
#         volume=1000,
#     )
#     pf.update_market_value(tick)
#
#     # Position should be closed
#     assert pf.positions["AAPL"] == 0
#     assert len(pf.trades) == 1
#     assert pf.trades[0].exit_price == 85.0
#     assert pf.trades[0].close_reason == TradeExitReason.sl


# def test_trailing_sl(pf: Portfolio):
#     signal = TradeSignal(
#         action=ActionType.long,
#         symbol="AAPL",
#         z_score=2.0,
#         timestamp=get_ts("2025-01-01"),
#         price=100.0,
#     )
#     pf.on_fill(get_fill(signal, pf))
#
#     tick = Tick(
#         timestamp=get_ts("2025-01-02"),
#         symbol="AAPL",
#         open=95.0,
#         high=105.0,
#         low=85.0,
#         close=105.0,  # Below SL
#         volume=1000,
#     )
#     pf.update_market_value(tick)
#     assert pf.trades[0].stop_loss == 105 * (1 - 0.1)
#
#
# def test_position_sizing(pf: Portfolio):
#     # Initial cash 10000, position_size 0.1, price 100
#     # Expected qty: 0.1 * 10000 / 100 = 10
#     signal = TradeSignal(
#         action=ActionType.long,
#         symbol="AAPL",
#         z_score=2.0,
#         timestamp=get_ts("2025-01-01"),
#         price=100.0,
#     )
#     pf.on_fill(get_fill(signal, pf))
#     expected_qty = round(0.1 * 10000 / 100.0, 4)
#     assert pf.positions["AAPL"] == expected_qty
#     assert pf.trades[0].qty == expected_qty
#
#
# def test_commissions(pf: Portfolio):
#     entry_price = 100
#     exit_price = 120
#     initial_cash = pf.cash
#     signal = TradeSignal(
#         action=ActionType.long,
#         symbol="AAPL",
#         z_score=2.0,
#         timestamp=get_ts("2025-01-01"),
#         price=entry_price,
#     )
#     pf.on_fill(get_fill(signal, pf))
#     qty = pf.positions["AAPL"]
#     cost = qty * entry_price
#     assert pf.cash == initial_cash - cost - pf.commission
#
#     updated_cash = pf.cash
#     # Close and check commission on exit
#     close_signal = TradeSignal(
#         action=ActionType.close,
#         symbol="AAPL",
#         z_score=0.0,
#         timestamp=get_ts("2025-01-02"),
#         price=exit_price,
#     )
#     pf.on_fill(get_fill(close_signal, pf))
#     # Commission deducted again on close
#     pnl = (exit_price - entry_price) * qty
#     assert pf.cash == updated_cash + pnl - pf.commission
#
#
# def test_equity_updates_on_every_tick(pf: Portfolio):
#     commission = pf.commission
#
#     pf.on_fill(
#         get_fill(
#             TradeSignal(
#                 timestamp=get_ts("2025-01-01 10:00"),
#                 action=ActionType.long,
#                 symbol="AAPL",
#                 z_score=2.0,
#                 price=100.0,
#             ),
#             pf,
#         )
#     )
#
#     qty = pf.positions["AAPL"]
#     expected_cash = pf.initial_capital - (qty * 100.0) - commission
#     # assert abs(portfolio.cash - expected_cash) < 0.01
#
#     ticks = [
#         Tick(
#             timestamp=get_ts("2025-01-01 10:00"),
#             symbol="AAPL",
#             open=102.0,
#             high=103.0,
#             low=101.0,
#             close=102.0,
#             volume=1000,
#         ),
#         Tick(
#             timestamp=get_ts("2025-01-01 11:00"),
#             symbol="AAPL",
#             open=102.0,
#             high=105.0,
#             low=101.0,
#             close=104.0,
#             volume=1000,
#         ),
#         Tick(
#             timestamp=get_ts("2025-01-01 12:00"),
#             symbol="AAPL",
#             open=104.0,
#             high=106.0,
#             low=103.0,
#             close=103.0,
#             volume=1000,
#         ),
#     ]
#
#     for tick in ticks:
#         pf.update_market_value(tick)
#
#     equity_curve = pf.get_results().equity_curve
#     assert len(equity_curve) == 4
#
#     expected_equities = [
#         10000.0,
#         expected_cash + qty * 102.0,
#         expected_cash + qty * 104.0,
#         expected_cash + qty * 103.0,
#     ]
#     for i, expected_eq in enumerate(expected_equities):
#         assert abs(equity_curve.iloc[i] - expected_eq) < 1, f"Tick {i}"


# def test_equity_curve_on_close_all_positions(pf: Portfolio):
# """Ensure equity curve uses exit prices, not stale last_price."""
# signal = TradeSignal(
#     action=ActionType.long,
#     symbol="AAPL",
#     z_score=2.0,
#     timestamp=get_ts("2025-01-03 09:00"),
#     price=100.0,
# )
# signal2 = TradeSignal(
#     action=ActionType.long,
#     symbol="MSFT",
#     z_score=2.0,
#     timestamp=get_ts("2025-01-03 09:00"),
#     price=200.0,
# )
#
# pf.on_fill(get_fill(signal, pf))
# pf.on_fill(get_fill(signal2, pf))
#
# assert pf.cash < pf.initial_capital
#
# qty_aapl = pf.positions["AAPL"]
# qty_msft = pf.positions["MSFT"]
# expected_cash_after_entry = pf.cash
#
# pf.close_all_trades(
#     timestamp=get_ts("2025-01-03 11:00"), prices={"AAPL": 150.0, "MSFT": 250.0}
# )
#
# result = pf.get_results()
# final_equity = result.equity_curve.iloc[-1]
#
# pnl_aapl = (150.0 - 100.0) * qty_aapl
# pnl_msft = (250.0 - 200.0) * qty_msft
# expected_equity = (
#     expected_cash_after_entry + pnl_aapl + pnl_msft - (pf.commission * 2)
# )
#
# assert abs(final_equity - expected_equity) < 1, (
#     f"Expected {expected_equity}, got {final_equity}"
# )
