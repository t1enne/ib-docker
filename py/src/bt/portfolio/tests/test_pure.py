"""Tests for pure portfolio functions — critical paths only."""

from typing import cast

import pandas as pd
import pytest

from src.bt.portfolio.pure import apply_fill, update_prices
from src.bt.state import (
    ActionType,
    Candle,
    EquityPoint,
    FillEvent,
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    TradeStatus,
    create_initial_portfolio,
)


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def test_open_long_position():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            qty=10.0,
            stop_loss=95.0,
            take_profit=110.0,
        ),
        filled_qty=10.0,
        executed_price=100.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )
    new = apply_fill(portfolio, fill)
    assert "AAPL" in new.positions
    assert len(new.trades) == 1
    assert portfolio.cash == 10000  # immutable


def test_close_long_position():
    pid = "AAPL_close"
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=95.0,
        take_profit=110.0,
        last_price=100.0,
        type=ActionType.long,
        position_id=pid,
    )
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (position,)},
        trades=(
            Trade(
                entry_time=_ts("2024-01-01"),
                entry_price=100.0,
                exit_time=None,
                exit_price=None,
                last_price=100.0,
                reason="",
                symbol="AAPL",
                position=ActionType.long,
                qty=10.0,
                stop_loss=95.0,
                take_profit=110.0,
                pnl=0.0,
                status=TradeStatus.open,
                position_id=pid,
            ),
        ),
        equity_curve=(
            EquityPoint(
                timestamp=_ts("2024-01-01"),
                equity=6000,
                cash=5000,
                positions_value=1000,
            ),
        ),
        initial_capital=10000,
    )
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
            position_id=pid,
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    new = apply_fill(portfolio, fill)
    assert "AAPL" not in new.positions
    assert new.trades[0].status == TradeStatus.closed
    assert new.trades[0].pnl == 100.0  # (110-100)*10 - 1
    assert "AAPL" in portfolio.positions  # immutable


def test_close_requires_position_id():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    fill_open = FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            qty=10.0,
            position_id="AAPL_open",
        ),
        filled_qty=10.0,
        executed_price=100.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )
    portfolio = apply_fill(portfolio, fill_open)
    fill_close = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    with pytest.raises(ValueError, match="requires position_id"):
        apply_fill(portfolio, fill_close)


def test_update_prices():
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=95.0,
        take_profit=110.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (position,)},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=_ts("2024-01-01"),
                equity=6000,
                cash=5000,
                positions_value=1000,
            ),
        ),
        initial_capital=10000,
    )
    tick = Candle(
        timestamp=_ts("2024-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=99.0,
        close=105.0,
        volume=1000.0,
    )
    new = update_prices(portfolio, tick)
    aapl = new.positions["AAPL"]
    assert len(aapl) == 1
    assert aapl[0].last_price == 105.0
    assert len(new.equity_curve) == 2
