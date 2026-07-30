"""Critical execution issues: both legs execute, sizing, SL/TP from config."""

from src.bt.state import (
    TradeSignal,
    ActionType,
    create_initial_portfolio,
    create_execution_params,
    Candle,
)
from src.bt.portfolio.pure import apply_fill
from src.bt.execution.pure import execute_signal
from src.utils import get_ts


def test_both_legs_execute():
    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=get_ts("2025-01-01")
    )
    params = create_execution_params(spread_bps=0, slippage_bps=0, fixed_commission=0.0)

    tick_spy = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    tick_qqq = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="QQQ",
        open=400.0,
        high=401.0,
        low=399.0,
        close=400.0,
        volume=1_000_000,
    )

    fill_spy = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        ),
        tick_spy,
        params,
    )
    fill_qqq = execute_signal(
        TradeSignal(
            action=ActionType.short,
            symbol="QQQ",
            timestamp=get_ts("2025-01-01"),
            price=400.0,
            z_score=3.0,
            qty=0.0,
        ),
        tick_qqq,
        params,
    )

    portfolio = apply_fill(apply_fill(portfolio, fill_spy), fill_qqq)
    assert "SPY" in portfolio.positions
    assert "QQQ" in portfolio.positions
    assert len(portfolio.trades) == 2


def test_position_size_from_config():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=get_ts("2025-01-01")
    )
    tick = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    params = create_execution_params(fixed_commission=0.0, spread_bps=0)
    fill = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        ),
        tick,
        params,
    )
    portfolio_after = apply_fill(
        portfolio, fill, position_size_pct=0.3, stop_loss_pct=0.05, take_profit_pct=0.1
    )
    spy = portfolio_after.positions["SPY"]
    assert len(spy) == 1
    expected_qty = (10000 * 0.3) / 500.0
    assert abs(abs(spy[0].qty) - expected_qty) < 0.01


def test_sl_tp_from_config():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=get_ts("2025-01-01")
    )
    tick = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    params = create_execution_params(fixed_commission=0.0, spread_bps=0)
    fill = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        ),
        tick,
        params,
    )
    portfolio_after = apply_fill(
        portfolio, fill, position_size_pct=0.2, stop_loss_pct=0.05, take_profit_pct=0.1
    )
    pos = portfolio_after.positions["SPY"][0]
    # executed_price includes default slippage even with spread_bps=0
    assert pos.stop_loss is not None and abs(pos.stop_loss - 475.095) < 0.1
    assert pos.take_profit is not None and abs(pos.take_profit - 550.11) < 0.1
