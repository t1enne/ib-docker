"""Tests for functional execution and portfolio issues.

Tests verify:
1. Both legs of a pairs trade execute together
2. Position sizing uses config values
3. SL/TP uses config values
"""

import pytest
import pandas as pd
from src.bt.state import (
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    FillEvent,
    EquityPoint,
    ActionType,
    TradeStatus,
    create_initial_portfolio,
    create_execution_params,
    Tick,
)
from src.bt.portfolio.pure import apply_fill, update_prices
from src.bt.execution.pure import execute_signal
from src.utils import get_ts


class TestBothLegsExecution:
    """Test that both legs of a pairs trade execute together."""

    def test_both_legs_execute_together(self):
        """When we have signals for both legs, both should execute."""
        # Create portfolio with enough cash for both positions
        portfolio = create_initial_portfolio(
            initial_capital=100000,  # More capital
            start_timestamp=get_ts("2025-01-01"),
        )

        # Create two signals (both legs of a pairs trade)
        params = create_execution_params(
            spread_bps=0,
            slippage_bps=0,
            fixed_commission=0.0,
        )

        tick_spy = Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="SPY",
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0,
            volume=1000000,
        )

        tick_qqq = Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="QQQ",
            open=400.0,
            high=401.0,
            low=399.0,
            close=400.0,
            volume=1000000,
        )

        signal_spy = TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        )

        signal_qqq = TradeSignal(
            action=ActionType.short,
            symbol="QQQ",
            timestamp=get_ts("2025-01-01"),
            price=400.0,
            z_score=3.0,
            qty=0.0,
        )

        # Execute both signals with their respective ticks
        fill_spy = execute_signal(signal_spy, tick_spy, params)
        fill_qqq = execute_signal(signal_qqq, tick_qqq, params)

        # Apply SPY fill
        portfolio_after_spy = apply_fill(portfolio, fill_spy)

        # Apply QQQ fill
        portfolio_after_both = apply_fill(portfolio_after_spy, fill_qqq)

        # Verify BOTH positions exist
        assert "SPY" in portfolio_after_both.positions, "SPY position should exist"
        assert "QQQ" in portfolio_after_both.positions, "QQQ position should exist"

        # Verify both are recorded in trades
        assert len(portfolio_after_both.trades) == 2, (
            f"Expected 2 trades, got {len(portfolio_after_both.trades)}"
        )


class TestPositionSizing:
    """Test that position sizing uses config values."""

    def test_position_size_from_config(self):
        """Position size should be calculated from config, not hardcoded."""
        initial_capital = 10000
        position_size_pct = 0.3  # 30% from config

        portfolio = create_initial_portfolio(
            initial_capital=initial_capital,
            start_timestamp=get_ts("2025-01-01"),
        )

        tick = Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="SPY",
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0,
            volume=1000000,
        )

        params = create_execution_params(
            fixed_commission=0.0,
            spread_bps=0,  # No spread for easier calculation
        )

        signal = TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        )

        fill = execute_signal(signal, tick, params)

        # Apply fill WITH custom position size
        portfolio_after = apply_fill(
            portfolio,
            fill,
            position_size_pct=position_size_pct,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
        )

        # Expected: 30% of capital / 500 = 6 shares
        expected_qty = (initial_capital * position_size_pct) / 500.0
        actual_qty = abs(portfolio_after.positions["SPY"].qty)

        assert abs(actual_qty - expected_qty) < 0.01, (
            f"Position size should use config (expected {expected_qty}, got {actual_qty})"
        )


class TestStopLossTakeProfit:
    """Test that SL/TP use config values."""

    def test_sl_tp_from_config(self):
        """Stop loss and take profit should use config values."""
        portfolio = create_initial_portfolio(
            initial_capital=10000,
            start_timestamp=get_ts("2025-01-01"),
        )

        tick = Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="SPY",
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0,
            volume=1000000,
        )

        params = create_execution_params(
            fixed_commission=0.0,
            spread_bps=0,  # No spread for easier calculation
        )

        signal = TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=0.0,
        )

        fill = execute_signal(signal, tick, params)

        # Apply fill WITH custom SL/TP
        portfolio_after = apply_fill(
            portfolio,
            fill,
            position_size_pct=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
        )

        position = portfolio_after.positions["SPY"]

        # executed_price = 500.1 (signal price + default slippage)
        # Config values: stop_loss: 5%, take_profit: 10%
        executed_price = 500.1  # From execution with default slippage
        expected_sl = executed_price * (1 - 0.05)  # 475.095 -> 475.1 (rounded)
        expected_tp = executed_price * (1 + 0.1)  # 550.11 -> 550.11 (rounded)

        assert (
            position.stop_loss is not None
            and abs(position.stop_loss - expected_sl) < 0.1
        ), f"Stop loss should be 5% (expected ~{expected_sl}, got {position.stop_loss})"
        assert (
            position.take_profit is not None
            and abs(position.take_profit - expected_tp) < 0.1
        ), (
            f"Take profit should be 10% (expected ~{expected_tp}, got {position.take_profit})"
        )


class TestEquityCurveUpdates:
    """Test equity curve is updated correctly."""

    def test_equity_updated_without_positions(self):
        """Equity should update even without open positions."""
        portfolio = create_initial_portfolio(
            initial_capital=10000,
            start_timestamp=get_ts("2025-01-01"),
        )

        # Multiple ticks without any positions
        for i in range(5):
            tick = Tick(
                timestamp=get_ts(f"2025-01-{i + 2:02d}"),
                symbol="SPY",
                open=500.0 + i,
                high=501.0 + i,
                low=499.0 + i,
                close=500.0 + i,
                volume=1000000,
            )
            portfolio = update_prices(portfolio, tick)

        # Should have equity points for initial + 5 ticks
        assert len(portfolio.equity_curve) == 6, (
            f"Expected 6 equity points, got {len(portfolio.equity_curve)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
