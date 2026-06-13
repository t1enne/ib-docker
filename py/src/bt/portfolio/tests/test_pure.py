"""Tests for pure portfolio functions."""

import pytest
import pandas as pd
from typing import cast
from datetime import datetime


def _ts(val: str) -> pd.Timestamp:
    """Create a Timestamp with a narrowing assertion for ty."""
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


from datetime import datetime

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
)
from src.bt.portfolio.pure import apply_fill, update_prices, calculate_equity


class TestApplyFill:
    """Tests for apply_fill function."""

    def test_open_long_position(self):
        """Test opening a long position."""
        portfolio = create_initial_portfolio(
            initial_capital=10000, start_timestamp=_ts("2024-01-01")
        )

        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
        )

        fill = FillEvent(
            signal=signal,
            filled_qty=10.0,
            executed_price=100.0,
            commission=1.0,
            slippage=0.0,
            timestamp=_ts("2024-01-01"),
        )

        new_portfolio = apply_fill(portfolio, fill)

        # Verify changes
        assert new_portfolio.cash < portfolio.cash  # Cash decreased
        assert "AAPL" in new_portfolio.positions
        assert new_portfolio.positions["AAPL"].qty > 0
        assert len(new_portfolio.trades) == 1

        # Original portfolio unchanged (immutability)
        assert portfolio.cash == 10000
        assert "AAPL" not in portfolio.positions

    def test_close_long_position(self):
        """Test closing a long position."""
        # Create portfolio with existing position
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
            positions={"AAPL": position},
            trades=(
                Trade(
                    entry_time=_ts("2024-01-01"),
                    entry_price=100.0,
                    exit_time=None,
                    exit_price=None,
                    last_price=100.0,
                    reason="Z: 2.0",
                    symbol="AAPL",
                    position=ActionType.long,
                    qty=10.0,
                    stop_loss=95.0,
                    take_profit=110.0,
                    pnl=0.0,
                    status=TradeStatus.open,
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

        signal = TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
            reason=0.5,
        )

        fill = FillEvent(
            signal=signal,
            filled_qty=10.0,
            executed_price=110.0,
            commission=1.0,
            slippage=0.0,
            timestamp=_ts("2024-01-02"),
        )

        new_portfolio = apply_fill(portfolio, fill)

        # Verify changes
        assert "AAPL" not in new_portfolio.positions  # Position closed
        assert len(new_portfolio.trades) == 1
        assert new_portfolio.trades[0].status == TradeStatus.closed
        assert new_portfolio.trades[0].pnl == 100.0  # (110-100) * 10 - 1 commission
        assert new_portfolio.cash < 10000

        # Original unchanged
        assert "AAPL" in portfolio.positions


class TestImmutability:
    """Tests demonstrating immutability benefits."""

    def test_no_side_effects(self):
        """Prove that functions don't mutate input."""
        portfolio = create_initial_portfolio(
            initial_capital=10000, start_timestamp=_ts("2024-01-01")
        )

        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            z_score=2.0,
        )

        fill = FillEvent(
            signal=signal,
            filled_qty=10.0,
            executed_price=100.0,
            commission=1.0,
            slippage=0.0,
            timestamp=_ts("2024-01-01"),
        )

        # Store original state
        original_cash = portfolio.cash
        original_positions = dict(portfolio.positions)

        # Apply fill
        new_portfolio = apply_fill(portfolio, fill)

        # Verify original unchanged
        assert portfolio.cash == original_cash
        assert portfolio.positions == original_positions
        assert portfolio is not new_portfolio

    def test_state_snapshots(self):
        """Test that we can snapshot states."""
        portfolio = create_initial_portfolio(
            initial_capital=10000, start_timestamp=_ts("2024-01-01")
        )

        # Take snapshot
        snapshot = portfolio

        # Mutate (well, create new state)
        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            z_score=2.0,
        )

        fill = FillEvent(
            signal=signal,
            filled_qty=10.0,
            executed_price=100.0,
            commission=1.0,
            slippage=0.0,
            timestamp=_ts("2024-01-01"),
        )

        new_portfolio = apply_fill(portfolio, fill)

        # Can compare states
        assert snapshot.cash != new_portfolio.cash
        assert len(snapshot.positions) != len(new_portfolio.positions)


class TestUpdatePrices:
    """Tests for update_prices function."""

    def test_update_position_price(self):
        """Test updating position prices."""
        from src.bt.state import Tick

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
            positions={"AAPL": position},
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

        tick = Tick(
            timestamp=_ts("2024-01-02"),
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=99.0,
            close=105.0,
            volume=1000.0,
        )

        new_portfolio = update_prices(portfolio, tick)

        # Price updated
        assert new_portfolio.positions["AAPL"].last_price == 105.0

        # Equity curve updated
        assert len(new_portfolio.equity_curve) == 2
        assert (
            new_portfolio.equity_curve[-1].equity
            > new_portfolio.equity_curve[-2].equity
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
