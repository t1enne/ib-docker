#!/usr/bin/env python3
"""Quick test to verify equity curve is built correctly."""

import pandas as pd
from datetime import datetime, timedelta
from src.bt.state import (
    create_initial_portfolio,
    Tick,
    EquityPoint,
    PortfolioState,
)
from src.bt.portfolio.pure import update_prices


def test_equity_curve_updates_without_positions():
    """Test that equity curve is updated even when no positions exist."""
    portfolio = create_initial_portfolio(
        initial_capital=10000,
        start_timestamp=pd.Timestamp("2025-01-01"),
    )

    # Verify initial state
    assert len(portfolio.equity_curve) == 1
    print(f"Initial equity curve length: {len(portfolio.equity_curve)}")

    # Simulate 10 days of ticks without any positions
    for i in range(10):
        tick = Tick(
            timestamp=pd.Timestamp(f"2025-01-{i + 2:02d}"),
            symbol="AAPL",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i,
            volume=1000,
        )
        portfolio = update_prices(portfolio, tick)

    # Verify equity curve has 11 points (initial + 10 updates)
    assert len(portfolio.equity_curve) == 11, (
        f"Expected 11 equity points, got {len(portfolio.equity_curve)}"
    )
    print(f"Final equity curve length: {len(portfolio.equity_curve)}")

    # Verify timestamps
    for i, point in enumerate(portfolio.equity_curve):
        print(f"  Point {i}: {point.timestamp} - Equity: ${point.equity:.2f}")

    print("\n✓ Test passed - equity curve is updated for all periods")


if __name__ == "__main__":
    test_equity_curve_updates_without_positions()
