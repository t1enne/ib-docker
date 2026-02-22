#!/usr/bin/env python3
"""
Demo script showing the functional backtest engine.

This demonstrates:
1. Immutable state transformations
2. State snapshots and replay
3. Deterministic behavior
4. Easy testing
"""

import asyncio
import pandas as pd
from datetime import datetime

from src.bt.state import (
    create_initial_portfolio,
    create_initial_backtest_state,
    Tick,
    TradeSignal,
    FillEvent,
    ActionType,
    PortfolioState,
)
from src.bt.portfolio.pure import apply_fill, update_prices, calculate_equity
from src.bt.execution.pure import execute_signal
from src.bt.state import create_execution_params


def demo_immutability():
    """Demo: State is immutable - original never changes."""
    print("=" * 70)
    print("DEMO 1: Immutability")
    print("=" * 70)

    # Create initial state
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=pd.Timestamp("2024-01-01")
    )

    print(f"Initial portfolio cash: ${portfolio.cash}")
    print(f"Initial positions: {portfolio.positions}")

    # Create a signal and fill
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        timestamp=pd.Timestamp("2024-01-01"),
        price=100.0,
        z_score=2.0,
    )

    params = create_execution_params()
    tick = Tick(
        timestamp=pd.Timestamp("2024-01-01"),
        symbol="AAPL",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )

    fill = execute_signal(signal, tick, params)

    # Apply fill - returns NEW state
    new_portfolio = apply_fill(portfolio, fill)

    print(f"\nAfter applying fill:")
    print(f"  New portfolio cash: ${new_portfolio.cash:.2f}")
    print(f"  New positions: {list(new_portfolio.positions.keys())}")

    print(f"\nOriginal portfolio (unchanged!):")
    print(f"  Original cash: ${portfolio.cash}")
    print(f"  Original positions: {portfolio.positions}")

    print("\n✓ Original state is preserved - pure functions return new state")


def demo_state_snapshots():
    """Demo: Can snapshot and compare states."""
    print("\n" + "=" * 70)
    print("DEMO 2: State Snapshots")
    print("=" * 70)

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=pd.Timestamp("2024-01-01")
    )

    # Take snapshots at each step
    snapshots = [portfolio]

    # Simulate some trades
    for i, price in enumerate([100.0, 105.0, 102.0, 110.0]):
        signal = TradeSignal(
            action=ActionType.long if i == 0 else ActionType.close,
            symbol="AAPL",
            timestamp=pd.Timestamp(f"2024-01-{i + 1:02d}"),
            price=price,
            z_score=2.0 if i == 0 else 0.5,
        )

        tick = Tick(
            timestamp=pd.Timestamp(f"2024-01-{i + 1:02d}"),
            symbol="AAPL",
            open=price - 1,
            high=price + 1,
            low=price - 2,
            close=price,
            volume=1000.0,
        )

        params = create_execution_params()
        fill = execute_signal(signal, tick, params)
        portfolio = apply_fill(portfolio, fill)
        snapshots.append(portfolio)

    print(f"Took {len(snapshots)} state snapshots")
    print("\nCan inspect any point in time:")
    for i, snap in enumerate(snapshots):
        print(f"  Step {i}: Cash=${snap.cash:.2f}, Positions={len(snap.positions)}")

    print("\n✓ Can replay/debug from any snapshot")


def demo_determinism():
    """Demo: Same input always produces same output."""
    print("\n" + "=" * 70)
    print("DEMO 3: Determinism")
    print("=" * 70)

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=pd.Timestamp("2024-01-01")
    )

    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        timestamp=pd.Timestamp("2024-01-01"),
        price=100.0,
        z_score=2.0,
    )

    tick = Tick(
        timestamp=pd.Timestamp("2024-01-01"),
        symbol="AAPL",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )

    params = create_execution_params()
    fill = execute_signal(signal, tick, params)

    # Run multiple times
    result1 = apply_fill(portfolio, fill)
    result2 = apply_fill(portfolio, fill)
    result3 = apply_fill(portfolio, fill)

    print(f"Result 1 cash: ${result1.cash:.2f}")
    print(f"Result 2 cash: ${result2.cash:.2f}")
    print(f"Result 3 cash: ${result3.cash:.2f}")

    assert result1.cash == result2.cash == result3.cash
    print("\n✓ Same inputs always produce same outputs")


def demo_time_travel():
    """Demo: Can replay from any point."""
    print("\n" + "=" * 70)
    print("DEMO 4: Time Travel")
    print("=" * 70)

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=pd.Timestamp("2024-01-01")
    )

    # Execute some trades
    trades = [
        (100.0, ActionType.long),
        (105.0, ActionType.close),
    ]

    # Full run
    states = [portfolio]
    for price, action in trades:
        signal = TradeSignal(
            action=action,
            symbol="AAPL",
            timestamp=pd.Timestamp("2024-01-01"),
            price=price,
            z_score=2.0,
        )
        tick = Tick(
            timestamp=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            open=price - 1,
            high=price + 1,
            low=price - 2,
            close=price,
            volume=1000.0,
        )
        params = create_execution_params()
        fill = execute_signal(signal, tick, params)
        portfolio = apply_fill(portfolio, fill)
        states.append(portfolio)

    final_state = states[-1]

    # Now replay from middle
    middle_state = states[1]  # After first trade

    price, action = trades[1]
    signal = TradeSignal(
        action=action,
        symbol="AAPL",
        timestamp=pd.Timestamp("2024-01-01"),
        price=price,
        z_score=2.0,
    )
    tick = Tick(
        timestamp=pd.Timestamp("2024-01-01"),
        symbol="AAPL",
        open=price - 1,
        high=price + 1,
        low=price - 2,
        close=price,
        volume=1000.0,
    )
    params = create_execution_params()
    fill = execute_signal(signal, tick, params)
    replayed_state = apply_fill(middle_state, fill)

    print(f"Full run final cash: ${final_state.cash:.2f}")
    print(f"Replayed final cash: ${replayed_state.cash:.2f}")

    assert final_state.cash == replayed_state.cash
    print("\n✓ Can replay from any saved state")


def main():
    print("\n" + "=" * 70)
    print("FUNCTIONAL BACKTEST ENGINE DEMO")
    print("=" * 70)

    demo_immutability()
    demo_state_snapshots()
    demo_determinism()
    demo_time_travel()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print("\nKey Benefits:")
    print("  ✓ Predictable: Same input → same output")
    print("  ✓ Testable: Easy to unit test pure functions")
    print("  ✓ Debuggable: Snapshot/compare states at any point")
    print("  ✓ Composable: Functions chain naturally")
    print("  ✓ Concurrent: No shared mutable state")


if __name__ == "__main__":
    main()
