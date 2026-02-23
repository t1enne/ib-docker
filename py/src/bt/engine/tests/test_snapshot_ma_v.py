"""Snapshot test for ma_v strategy.

This test ensures the backtest output remains consistent across refactors.
"""

from src.bt.types import StrategyConfig

import pytest
import asyncio
import os

from src.bt import backtest


# Path to ma_v.yaml relative to project root
MA_V_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "snapshots", "ma_v.txt")


@pytest.fixture
def ma_v_config():
    """Load the ma_v strategy config."""
    return StrategyConfig(
        name="ma_vma_v",
        training_start="2025-01-01",
        training_end="2025-02-01",
        trading_start="2025-02-02",
        trading_end="2026-02-20",
        rolling_window_size=50,
        commission=0.1,
        initial_capital=10000,
        plot=False,
        position_size=0.2,
        strategy_type="pnd",
        stop_loss=0.05,
        take_profit=0.1,
        bar="1h",
        strategy_params={
            "entry_z": 3,
            "exit_z": 0.5,
        },
        symbols=["spy", "qqq"],
    )


@pytest.mark.asyncio
async def test_ma_v_snapshot(ma_v_config):
    """Test that ma_v backtest output matches snapshot."""
    # Run backtest with return_output=True
    output = await backtest(ma_v_config)
    assert output is not None, "backtest returned None"
    output = output.rstrip("\n")  # Normalize trailing newlines

    # Read snapshot
    with open(MA_V_SNAPSHOT_PATH, "r") as f:
        expected = f.read().rstrip("\n")  # Normalize trailing newlines

    # Compare
    assert output == expected, (
        f"Backtest output changed!\n"
        f"Expected {MA_V_SNAPSHOT_PATH}\n"
        f"Output length: {len(output)}, Expected length: {len(expected)}\n"
        f"First difference at position: {next((i for i, (a, b) in enumerate(zip(output, expected)) if a != b), -1)}"
    )
