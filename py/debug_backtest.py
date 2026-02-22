#!/usr/bin/env python3
"""Debug script to check data feed and equity curve building."""

import asyncio
import pandas as pd
from src.bt.engine.functional_engine import FunctionalBacktestEngine
from src.bt.types import StrategyConfig, EngineWindow
from src.utils import parse_timestamp


async def debug_backtest():
    """Debug the backtest to see what's happening with data."""
    config = StrategyConfig(
        name="test",
        strategy_type="pnd",
        symbols=["spy", "qqq"],
        entry_z=3.0,
        exit_z=0.5,
        stop_loss=0.05,
        take_profit=0.1,
        initial_capital=10000,
        position_size=0.2,
        commission=0.1,
        training_start="2025-01-01",
        training_end="2025-02-01",
        trading_start="2025-02-02",
        trading_end="2026-02-20",
        rolling_window_size=50,
        plot=False,
        bar="1h",
    )

    engine = FunctionalBacktestEngine(config)

    # Check window
    print(f"Training start: {engine.window.train_start}")
    print(f"Training end: {engine.window.train_end}")
    print(f"Test start: {engine.window.test_start}")
    print(f"Test end: {engine.window.test_end}")

    # Check data feed
    from src.bt.data_feed import DataFeed

    data_feed = DataFeed(config, engine.window)

    print(f"\nData feed symbols: {data_feed.symbols}")
    print(f"Data feed bar: {data_feed.bar}")
    print(f"Candles df shape: {data_feed.candles_df.shape}")
    print(
        f"Candles df index range: {data_feed.candles_df.index.min()} to {data_feed.candles_df.index.max()}"
    )

    # Count ticks
    tick_count = 0
    async for tick in data_feed.get_data_stream():
        if tick is None:
            break
        tick_count += 1
        if tick_count <= 5:
            print(f"Tick {tick_count}: {tick.timestamp} - {tick.symbol} @ {tick.close}")

    print(f"\nTotal ticks: {tick_count}")

    # Run the actual backtest
    print("\nRunning backtest...")
    results = await engine.run()

    print(f"\nResults:")
    print(f"  Equity curve points: {len(results.pf.equity_curve)}")
    print(f"  Trades: {len(results.pf.trades)}")
    print(f"  Data shape: {results.data.shape}")
    print(f"  Z-scores shape: {results.z_scores.shape}")

    if len(results.pf.equity_curve) > 0:
        print(f"\n  Equity curve date range:")
        print(f"    Start: {results.pf.equity_curve.index[0]}")
        print(f"    End: {results.pf.equity_curve.index[-1]}")


if __name__ == "__main__":
    asyncio.run(debug_backtest())
