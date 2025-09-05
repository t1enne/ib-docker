#!/usr/bin/env python3
"""
Backtester using vectorbt
"""

import argparse
import vectorbt as vbt
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV candlestick data from IBKR"
    )
    parser.add_argument("csv", help="Path to csv with OHLCV data to backtest")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    # df = df.sort_index()
    close_price = df["Close"]

    fast_window = 10
    slow_window = 50
    fast_ma = vbt.MA.run(close_price, fast_window, short_name="fast")
    slow_ma = vbt.MA.run(close_price, slow_window, short_name="slow")

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(close_price, entries, exits)
    print(pf.total_return())
    print(pf.stats())


if __name__ == "__main__":
    main()
