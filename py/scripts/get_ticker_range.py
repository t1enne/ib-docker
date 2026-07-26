import argparse
import sys
from datetime import datetime

sys.path.insert(0, "src")

from src.data.types import db, CandleSchema


def get_contiguous_ranges(ticker: str):
    query = (
        CandleSchema.select(CandleSchema.timestamp)
        .where(CandleSchema.ticker == ticker)
        .order_by(CandleSchema.timestamp)
        .distinct()
    )
    timestamps = sorted(set(row.timestamp for row in query))

    if not timestamps:
        return []

    ranges = []
    range_start = timestamps[0]
    prev_ts = timestamps[0]

    for ts in timestamps[1:]:
        if ts - prev_ts != 3600000:
            ranges.append((range_start, prev_ts))
            range_start = ts
        prev_ts = ts

    ranges.append((range_start, prev_ts))
    return ranges


def print_table(ranges):
    header = f"{'#':<4} | {'From (datetime)':<20} | {'To (datetime)':<20}"
    separator = "-" * len(header)

    print(header)
    print(separator)

    for i, (start, end) in enumerate(ranges, 1):
        start_dt = datetime.fromtimestamp(start / 1000).strftime("%Y-%m-%d %H:%M:%S")
        end_dt = datetime.fromtimestamp(end / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:<4} | {start_dt:<20} | {end_dt:<20}")


def main():
    parser = argparse.ArgumentParser(
        description="Get contiguous timestamp ranges for a ticker"
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    args = parser.parse_args()

    db.connect(reuse_if_open=True)

    try:
        ranges = get_contiguous_ranges(args.ticker)

        if not ranges:
            print(f"No candles found for ticker: {args.ticker}")
            return

        print(f"\nFound {len(ranges)} contiguous range(s) for {args.ticker}:\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
