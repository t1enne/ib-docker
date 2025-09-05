# Candles Fetcher

Fetches OHLCV candlestick data from Interactive Brokers API.

## Usage

```bash
# Install dependencies
uv sync

# Fetch daily candles for AAPL
python main.py AAPL

# Fetch hourly candles with date range
python main.py AAPL -i 1h -s 2024-01-01 -e 2024-01-31

# Save to CSV file
python main.py AAPL -o aapl_data.csv

# Available intervals: 1min, 5min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m
```

## Output

Returns OHLCV data with columns:
- Date (timestamp)
- Open
- High  
- Low
- Close
- Volume