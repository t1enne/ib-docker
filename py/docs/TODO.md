# Backtesting Fixes (Quant Perspective)

## Portfolio and risk

- Implement pair sizing with hedge ratio and dollar neutrality.
  - Why: current sizing uses remaining cash per leg, causing unequal exposure; use beta- or price-weighted sizing to hedge and control net exposure.

## Strategy logic

- Add time-based decay
  - Why: positions should be able to close based on time-based decay.

- Bound the rolling buffer and handle missing ticks.
  - Why: pending tick logic can accumulate if a symbol is missing for a timestamp, which can leak memory and skew signals.

## Validation and instrumentation

- Add metrics for gross/net exposure, turnover, and slippage impact.
  - Why: pairs strategies need exposure and turnover tracking to understand risk and cost sensitivity.

## Data resampling

- Add resampling to market data, to construct higher timeframes candles from lower ones.

## Indicators

- Add indicator helpers for use inside strategies, similar to pinescript functions

## HTF data from strategies

- Add `security()` function to get access to higher time-frame candles from inside the strategy
  - Why: it's useful to verify trends and process other indicators in HTF before entering positions.
