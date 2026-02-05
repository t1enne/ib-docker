# Backtesting Fixes (Quant Perspective)

## Data and methodology

- Fix walk-forward to be true walk-forward.
  - Why: current implementation uses a single training window and one trading window, so it is closer to an expanding-window backtest than a walk-forward. True walk-forward needs rolling train/test splits and scheduled re-fit to avoid optimistic bias.

- Enforce bar-to-bar execution timing.
  - Why: signals are generated off the same bar close and filled at that close, which is look-ahead for daily OHLCV. You should at minimum execute on the next bar open or next bar close.

- Add survivorship and corporate action handling.
  - Why: raw OHLCV without split/dividend adjustments or survivorship control biases performance upward, especially in equity pairs.

## Execution and costs

- Add bid/ask spread, slippage, and latency model.
  - Why: fills at last close without costs materially overstate returns; pairs trading is especially spread-sensitive.

- Model partial fills and insufficient liquidity on small caps.
  - Why: position sizing assumes infinite liquidity and immediate fills.

## Portfolio and risk

- Fix long take-profit math.
  - Why: take-profit uses `entry * take_profit` for longs, which immediately exits for typical values. It should be `entry * (1 + take_profit)`.

- Correct equity curve mark-to-market.
  - Why: equity uses `pos * 100` instead of `pos * price`, so returns and Sharpe are invalid.

- Implement pair sizing with hedge ratio and dollar neutrality.
  - Why: current sizing uses remaining cash per leg, causing unequal exposure; use beta- or price-weighted sizing to hedge and control net exposure.

- Clarify and correct stop-loss logic for shorts.
  - Why: stop updates use `min_price * (1 - stop_loss)` for shorts, which can move the stop away from price and not act as a protective stop.

## Strategy logic

- Add explicit exit rules tied to spread reversion.
  - Why: positions only close via SL/TP; mean-reversion strategies should exit on z-score reversion or time-based decay.

- Bound the rolling buffer and handle missing ticks.
  - Why: pending tick logic can accumulate if a symbol is missing for a timestamp, which can leak memory and skew signals.

## Validation and instrumentation

- Add unit tests for PnL, sizing, and stop/TP behavior.
  - Why: current errors in TP and equity valuation would be caught by basic tests.

- Add metrics for gross/net exposure, turnover, and slippage impact.
  - Why: pairs strategies need exposure and turnover tracking to understand risk and cost sensitivity.
