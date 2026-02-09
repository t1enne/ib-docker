# Backtesting Fixes (Quant Perspective)

## Portfolio and risk

- Implement pair sizing with hedge ratio and dollar neutrality.
  - Why: current sizing uses remaining cash per leg, causing unequal exposure; use beta- or price-weighted sizing to hedge and control net exposure.

## Strategy logic

- Add explicit exit rules tied to spread reversion.
  - Why: positions only close via SL/TP; mean-reversion strategies should exit on z-score reversion or time-based decay.

- Bound the rolling buffer and handle missing ticks.
  - Why: pending tick logic can accumulate if a symbol is missing for a timestamp, which can leak memory and skew signals.

## Validation and instrumentation

- Add metrics for gross/net exposure, turnover, and slippage impact.
  - Why: pairs strategies need exposure and turnover tracking to understand risk and cost sensitivity.
