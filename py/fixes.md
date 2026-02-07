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

- Add metrics for gross/net exposure, turnover, and slippage impact.
  - Why: pairs strategies need exposure and turnover tracking to understand risk and cost sensitivity.

## Displaying results

- Add comprehensive results analysis similar to `pyfolio`

```
Entire data start date: 2004-01-09
Entire data end date: 2009-12-31


Out-of-Sample Months: 2
Backtest Months: 69
                   Backtest  Out_of_Sample  All_History
annual_return          0.12           0.16         0.12
annual_volatility      0.26           0.22         0.25
sharpe_ratio           0.48           0.74         0.48
calmar_ratio           0.21           2.23         0.21
stability              0.00           0.04         0.01
max_drawdown          -0.60          -0.07        -0.60
omega_ratio            1.09           1.13         1.09
sortino_ratio          0.71           1.04         0.71
skewness               0.28          -0.29         0.27
kurtosis               4.07           0.36         4.03
alpha                  0.09          -0.06         0.09
beta                   0.81           1.20         0.81

Worst Drawdown Periods
   net drawdown in %  peak date valley date recovery date duration
0              59.52 2007-11-06  2008-11-20           NaT      NaN
1              22.34 2006-02-16  2006-08-31    2007-05-21      328
2              12.52 2005-07-28  2005-10-12    2006-01-11      120
3              11.29 2004-11-15  2005-04-28    2005-07-28      184
4               9.44 2007-07-16  2007-08-06    2007-09-04       37
```
