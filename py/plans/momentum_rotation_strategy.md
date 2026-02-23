# Institutional Cross-Sectional ETF Momentum Strategy

## 1. Strategy Overview

A cross-sectional, multi-horizon momentum strategy trading the top 200 ETFs by rolling market capitalization.
The strategy selects the strongest ETFs based on volatility-adjusted composite momentum,
applies correlation filtering and risk parity weighting, and targets a constant portfolio volatility.

Rebalance frequency: Monthly  
Positioning: Long-only (optionally beta-hedged)  
Risk target: 12% annualized volatility

---

## 2. Universe Construction

At each rebalance date:

1. Select top 200 ETFs ranked by rolling market capitalization.
2. Apply liquidity filters:
   - 60-day ADV >= $20M
   - Close price >= $10
3. Exclude:
   - Leveraged ETFs
   - Inverse ETFs
   - ETNs
4. Survivorship-bias-free historical constituents required.

---

## 3. Signal Definition

### 3.1 Lookback Windows (1-month skip)

Compute total returns excluding most recent 21 trading days:

- 12M return (weight 40%)
- 6M return (weight 30%)
- 3M return (weight 20%)
- 1M return (weight 10%)

### 3.2 Volatility Adjustment

For each ETF:

Score_i = WeightedReturn_i / RollingVolatility_i

Where:

- RollingVolatility_i = 126-day annualized std deviation
- Returns computed using log returns

---

## 4. Ranking and Selection

1. Rank ETFs by composite momentum score.
2. Select Top N = 10.
3. Apply correlation filter:
   - Remove ETF if pairwise 90-day correlation > 0.8 with higher-ranked ETF.
   - Replace with next-ranked candidate.

Optional:

- Absolute momentum filter: Only include ETFs with 12M return > 0.

---

## 5. Portfolio Construction

### 5.1 Covariance Estimation

- Rolling 90-day covariance matrix
- Exponentially weighted (lambda = 0.94 optional)

### 5.2 Weighting Scheme

Risk parity within selected ETFs:

Minimize:
sum_i (RC_i - 1/N)^2

Constraints:

- Sum weights = 1
- Max weight = 25%
- Min weight = 5%

---

## 6. Volatility Targeting

Compute portfolio ex-ante volatility:

Vol_p = sqrt(w^T Σ w)

Scale exposure:

LeverageFactor = TargetVol / Vol_p

Clamp:
0.5 <= LeverageFactor <= 1.5

---

## 7. Regime Filter (Crash Protection)

If SPY < 200-day moving average:

- Reduce gross exposure by 50%

Alternative:

- If VIX > threshold → reduce exposure

---

## 8. Risk Controls

- Max position weight: 25%
- Max portfolio drawdown: 20% → cut exposure 50%
- Turnover cap per rebalance: 50%
- Transaction cost model included

---

## 9. Rebalance Logic

Rebalance monthly:

1. Recompute universe
2. Recompute signals
3. Select top ETFs
4. Recompute covariance
5. Solve risk parity weights
6. Apply vol targeting
7. Apply regime filter
8. Execute trades

---

## 10. Expected Characteristics

- Sharpe Ratio: 0.8 – 1.2
- CAGR: 10 – 18%
- Max Drawdown: 15 – 30%
- Turnover: Moderate to High
- Factor exposure: Momentum, partial equity beta
