# Strategy: Volatility Expansion + Pullback Continuation

## Hybrid Momentum (Trend + Breakout + Pullback Entry)

---

## 1. Overview

This strategy captures post-compression trend expansions by:

1. Detecting volatility compression
2. Confirming breakout with expansion + volume
3. Entering on controlled pullback
4. Riding continuation with volatility-based trailing exits

Timeframe: Daily  
Holding Period: 5–20 trading days  
Style: Tactical Momentum / Convex Trend  
Turnover: High  
Risk Profile: Aggressive

---

## 2. Universe

Tradable universe consists of liquid ETFs.

### Filters:

- Top 200 ETFs by market capitalization
- 60-day ADV >= $30M
- Close price >= $15
- Exclude leveraged ETFs
- Exclude inverse ETFs
- Exclude ETNs

Preferred exposures:

- Equity sectors
- Broad equity indices
- Commodities
- High-beta thematic ETFs

---

## 3. Signal Architecture

### 3.1 Volatility Compression (Setup Condition)

Detect coiled volatility regime:

CompressionRatio = ATR(14) / ATR(100)

Condition:
CompressionRatio < 0.6

Alternative:

- Bollinger Band width percentile < 20%

---

### 3.2 Expansion Breakout (Trigger Phase)

Breakout confirmation:

- Close > Highest High (20 days)
- True Range > 1.5 × ATR(14)
- Volume > 1.5 × 30-day average volume

All must be satisfied.

---

### 3.3 Pullback Entry (Execution Timing)

After breakout signal:

Wait for pullback defined as one of:

- Price retraces to 5–10 EMA
- OR retraces 30–50% of breakout candle
- OR 2–4 day consolidation with declining volume

Entry trigger:

- Close > previous day's high
  OR
- Bullish engulfing candle

---

### 3.4 Trend Continuation Filter

To avoid fake breakouts:

Require:

- 20-day MA > 50-day MA
- AND 50-day MA slope > 0
- AND 63-day return > 0

---

## 4. Position Sizing

### 4.1 Volatility-Based Risk Model

Risk per trade: 1% of capital

Stop distance:
Stop = Entry - 2.5 × ATR(14)

Position Size:
Shares = (Capital × Risk%) / (2.5 × ATR)

---

### 4.2 Portfolio Constraints

- Max 5 concurrent positions
- Max 10% capital per position
- Max gross exposure: 100%
- Correlation filter: avoid positions with 90-day correlation > 0.8

---

## 5. Exit Rules

### Primary Exit

- Trailing stop = 3 × ATR(14)
  OR
- Close < 10-day low

### Secondary Exit

- Close < 20 EMA
- Time stop = 20 trading days

Optional:

- Take 50% profit at 2R
- Trail remaining position

---

## 6. Risk Controls

- If monthly drawdown > 10% → reduce new position size by 50%
- No new entries if VIX > 35 (optional filter)
- Portfolio volatility cap: 25% annualized

---

## 7. Rebalance Logic

Event-driven (not periodic):

For each symbol daily:

1. Check compression
2. Check breakout
3. Monitor pullback window
4. Execute entry on continuation trigger
5. Manage trailing stop
6. Enforce portfolio risk constraints

---

## 8. Expected Performance Profile

Win Rate: 45–60%  
Reward/Risk: 1.8–3.0  
Sharpe: Regime-dependent  
Drawdowns: Larger than slow momentum  
Convexity: High

---

## 9. Failure Modes

- Choppy sideways markets
- False breakouts in high-vol regimes
- Overcrowded sector rotations
- Correlated exposure stacking

Proper volatility sizing and correlation filtering are critical.
