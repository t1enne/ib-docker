# Pairs Trading Strategy — Fix Plan & Hedge Fund Comparison

## Context

Latest run: SPY/QQQ 1h bars, entry_z=3, exit_z=0.5, SL=5%, TP=10%, 20% position sizing.
Result: -0.72% annual return, -$75.42 total P&L, 47% win rate, 34 trades, Sharpe -0.86.

---

## Issues Identified

### 1. Trailing Stop-Loss Has No Memory (Critical — P&L Impact: High)

**File**: `src/bt/risk/__init__.py:76-85`

The trailing stop recalculates from `max(entry_price, current_tick.high)` every tick. It never remembers the highest price since entry:

- After a rally from 500 → 520, if the current bar has high=510, the stop drops to `max(500, 510) * 0.95 = 484.50` instead of staying at `520 * 0.95 = 494`.
- The stop **moves backwards** on every bar that doesn't make a new high.
- With 1h bars on SPY/QQQ, this turns the trailing stop into a static stop that jitters around entry, letting profitable trades give back all gains.

**Fix**: Add `highest_price` / `lowest_price` to the `Trade` dataclass (initialized to `entry_price`). Update `_update_trailing_sl` to persist the high-water mark:

```python
# For longs:
trade.highest_price = max(trade.highest_price, tick.high)
trade.stop_loss = round(trade.highest_price * (1 - self.stop_loss_pct), 2)

# For shorts:
trade.lowest_price = min(trade.lowest_price, tick.low)
trade.stop_loss = round(trade.lowest_price * (1 + self.stop_loss_pct), 2)
```

**Files to change**:
- `src/bt/types.py` — add `highest_price` and `lowest_price` fields to `Trade`
- `src/bt/risk/__init__.py` — fix `_update_trailing_sl` to use persistent high/low
- `src/bt/portfolio/__init__.py` — set `highest_price`/`lowest_price` on trade open

**Tests to add/update**: `src/bt/risk/tests/test_risk.py`
- Test that SL ratchets up after price makes new high
- Test that SL never moves backwards (monotonically tightening)
- Test short-side equivalent

---

### 2. SL/TP Fires Per-Leg, Not Per-Pair (Moderate — P&L Impact: Moderate)

**File**: `src/bt/risk/__init__.py:33-74`

The risk manager checks each symbol independently. If SPY (long leg) drops 5%, SL fires and closes it — even if QQQ (short leg) also dropped 5%, making the pair P&L flat. The result is:

1. Long leg closed via SL → loss realized
2. Short leg left open as naked directional bet
3. Short leg eventually closed via z-score regression or its own SL/TP

This creates unhedged exposure and converts what should be a flat pair trade into two separate losing directional trades.

**Fix options**:

**Option A — Pair-level P&L check**: Before triggering SL on one leg, compute the combined unrealized P&L of both legs. Only trigger if the *pair* P&L exceeds the threshold. Requires the risk manager to know which trades form a pair.

**Option B — Close both legs atomically**: When SL fires on one leg, automatically generate a close signal for the other leg too. Simpler to implement, doesn't require pair P&L math.

**Option C — Disable per-leg SL, rely on z-score exit only**: The simplest approach. Remove SL/TP from individual legs entirely and let z-score regression handle exits. Add a time-based or drawdown-based pair-level stop as a safety net.

**Recommended**: Option B as a first step — smallest change, prevents naked legs.

**Files to change**:
- `src/bt/risk/__init__.py` — return both legs' close events when one triggers
- `src/bt/types.py` — potentially add a `pair_id` or `counterpart_symbol` to `Trade`
- `src/bt/engine/backtest_engine.py` — ensure both close fills are processed

**Tests**: Verify that when one leg's SL fires, the other leg is also closed.

---

### 3. `close_order` Spread Always Subtracted (Minor — P&L Impact: Low)

**File**: `src/bt/execution/__init__.py:73`

```python
base_price = signal_price - base_spread  # always subtracts, regardless of direction
```

When closing a long (selling), subtracting spread is correct (you hit the bid). When closing a short (buying back), spread should be *added* (you lift the ask). Currently shorts get a slightly better fill on close than they should.

At `spread_bps=5` and ~$2000 positions, this is ~$0.10/trade — minor but systematically biased.

**Fix**: Accept trade direction in `close_order` or infer it from the event. Apply spread in the correct direction.

**Files to change**: `src/bt/execution/__init__.py`
**Tests to update**: `src/bt/execution/tests/test_handler.py`

---

### 4. `except IndexError, ValueError` — Python 2 Syntax (Bug)

**File**: `src/bt/models/strategy_model.py:140, 160, 182`

```python
except IndexError, ValueError:  # catches IndexError, binds to name "ValueError"
```

Should be:
```python
except (IndexError, ValueError):  # catches both
```

Not impacting this run (HMM disabled), but will crash when HMM is enabled and raises `ValueError`.

**Files to change**: `src/bt/models/strategy_model.py` (3 occurrences)

---

### 5. Debug `print(len(data))` Left in Production Code (Cosmetic)

**File**: `src/utils.py:44`

The "1970" output at the top of the backtest run is `print(len(data))` — the number of candle rows loaded per symbol (1970 rows each). Should be removed or converted to `logger.debug()`.

**Files to change**: `src/utils.py:44`

---

## Implementation Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| 1 | Trailing SL memory (#1) | Small | High — stops giving back profits |
| 2 | Pair-level SL (#2, Option B) | Medium | Moderate — prevents naked legs |
| 3 | except syntax (#4) | Trivial | Prevents crashes with HMM |
| 4 | close_order spread (#3) | Small | Low — correct but small dollar impact |
| 5 | Remove debug print (#5) | Trivial | Cosmetic |

---

## How a Hedge Fund Would Approach This Differently

### Signal Generation

**What you have**: Rolling OLS in log-price space → z-score with fixed entry/exit thresholds.

**What a fund would do**:

- **Cointegration testing first**: Run Engle-Granger or Johansen cointegration tests on the training period. If the pair isn't cointegrated (SPY/QQQ often aren't — they're correlated but not cointegrated), the entire mean-reversion premise is invalid. A fund would test the null hypothesis before deploying capital.
- **Kalman filter instead of rolling OLS**: The hedge ratio (beta) drifts over time. Rolling OLS with a fixed window is a crude approximation. A Kalman filter provides a dynamic, smooth estimate of beta and spread that adapts to regime changes without the cliff-edge effect of a rolling window.
- **Half-life of mean reversion**: Compute the Ornstein-Uhlenbeck half-life of the spread. If the half-life is longer than your intended holding period, the spread won't mean-revert fast enough. A fund would reject pairs with half-life > N bars.
- **Adaptive thresholds**: Instead of fixed entry_z=3 / exit_z=0.5, use thresholds that adapt to recent spread volatility or are optimized per pair based on historical distribution.

### Risk Management

**What you have**: Per-leg trailing SL/TP at fixed percentages.

**What a fund would do**:

- **Dollar-neutral enforcement**: Continuously rebalance to maintain dollar neutrality. If one leg moves significantly, the hedge ratio drifts, and you're taking on market beta. A fund would rebalance intraday.
- **Pair-level risk, not leg-level**: Risk limits on the *spread P&L*, not individual leg prices. A 5% move in SPY doesn't matter if QQQ moved 5% too. The relevant metric is the spread deviation from its expected value.
- **Maximum holding period**: If the spread hasn't reverted within N bars (based on half-life), close the position. Holding too long means the OLS parameters have likely drifted and the original trade thesis is stale.
- **Correlation regime monitoring**: SPY/QQQ correlation can break down during stress events. A fund would monitor rolling correlation and widen stops or halt trading when correlation drops below a threshold.
- **Gross/net exposure limits**: Cap total portfolio exposure regardless of individual pair signals.

### Execution

**What you have**: Fixed spread + slippage bps, signal-price-based fills.

**What a fund would do**:

- **VWAP/TWAP execution**: Split large orders across the bar to minimize market impact, especially at hourly frequency on ETFs.
- **Simultaneous leg execution**: Execute both legs of a pair atomically (or as close to simultaneously as possible). Your current code fills legs on different ticks, creating execution risk — the spread can move between fills.
- **Market impact modeling**: Slippage as a function of order size relative to ADV (average daily volume), not fixed bps. SPY is extremely liquid; slippage should be near-zero.

### Portfolio Construction

**What you have**: Single pair, fixed position size (20% of cash).

**What a fund would do**:

- **Multi-pair portfolio**: Trade 20-50 pairs simultaneously across sectors to diversify idiosyncratic risk. Any single pair has ~50% hit rate; diversification across uncorrelated pairs generates consistent returns.
- **Kelly criterion or risk-parity sizing**: Size positions based on expected Sharpe, correlation with existing positions, and current portfolio risk — not a fixed fraction of cash.
- **Sector neutrality**: Ensure the pairs portfolio doesn't have hidden sector or factor exposure. SPY/QQQ is essentially a large-cap growth bet when the hedge breaks down.

### Pair Selection

**What you have**: SPY/QQQ hardcoded in config.

**What a fund would do**:

- **Universe screening**: Start with a universe of 500+ liquid securities. Screen for cointegration, correlation stability, and mean-reversion speed. Reject pairs that fail statistical tests.
- **SPY/QQQ specifically**: This pair is extremely well-arbitraged by institutional desks. The spread rarely deviates enough to overcome transaction costs at hourly frequency. A fund would likely pass on this pair or trade it only at much higher frequency (seconds/minutes) with minimal costs.
- **Fundamental linkage**: Prefer pairs with a fundamental reason for cointegration (same sector, supplier/customer, holding company/subsidiary) over purely statistical pairs, which are more likely to break down.

### Infrastructure

**What you have**: Walk-forward backtest with SQLite data.

**What a fund would do**:

- **Out-of-sample validation**: Strict train/validate/test splits. Never peek at test data for parameter selection.
- **Monte Carlo simulation**: Run thousands of bootstrapped backtests to estimate the distribution of outcomes, not just point estimates.
- **Transaction cost sensitivity analysis**: Sweep commission and slippage assumptions to find the breakeven point. If your strategy is only profitable at unrealistically low costs, it's not a real edge.
- **Live paper trading**: Run the strategy in paper trading for months before deploying capital. Backtests systematically overestimate performance.

### The Core Problem

The fundamental issue with this specific backtest isn't the code bugs (though they make it worse) — it's that **SPY/QQQ at hourly frequency with entry_z=3 is a low-opportunity, high-cost setup**. The pair is too well-arbitraged for large deviations to occur frequently, and when they do occur at z=3, they often represent genuine structural divergence (e.g., tech sell-off) rather than mean-reverting noise. A fund would either:

1. Trade this pair at much higher frequency (sub-minute) with colocation and near-zero costs, or
2. Trade less correlated, less arbitraged pairs where statistical edges persist longer
