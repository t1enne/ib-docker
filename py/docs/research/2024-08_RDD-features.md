# Feature Math — Real-Discount-Dominance (RDD) leading features

**Research artifact #2** — concrete formulas for the three HMM features described in
`2024-08_leading-indicator.md`. No code. The goal is an unambiguous, reproducible spec
that a backtest engineer can port to the library's `src/bt` idioms.

---

## 0. Notation & shared scaffolding

- `CPI(t)` — daily-stepped, base-1.0 CPI price-index **level** (as produced by
  `src/bt/strategies/cpi.py::load_cpi_price_index`). Already FF'd onto the daily grid and
  forward-filled over the residual post-CPI window.
- `P(t)` — close of the broad index basket at each backtest timestamp.
- `Y(t)` — nominal 10y UST yield proxy (daily), if the `rdisc` leg is included.
- `τ` — the backtest engine cursor timestamp. Every feature is evaluated only on
  information with `timestamp ≤ τ` (cursor-safe; look-ahead forbidden). Because CPI is
  observed monthly with a release lag, the FF'd value at `τ` is the *latest known*, which
  is exactly the right causal object.

All three features are computed on the **full history then z-scored with a rolling
window**, exactly as `MarketRegimeHMM._create_features` standardises `volatility` /
`momentum` before fitting.

---

## 1. Feature A — inflation acceleration (`acc`)

**Rationale.** The level of CPI is a coincident/lagging macro series; its *second
difference* is a low-frequency leading gauge of cost pressure turning. `acc > 0` means
inflation is *speeding up* (cost-push intensifying), which historically predates the
real discount rate squeeze and the eventual margin/multiple compression.

**Formula (daily-stepped CPI, log domain):**

```
lp(t) = log(CPI(t))
acc(t) = lp(t) − 2·lp(t−1) + lp(t−2)
```

This is the discrete second difference of the log CPI price level. Because `CPI` is
already stepped to daily from monthly observations, most daily increments are 0 and
`acc` is spike-train-like. **Therefore apply smoothing before use** (one of):

- rolling mean of `acc` over `w_a ∈ {21, 63}` trading days (preferred — simple, causal),
- or an HP filter (λ≈1e5) with causal/one-sided smoothing (heavier, optional).

Smooth first, then z-score:

```
z_acc(t) = sm_acc(t) − rolling_mean(sm_acc, 252) ) / rolling_std(sm_acc, 252)
```

**Direction convention.** `z_acc` high = cost-push accelerating → risk-off.

---

## 2. Feature B — real (inflation-adjusted) equity level (`real`)

**Rationale.** Nominally flat price can still be de-risking once CPI is subtracted. `real`
is the exact quantity the library already computes (`cpi.deflated_log_prices`: `real_log
= ln(P) − ln(CPI)`). A *level* of real equity is trendless in principle (z-score stable),
so its own z-score captures **deviation from purchasing-power parity**, i.e. how far the
market has out- or under-run inflation — a valuation-style overflow into the signal.

**Formula:**

```
real(t) = log(P(t)) − log(CPI(t))
z_real(t) = ( real(t) − rolling_mean(real, 252) ) / rolling_std(real, 252)
```

No heavy smoothing needed: `P` is daily and `CPI` is stepped, so `real` inherits daily
granularity from `P` (its slow CPI component only injects a gentle drift).

**Direction convention.** `z_real` high = market over-stretched vs. purchasing power →
*ambiguous* (can be an overshoot to fade, or genuine real carry). This ambiguity is
exactly why it is fused with `acc` and `rdisc` in the joint state rather than used as a
standalone rule.

---

## 3. Feature C — real-discount proxy (`rdisc`)

**Rationale.** The discount rate is what converts a cash-flow into a price; real-rates
are the dominant discount driver at the index horizon (Piazzesi–Schneider 2007). A
plausible, data-light *real-rate proxy* is the nominal yield minus trailing inflation,
which turns *before* margin/earnings revisions.

**Formula:**

```
π_k(t) = 252 · ( log(CPI(t)) − log(CPI(t−k)) ) / k      # annualised trailing CPI, k∈{63,252}
rdisc(t) = Y(t) − π_k(t)
z_rdisc(t) = ( rdisc(t) − rolling_mean(rdisc, 252) ) / rolling_std(rdisc, 252)
```

**Direction convention.** `z_rdisc` high = real discount expensive → risk-off.

---

## 4. Standardisation & smoothing knobs (all causal, all cursor-safe)

| Knob | Symbol | Suggested range | Default | Notes |
|---|---|---|---|---|
| Acc smooth window | `w_a` | {21, 63} | 63 | dampen monthly-jump spikes |
| Trailing-inflation window | `k` | {63, 252} | 252 | annualized real-rate horizon |
| Z-score window | `w_z` | {126, 252} | 252 | matches HMM vol window order |
| HMM state prob threshold | `θ` | 0.5–0.8 | 0.6 | before any sizing |

All windows are **trailing**, evaluated at `τ` from `state.candles` data (cursor-truncated)
or the stepped CPI/yield series aligned to the same grid. Nothing reads
`state.model_state.resample_cache` or any non-cursor view.

---

## 5. Final feature matrix fed to the HMM

```
X(t) = [ z_acc(t), z_real(t), z_rdisc(t) ]
```

- If `rdisc` is unavailable (no yield asset yet), drop column 3 and run the 2-feature
  CPI-only variant — keep the same pipeline so the HMM interface is unchanged.
- Drop any row with a NaN (warmup of the z-scores) before fitting, matching
  `MarketRegimeHMM`'s `.dropna()` behaviour.

**Release-lag note.** Because `CPI` is FF'd, `log CPI` reflects the *latest known* print
at `τ`. `acc` and `π_k` therefore never peek at future prints. This is the correct causal
stance and is inherited from the existing CPI loader — no new look-ahead risk is
introduced by reuse.

---

## 6. Invariants / style requirements (library AGENTS.md)

- **Pure functions**, immutable state, full type annotations — no `Any`.
- **Vectorized** `diff()/diff()` and `rolling()` — no Python loops over observations.
- **State via `replace()`**; rolling z-scores cached in strategy `GLOBAL` with a
  `reset_global()` (split/OOS safety).
- **Look-ahead safe** by construction: rolling windows + cursor data only.
- Functions ≤ 50 LOC; keep each feature as its own pure function returning a `Series`.
