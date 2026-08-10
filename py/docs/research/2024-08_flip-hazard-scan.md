# Flip-Hazard (Sojourn) Feasibility Scan for SPY — Alternative C

**Status:** Run + interpretation. **VERDICT: DEAD** as a standalone signal. SPY trend-flip
timing is **memoryless** (Poisson-like) — regime age carries no forecastable information
about imminent flips.

**Why tested:** The prior RDD `/real-economy` lines both failed at {3,21}d *return-lead*
prediction. Their shared failure mode is concurrent macro-state classification, not
leadership. Alternative C pivoted from predicting *returns* to predicting the **flip
process itself** via an age-dependent (sojourn / hazard) model of trend-regime duration —
using the newly-extended SPY 2004–2026 daily sample (~5667 bars, roughly double the 
2014-only data the prior work was confined to).

**Repro:** this scan. Pure computation on SPY 1D from local DB (`load_candles` → `query_candles`,
resampled to daily). No strategy code, no new data.

---

## 1. Claim tested

> **H0 for C:** SPY trend-regime **flip times are forecastable from regime age** — i.e. the
> hazard of flipping is duration-dependent (Sojourn/Markov-chain-with-age structure), not
> constant.

Falsifiable pass/fail: Weibull shape `k ≠ 1` (rejecting memoryless **and** monotone-age
signature in the empirical hazard function) with a statistically meaningful effect.

## 2. Method

- Regime = sign of (fast SMA − slow SMA) — the same low-pass trend structure the trend
  detectors in `src/bt/regime/detectors.py` (`create_sma_detector`) canonicalise.
- Persistence: sign series with zeros dropped; run-lengths = regime durations in bars.
- Extract **regime durations** and **inter-flip intervals** per config.
- Tests on durations:
  - **Weibull shape k** (`weibull_min.fit`): k=1 ⇒ memoryless exponential; k>1 ⇒ aging
    (increasing hazard).
  - **LRT k≠1** and **KS vs exponential** GOF.
  - **Empirical age-hazard:** for bins of regime age `a`, `P(flip within next 5d | alive at a)`;
    Spearman rho of hazard vs age (the direct, model-free duration-dependence test).
  - **Inter-flip CV:** Poisson CV=1 ⇒ no clustering; CV<1 ⇒ regular/periodic; CV>1 ⇒ bursty.

Regime configs swept (fast/slow): 50/200, 20/50, 10/30, 5/20, 21/63. All on full 2004–2026.

## 3. Results

### 3.1 Flip sample sufficiency (2004+ vs 2014 only)

| fast/slow | # flips | regime frequency sane? | notes |
|---|---|---|---|
| 50/200 | 20 | yes (5 up / 6 down regimes) | **below 30-event bar** |
| 21/63 | ~94 | yes | fine |
| 20/50 | 118 | yes | fine |
| 10/30 | 185 | yes | fine |
| 5/20 | 331 | yes | fine |

The deeper sample delivers well over the 30-event bar for all but the slowest config — the
**sample-size caveat that plagued the prior RDD flip studies (10–26 flips) is resolved.**

### 3.2 Parametric Weibull / memoryless test

| fast/slow | n | k_shape | LRT k≠1 p | KS-vs-exp p |
|---|---|---|---|---|
| 50/200 | 21 | 1.15 | 0.438 | 0.859 |
| 20/50 | 118 | 1.24 | 0.004 | 0.025 |
| 10/30 | 185 | 1.29 | 0.000 | 0.002 |
| 5/20 | 331 | 1.12 | 0.007 | 0.074 |
| 21/63 | 94 | 1.01 | 0.949 | 0.755 |

The medium configs **reject** strict exponentiality (k≈1.24–1.29, p<0.01) — this *looked* like
a promising "aging" signature initially.

### 3.3 Model-free empirical hazard (the decisive check)

For each config the **empirical flip-hazard vs regime age** was estimated directly:

| fast/slow | rho(hazard vs age) | p |
|---|---|---|
| 20/50 | +0.156 | 0.537 |
| 10/30 | −0.064 | 0.829 |
| 5/20 | +0.006 | 0.987 |

**No config shows a significant monotone age trend.** The hazard bounces between ~0.03 and
~0.38 with age but the trend is statistically indistinguishable from flat (all p ≥ 0.54).
The Weibull k>1 was a **parametric misfit artifact** — a boundary/left-truncation effect of
fitting Weibull to finite durations, not real duration dependence. Model-free the aging signal
is null.

### 3.4 Inter-flip interval regularity

| fast/slow | inter-flip CV | read |
|---|---|---|
| 50/200 | 0.89 | ≈ Poisson (no clustering) |
| 20/50 | 0.86 | ≈ Poisson |
| 10/30 | 0.84 | ≈ Poisson |
| 5/20 | 0.94 | ≈ Poisson |
| 21/63 | 1.05 | exactly Poisson |

Spacing is close to memoryless too — no burstiness (CV>1) that a "flip clusters then
recurrent flips are likely" rule could capture.

## 4. Verdict

**DEAD.** The falsifiable claim for Alternative C — "flip hazard is duration-dependent" — is
**rejected**. At every regime scale, SPY 2004–2026 trend-flip timing is **memoryless**:

1. Regime **age** gives no exploitable forecast: empirical hazard is flat vs age (all p≥0.54).
2. Flip **spacing** is Poisson/regular (CV≈0.84–1.05), no clustering to exploit.
3. The one parametric blush of "aging" (Weibull k≈1.2–1.3) is a **model artifact**, not a
   mechanism — the model-free hazard, which is what you'd actually trade on, is null.

There is **no sojourn structure to forecast**. A state-duration Markov / hazard overlay on SPY
is a dead end as a *standalone* leading-flip signal.

## 5. What this does and does not settle

- **Settles:** you cannot predict SPY trend flips from "how long the trend has been running."
  The flip clock is effectively memoryless over 21 years across 5 regime scales.
- **Does not settle:** whether flips are predictable from **current market conditions other
  than age** — volatility *level*, range, ADX, volume, cross-asset dispersion (Alternatives A
  and B). Age is the *one* variable (duration) that C isolated, and it is null; other
  covariates remain untested.

## 6. Next test recommendation (single A/B)

The age-variable is dead, but the harvestable sample is now proven large (118–331 flips). The
highest-information pivot, consistent with the "we just need a consistent way to forecast the
flip" goal, is **Alternative A — state-covariate hazard, not age**: model `P(flip in next h |
current vol, range, ADX, volume-shape, and *prior-flip recency*)` via a logistic/GLM hazard on
2004+ flips. This keeps the same flip-event target (proven sample-sufficient) but swaps the
null duration covariate for **contemporaneous condition covariates** that the two dead macro
families never measured.

If the condition-covariate hazard is also flat, that forces the honest conclusion that SPY
flip timing is unpredictable *in-sample at all* for 2004–2026, and the research should stop
chasing flip leads and refocus on non-flip overlays (level/beta tilt).
