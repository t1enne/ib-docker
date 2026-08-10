# Real-Economy Leading-Indicator Scan (2026) — the added macro data

**Status:** Scan implemented + run (single-feature **and** multi-feature/composite).
**No consistent leading indicator found in the newly added real-economy FRED data** under
autocorrelation-honest testing. This is a *finding*,
not a failure: it closes the "different signal family" hypothesis the RDD line raised, and
it did so with a cleaner methodology than the original RDD validation.

Repro: `scripts/lead_scan.py` (autocorrelation-honest lead/lag scan).

---

## 1. Motivation

The prior RDD line (`docs/research/README.md`, `2024-08_leading-indicator.md`) tested an
HMM over *inflation/discount* features `{acc, real, rdisc}` and downgraded it to
REFINE/DEAD: at series scale it was a **contemporaneous macro filter, not a return-leader**
(leads only at non-tradeable τ=42/63d, sign/horizon unstable across splits, no significant
edge in the {3,21}d tradeable band).

The proposed pivot — now that real-economy FRED data (payrolls, industrial production,
capacity utilisation, unemployment, GDP, trade balance) exists in `assets/` — was:

> **Hypothesis.** A *real-economy activity* signal (output gap, payroll/IP momentum,
> capacity-utilisation turning points, Sahm-style labour slack) has a **consistent,
> stable, tradeable {3,21}d leading component for SPY**, unlike the failed price/inflation
> (discount/rate) family.

Mechanism: real activity is the classical forward driver of *earnings* (protracted growth
or contraction works into margins with a lag), whereas the prior `rdisc`/real-price family
is a *discount-rate* repricing that is contemporaneous with the equity tape.

## 2. Method (and the correction to how the RDD lead was scored)

The scan reproduces the RDD *series-level* discipline (lead-lag correlation, leader-vs-
filter, EARLY/LATE split) plus — critically — an **autocorrelation-honest significance
test that the original RDD run did not apply at series level**.

The key methodological finding is a **over-resampling trap**:

- FRED monthly/quarterly macros are forward-filled onto a **daily** grid by the loader.
- Scoring correlations on that daily grid inflates n from ~48 (quarters) / ~120 (months)
  to ~4300 daily "observations" of the same print.
- A **naive pairwise bootstrap then declares almost everything significant** — including
  the GDP output-gap, which showed a seductive +0.11…+0.18 corr across the tradeable band
  with "p < 0.01".
- The **stationary block bootstrap** (sweep blocks of consecutive pairs, block=63d)
  collapses every candidate to p ≈ 0.44–0.82. **Nothing is significant.**

Two independent honesty checks agree:

| Check | GDP-gap result |
|---|---|
| Daily-grid corr + naive bootstrap | +0.137 (h=21), listed p<0.01 — **looks significant, is an artifact** |
| Block bootstrap (block=63d) | p = 0.52 (h=10), 0.50 (h=21) — **not significant** |
| Quarterly coarsening (n=48) | next-1Q −0.13, next-2Q −0.29 — **sign flips negative** |

## 3. Results — all candidates fall to the block bootstrap

Full table in the scan output. Per-feature forward correlation at h=21 and marker under
the autocorrelation-honest test (nothing earns `*`/`!`):

| Feature | corr@21d (naive) | block-bootstrap p | direction | leader-vs-filter |
|---|---|---|---|---|
| `gdp_gap_z` (output gap) | +0.137 | ~0.50 | most promising naive | LEADER @21 |
| `payems_gr_z` (payroll 12m gr) | +0.068 | ~0.52 | + | LEADER (weak) |
| `indpro_gr_z` (IP 12m gr) | +0.038 | ~0.83-h10 | + | FILTER |
| `tcu_*` (capacity ut) | +0.04 | >0.54 | + | FILTER |
| `sahm_z` / `unrate_d12` | −0.07 | >0.44 | − | FILTER |
| `acc_z` (inflation accel) | +0.031 | >0.05 | + | LEADER (weak, ns) |
| `rdisc_z` (real discount) | −0.077 | ~0.08 | − | FILTER |

**No candidate passes the block-bootstrap bar in the tradeable {3,21}d band.** The
apparent real-economy "leads" (output gap above all) are overlapping-observation
artifacts, not independent-sample truths.

## 4. Why this is the honest answer (not a failure of the data)

The negative result is **strong and informative**:

1. It **matches** the RDD series-level conclusion at every point — the real-economy family
   is as non-leading as the inflation/discount family once autocorrelation is respected.
2. It **closes the "wrong family" hypothesis**: the pivot from price/rate to real-activity
   was the most plausible remaining route to a macro lead, and it does not hold for SPY
   2014–2026 in the tradeable band.
3. It isolates the trap that would have produced a *false positive*: naive bootstrap on a
   daily grid. Any future macro-lead scan must gate on the block-bootstrap p, not the
   high-n corr.

## 5. Verdict

**REFINE / DEAD as a direct macro-lead for SPY.** The added real-economy data is not a
consistent leading indicator in the tradeable {3,21}d band at the effective (monthly/
quarterly) sample sizes available (2014–2026 ≈ 11 yrs — n≈130 monthly, n≈48 quarterly).
This is the same null the RDD line reached for the inflation/discount family, expressed
with a corrected significance test.

## 6. Combination scan (2026) — pairs/triples do not rescue it

`lead_scan` tested each macro *individually*. A reviewer correctly pushed back: a
**combination** of macros could carry a signal no single series does. That is the RDD
premise, so `scripts/combo_lead_scan.py` sweeps equal-weight **pairs and differences** (and
a few activity-vs-rates / activity-vs-inflation triples) over the standardised feature
pool, scoring each with the same block bootstrap, and — critically — estimates a
**multiple-testing null**: the best-p an empty search over the *same family size and
autocorrelation* would produce by chance.

***Result: the composite hypothesis is falsified too.***

| h | best p (block boot) | best composite                |
|---|---|---|---|
| 3 | 0.436 | `payems_gr + unrate_d12`     |
| 10 | 0.399 | `payems_gr − sahm`           |
| 21 | 0.465 | `gdp_gap − bopgstb`          |

- **The best composite p (~0.40–0.47) never approaches significance.** No pair exceeds
  the single-feature noise floor.
- **Multiple-testing null (the decisive number):** the best h=21 composite (p=0.465) sits
  at the **~40th percentile of the no-signal family-min null** (null min-p 5th pct ≈ 0.451,
  50th ≈ 0.471). A *random* empty composite search of equal size produces a best-p this
extreme or more **~60% of the time**. The "winner" is indistinguishable from noise.
- **Holdout:** the gdp_gap−bopgstb composite gets its +0.141 corr@21 from ONE half
  (EARLY p=1.000 / LATE p=0.441) — the same sign/horizon instability that sank the RDD
  regime. Not a stable lead.
- **Interpretation:** the block bootstrap on this data has a noise floor of best-p ≈ 0.45
  over a family-sized search. Every macro — single or combined, real-activity or
  price/rate — lands on that floor. There is no hidden composite lead under the effective
  (quarterly/monthly) sample size available.

> **Why no combination beats the parts:** the individual series are near-white-noise
> predictors with fat autocorrelation; equally-weighted sums of near-white noise stay
> near the same noise floor unless a *genuine* shared signal exists. No genuine shared
> signal is present in 2014–2026 SPY at tradeable horizons. The only route to a
> detectable composite (or single) lead is more independent macro cycles — i.e. a longer
> price history, not more feature arithmetic.

## 7. What would change the verdict (next A/B, single change))

The daily-grid correlation *is* positive and horizon-monotone for the output gap
(+0.04 → +0.14 → +0.22 at h=3/21/63) — the *sign/horizon* is right, only the significance
is spurious. The plausible routes forward, in descending informativeness:

- **Lengthen the price history** so the *quarterly* GDP-gap effective sample grows (the
  price DB starts 2014; a 1990s+ SPY history would give ~28 yrs ≈ n≈110 quarters, the
  only way to overcome the small effective n). This is the highest-value next step.
- **Test the lead as a low-frequency *regime overlay* (fixed 42–63d horizon, output gap
  only)** rather than {3,21}d flip-timing — the RDD line's own fallback, which requires a
  long effective sample to confirm and is explicitly **not a tradeable daily indicator**.
- Regress forward SPY returns directly on the *continuous* output-gap (not a discretised
  HMM), at quarterly frequency, over a 1990s+ history.

Do **not** re-attempt {3,21}d flip-timing on the current 2014–2026 window with these
features — the effective sample is too small to support a claim one way or the other, and
the naive number is a trap.
