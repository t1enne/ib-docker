# Research: a Novel Leading Indicator for Broad Indexes

Proposal for a **leading** (risk-shifting) indicator that trades a broad equity index
using the library's existing CPI deflator + HMM regime stack — deliberately avoiding
well-documented timing ratios (copper/gold, RSP/SPY breadth, plain HMM-on-CPI).

## Artifacts

| File | Content |
|---|---|
| `2024-08_leading-indicator.md` | Concept, mechanism, why it leads, data gaps, risks. Start here. |
| `2024-08_RDD-features.md` | Concrete feature math (`acc`, `real`, `rdisc`), z-scoring, smoothing, cursor-safety. |
| `2024-08_RDD-backtest-spec.md` | Regime→sizing map, walk-forward protocol, and the **lead-time validation** that proves "leading". |
| `2024-08_assets-data.md` | Loader contract for `assets/dgs10.csv` + reuse of the CPI loader. |
| `2024-08_real-economy-lead.md` | **2026 follow-up.** Scan of the added real-economy FRED data (GDP, payrolls, IP, TU, UNRATE) for a tradeable lead. Null result under autocorrelation-honest testing. |

## Core idea in one line

Run the existing `MarketRegimeHMM` over a joint set — inflation **acceleration**
(CPI 2nd-diff), **real equity** (`P`/`CPI`), and a **real-discount** proxy — and time the
index by the inferred regime, whose *flips lead* the index's own drawdowns because
discount-rate / cost-push moves precede margin and earnings revisions.

## Library reuse

- `load_cpi_price_index` / `deflated_log_prices` — **reused as-is** (the `real` feature is
  literally the existing `real_log` series).
- `MarketRegimeHMM` + `rank_states_by_vol` / `should_trade` — **reused**, with one
  deliberate change: macro features should be relabelled by *forward return* not return-
  variance (see lead doc §9).
- `regime.gates` / `momentum_regime` pattern + `GLOBAL` / `reset_global()` convention —
  structural template for gating + rebasing on split.

## Data status (collection done)

| Data | Status | File | Shape | Coverage |
|---|---|---|---|---|
| CPI (annual rate) | ✅ already present | `assets/cpi.csv` | World Bank style | to 2024 |
| Broad index OHLCV | ✅ local DB (`ibkr data query SPY`) | DB, 1h bars | 20,513 rows | 2014-11 → 2026-07 |
| **10y UST yield** | ✅ **downloaded** | `assets/dgs10.csv` | `date,yld10` (pct) | 1962-01 → 2026-08 |

### `assets/dgs10.csv` — format & provenance
- **Source:** FRED series `DGS10` (10-Year Treasury Constant Maturity Rate), pulled
  via the FRED API (`api_key=${FRED_API_KEY}`) with `file_type=csv`.
- **Header:** `date,yld10` — `date` YYYY-MM-DD, `yld10` = yield in percent (`float`).
- **Missing values:** blank (`''`) for the ~719 dates FRED has no print for; inside the
  2014+ backtest window every gap is an isolated 1-day holiday. **Loader must
  forward-fill** (`reindex(method="ffill")`), exactly like `cpi.py`. No duplicates;
  dates unique.
- **Sanity:** yld range 0.52 (2020) → 15.84 (1980s); 2022 rate-rise captured
  (1.63 → 4.25); 2024 avg 4.21%; 2026 avg 4.36%.
- **Usage:** feeds the `rdisc` feature = `yld10 − (annualised trailing CPI)`.

### Engineering status (done)
Reference run implemented and validated end-to-end. Files:
- `src/bt/strategies/rdd.py` — `load_yields()` (mirrors `load_cpi_price_index`;
  blank→NaN→daily-ffill of `assets/dgs10.csv`) + pure feature functions `zacc`,
  `zreal`, `zrdisc`, and joint `feature_matrix()`.
- `src/bt/strategies/rdd_regime.py` — `STRATEGY_TYPE="rdd_regime"`, adaptive
  (expanding-history, no window cap) `GaussianHMM`, forward-return state remap,
  `GLOBAL` + `reset_global()`. Params: `w_a=63, k=252, w_z=252, θ=0.6,
  n_regimes=2, h=10, warmup_bars=64, min_train_size=252, retrain_interval=50`.
- `strats/rdd_regime.json` — SPY daily reference run, live-CPI window.
- `scripts/rdd_lead.py` — the **lead-time validation harness** (spec §4): computes
  the inferred regime per bar, screens 1-2 bar flicker, detects persistent flips,
  and reports median lead-time + distribution + split-stability + discrimination
  gap, optionally per `--symbol` (SPY/QQQ/*).
- `scripts/rdd_lead_experiment.py` — frozen-envelope single-variable A/B (anchor, h,
  θ, persistence, feature-set) for lead-consistency.
- `scripts/rdd_lead_split.py` — lead-consistency across time sub-samples (EARLY/LATE).
- `scripts/rdd_lead_nregimes.py` — Test A: n_regimes 2→3 "Transition" A/B.
- `scripts/rdd_lead_lag.py` — Test B: series-level lead-lag corr + bootstrap premium.
- `src/bt/strategies/tests/test_rdd_regime.py` — 14 tests (loader blank-ffill,
  all 3 features, z-scoring edge cases, HMM remap determinism/ordering,
  long-only/exit gating).

## Findings (reference run + lead-time validation, full period 2014·11 → 2026·07)

> **⚠ This § is the *original, flip-level* results and is SUPERSEDED by the Follow-up
> A/B below.** The VALID verdict was small-n coincidence; the high-n series-level test
> downgrades the claim. Read the Follow-up A/B before relying on any of this.

**Verdict: VALID as a *leading* indicator — direction robust, horizon regime-dependent.**
The RDD regime turns *ahead* of index turning points (never lagging) across assets;
its forward horizon is not constant but tightens in the rate-driven era.

| Asset / features | persistent flips | median lead | lead ≤20d | coincident (<3d) | split (1st→2nd half) |
|---|---|---|---|---|---|
| SPY `{acc,real}` | 15 | **27 days** | 46% | 13% | 59 → 20.5d |
| SPY `{acc,real,rdisc}` | 15 | **42 days** | 34% | 27% | 29 → 42.5d |
| QQQ `{acc,real}` | 10 | 15.5 days | 40% | 30% | 105 → 12d |
| QQQ `{acc,real,rdisc}` | 16 | 17.5 days | 56% | **44%** | 1.5 → 29.5d |

Key reads:
1. **Leading, not coincident/lagged** — median lead is positive in every window that
   has a usable flip sample (SPY 27-42 days; QQQ 15.5-17.5). No variant shows a mean
   negative (lagging) lead.
2. **CPI-only is the tighter signal; `{acc,real,rdisc}` stretches the lead and widens
   its dispersion** (SPY 42d median, Q3 ≈ 140d vs CPI-only Q3 ≈ 80d).
3. **SPY is the cleaner environment than QQQ.** QQQ's tech-idiosyncratic turning
   points add 30-44% coincident flips (SPY 13-27%) and erode the lead's reliability.
   This is consistent with the mechanism: a discount-rate/inflation signal fits a
   broad macro-dominated index better than a tech-concentrated one.
4. **The lead is macro-regime-dependent, not universal.** On the early (quiet-tape)
   sub-sample the median was far longer; it tightens toward ~2-3 weeks in the
   rate-driven era, where the mechanism is most operant. This is the mechanism
   working as theorized rather than a flaw — but it is the main caveat to carry.

**Sampling caveat:** references with few persistent flips (QQQ ~10-16, sub-window runs
with 1 flip) sit under the 30-event bar and are low-confidence — flagged, not curve-fit.

## Follow-up A/B (2026): does the lead hold up under controlled single-variable A/B?

**TL;DR — the original VALID verdict was small-n coincidence. Both the
flip-level and the high-n series-level tests now characterise the RDD regime as a
*concurrent macro-state classifier*, not a *consistent time-leading return predictor*
for SPY.** The inconsistency is **structural** to the daily-macro 2-state HMM, not a
parameter choice. The original §9 verdict is downgraded REFINE/DEAD pending a
level-overlay (not flip-timing) hypothesis.

Repro: ``scripts/rdd_lead_nregimes.py`` (Test A), ``scripts/rdd_lead_lag.py`` (Test B),
``scripts/rdd_lead_experiment.py`` (probe), ``scripts/rdd_lead_split.py`` (split probe),
``scripts/rdd_lead.py`` (original harness). All reuse the exact strategy machinery —
nothing hand-rolled.

### Anchor, smoothing, θ, feature isolation (probe)
Frozen-envelope single-variable A/B showed **every lever trades one axis off another**:

| Lever | Effect on lead consistency |
|---|---|
| Feature set `{acc,real}` vs `{acc,real,rdisc}` | CPI-only: faster θ-pass (1→4 of 7 long flips) + tighter median, but **unstable across time splits** (26d full vs 70–130d halves) |
| Anchor fwd-return vs emission-mean / acc-sign | emission-mean maxes separation (+1.21% gap21) but *extends* lead (med 70d, >60%=56%); acc-sign kills signal (negative gap) |
| Smoothing `w_z` | 126 fastest pass-rate (5/6) but drifts median; 63 → noise (Q3≈135d) |
| Persistence `min_run` | 1 → 41 flips/16d median but θ-pass collapses 5/20 |
| `θ` gate | **Decoupled from raw lead-time study** (the §4.1 journal is computed on ungated state) — rejects most actionable long flips without changing the measured lead |

**No knob set gives a frequent, confident, tightly-leading flip set simultaneously.**

### Test A — n_regimes 2→3 ("Transition" state)

| Feature set | n=2 regime freq | n=3 regime freq | n=3 regime0 gap21 | effect |
|---|---|---|---|---|
| `{acc,real}` | 34/66% | **11/14/75%** | **−1.56%** | **FAIL** — 3rd state collapses to 75% catch-all, regime-0 starved (11%, below ~15% sanity bar), discrimination flips strongly negative |
| `{acc,real,rdisc}` | 35/65% | 10/45/45% | **+0.75%** | **REJECT as lead-fixer** — best monotone separation seen anywhere (+0.31/+0.48/+0.75 @h5/10/21, n=206 ✓) and flips double 14→26 (helps n<30 caveat), **but** Transition *proliferates* 1↔2 channel churn instead of absorbing it; flips-to-0 stay sparse |

### Test B — series-level lead-lag (n≈2000, bypasses the 10–26 flip sample)

**`{acc,real}`:** no short-horizon lead (corr(regime0, r_{+5..15}) **negative**, −0.02..−0.03);
significant premium only at **τ=42/63d** (+0.9/+1.0bp, CI excludes 0). Lead horizon shifts
**~6× across sub-samples**: EARLY best-lead τ=7d (corr +0.28) vs LATE **τ=40d** (corr +0.06);
premium@21d flips EARLY −0.23% → LATE +0.54%.

**`{acc,real,rdisc}`:** **zero significant forward edge** — all five horizons (h=5…63)
have a bootstrap CI including 0 (0 pos / 0 neg / 5 ambiguous). corr with *past* returns
(τ −5..−40: +0.03..+0.06) is **≥** corr with *future* returns (≤+0.022) → the regime is a
**contemporaneous macro filter**, not a return-leader. premium@21d flips EARLY −0.45% →
LATE +0.46%.

**Both methods now agree:** the original positive median lead was small-n coincidence.
Neither the feature set, anchor, n_regimes, θ, nor smoothing yields a regime-0 state with
(a) a significant edge in the tradeable {3,20}d band or (b) a sign/horizon-stable lead
across time. The underlying signal has no stable time-leading return component for SPY.

## Literature anchors

- Lettau & Ludvigson (2001), *Consumption, Aggregate Wealth, and Expected Stock Returns*,
  J. Finance — `cay`, the canonical academic leading variable for equity excess returns;
  our real-price/real-discount is a data-light cousin of the same price-vs-fundamental
  decomposition.
- Piazzesi & Schneider (2007), *Equilibrium Yield Curves*, QJE/BER — inflation
  expectations drive real discount rates ahead of cash flows (the theoretical lead).
- Eckstein (1983) *Core Inflation* + the Triangle-model cost-push/demand-pull taxonomy —
  2nd-diff CPI as the operational cost-push gauge.
- Okun's misery index (construct) — conceptual ancestor of fusing a rates level and an
  inflation level into a single state; not itself a market-timing rule.

## Explicitly out of scope (well-documented)

Copper/gold ("Dr. Copper"), RSP/SPY equal-vs-cap-weight breadth, HMM on the CPI *level*
or on index returns/vol in isolation.

## Next step (single A/B, not a sweep)

The state-relabel anchor toggle and the n_regimes + smoothing perturbations have now
been run (see Follow-up A/B above): **anchor is not the driver** — the lead instability
persists under every anchor and parameter set, so it belongs to the macro signal. The
highest-information next tests are either:

- **Level-overlay (not flip-timing):** drop the "flips lead turning points" framing and
test whether the regime-0 *level* adds value as a low-beta overlay at a **fixed 42–63d
horizon (CPI-only)** — the only window where a significant premium survives (+0.9/+1.0bp).
New falsifiable claim: level-selection, not flip-timing.
- **Continuous-feature regression:** the rdisc edge being ~0 at series scale suggests the
discount leg is inert; test dropping the HMM and regressing forward returns directly on
the *continuous* composite (`z_acc`/`z_rdisc`) to see whether the features themselves,
rather than the discretised regime, carry any consistent lead.

---

## Addendum (2026): the added real-economy data — no consistent lead

The RDD line used only price/inflation data (CPI, yields). Since then real-economy FRED
series were collected: `payems`, `indpro`, `tcu`, `unrate`, `gdpc1`/`gdp`, `bopgstb`
(loaders in `src/indicators/macro/`). `2024-08_real-economy-lead.md` + `scripts/lead_scan.py`
tests them for a tradeable {3,21}d leading component for SPY.

**Result: null.** Every candidate (output gap, payroll/IP momentum, capacity utilisation,
Sahm-style slack) — plus the original `acc`/`rdisc` controls — falls to an
**autocorrelation-honest (stationary block) bootstrap**; none is significant in the
tradeable band. The most promising naive result (GDP output-gap +0.14 @21d) was an
**over-resampling artifact**: monthly/quarterly prints was scored on ~4300 daily rows, and
a naive pairwise bootstrap declared it significant; a block bootstrap (p≈0.50) and
quarterly coarsening (sign flips negative at n=48) both kill it.

**Conclusions**
1. The negative RDD verdict now generalises **across both signal families** (price/rate and
   real-activity): for SPY 2014–2026, no daily-macro feature is a consistent tradeable lead.
2. **Any future macro-lead scan must gate on the block-bootstrap p**, not high-n daily
   correlation — the weekly/hourly-resampled FRED grid is a guaranteed false-positive
   machine otherwise.
3. The sign/horizon of the output gap *is* directionally right and monotone (stronger at
   42/63d) — but confirming it needs a **longer price history** (≥1990s) to lift the
   quarterly effective sample (n≈48 → ~110), not more features or a different HMM.
