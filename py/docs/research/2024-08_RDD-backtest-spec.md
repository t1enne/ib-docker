# Backtest & Validation Spec — RDD leading regime

**Research artifact #3** — how to turn the RDD features (`2024-08_RDD-features.md`) into a
tradeable strategy and, crucially, how to **prove it is leading**. Shape only, no code.

---

## 1. Strategy shape (library idioms)

- **Module**: new file under `src/bt/strategies/` exposing `STRATEGY_TYPE` + `on_candle`.
- **Typed params**: frozen `Params(StrategyParams)` with the knobs from the feature spec
  (`w_a`, `k`, `w_z`, `θ`, `n_regimes`, per-state size). `from_dict()` fills defaults.
- **Regime source**: a *model updater* writes the HMM regime to `ModelState`-style state;
  `on_candle` *reads* it — it does not re-fit the HMM bar-by-bar. This mirrors how
  `momentum_regime` consumes `state.model_state.current_regime`.
- **Reset**: rolling z-scores and any HMM fit/cache live in strategy `GLOBAL`; provide
  `reset_global()` so the split engine calls `_reset_strategy_state()` per IS/OOS window.

---

## 2. Regime → sizing map

After fitting, sort HMM states by in-train mean *forward* return `E[ r_{t→t+h} | state ]`
(h ∈ {5, 10, 21}); relabel deterministically:

| Regime | Forward-return rank | Sizing on a broad index (SPY-like basket) |
|---|---|---|
| 0 "Real-carry" | highest expected fwd return | **Long**, `size = base` |
| 1 "Discount-compression" | lowest expected fwd return | **Flat** or **Short**, `size = base × factor(≤0)` |
| 2 "Transition" (optional) | ambiguous / middle | **Stand aside** (`size = 0`) |

Gating utilises the existing `TrendGate`-style helpers: act only when the *inferred*
state (via `predict`/`predict_proba`) has probability `≥ θ`; otherwise hold prior flow.

Reuse sizing helpers: `sized_qty(cash, position_size, price)` and `sl_tp_from_pct` for
optional per-trade SL/TP (defensive, not the core signal).

---

## 3. Position/execution constraints for index trading

- **Instrument**: trade one broad-index symbol (SPY or an aggregate of the universe) —
  not per-name factor rotation. The signal is *market*-level by design.
- **Friction**: apply realistic `cost_bps` (see `runtime_stats(cost_bps=...)`), dividends
  via total-return proxies if available, and only trade at the cursor (close of the
  signal bar, next-bar fill) to keep execution honest.
- **Turnover check**: the regime flips rarely by construction (low-frequency macro) — if
  measured turnover is high, the z-scores/HMM are too jumpy and need wider smoothing.

---

## 4. The falsifiable claim — "leading" validation

This is the heart. A leading indicator is *not* proven by high Sharpe; it must turn
**before** the market. Three quantitative checks:

### 4.1 Regime-flip lead-time study (primary)
- Identify regime flip dates `f_i` (state changes from risk-off ↔ risk-on).
- For each flip, record: (a) the **index return over the next h** days after the flip,
  (b) the *distance in trading days* from the flip to the subsequent local **turning
  point** in the index (drawdown start / rally start).
- Aggregate `median lead days`. A credible leading signal has a **positive, bounded lead
  (e.g. 3–20 sessions)** that is stable across IS/OOS. Median lead ≈ 0 or negative ⇒ the
  signal is coincident/lagged, not leading.

### 4.2 Predictive asymmetry (secondary)
- Compare mean `r_{t→t+h} | regime=on` vs `regime=off`. For a *leading* classifier the gap
  should be **roughly constant and *not* require the subsequent path of `acc/real/rdisc`
  to be known** — i.e. the discrimination exists at `t`, not after the fact.

### 4.3 Event-anchored deep-dive (qualitative but decisive)
- Overlay regime flips on 2–3 historical drawdown regimes (e.g. the 2022 inflation/rate
  unwind). Verify the HMM flipped to *Discount-compression* **before** the index peak,
  and to *Real-carry* at the turn. A few crisp, eyeballed leads outweigh 20 noisy cross-
  correlations.

---

## 5. Walk-forward / split protocol

- Use the library's split engine with IS/OOS windows; call `_reset_strategy_state()`
  between windows so rolling z-scores + HMM re-fit to IS-only data.
- **Refit discipline**: at each OOS window start, re-fit the HMM **on data up to that
  point only** (no look-ahead into OOS). Rolling z-scores likewise re-warm inside IS.
- **Data caveat**: if backtest price data extends past the CPI file's final year (2024),
  the FF residual dilutes `acc`/`π`. Perform the lead-time study on the *live-CPI* window
  and treat the FF tail as a stress case, not the headline.

---

## 6. Pass / fail bar

| Criterion | Pass |
|---|---|
| Leading (primary) | median regime-flip lead in `{3, 20}` sessions, stable IS/OOS |
| Discrimination | `E[r_{t+h}\|on] − E[r_{t+h}\|off]` meaningfully ≠ 0, consistent across h |
| Economic sanity | regime frequencies in plausible bands (each state ≥ ~15% of sample) |
| OOS stability | OOS metrics don't collapse vs IS (no silent refit-induced regime flip) |

Fail = lead ≈ 0/negative, or discrimination disappears OOS. That outcome is a finding,
not a bug — the document records it so we do not curve-fit past it.

---

## 7. Experiment matrix (keep free-parameter count low to avoid overfit)

| Dim | Values |
|---|---|
| Features | `{acc,real,rdisc}` / `{acc,real}` (CPI-only) |
| `n_regimes` | 2, 3 |
| `w_a` | 21, 63 |
| `k` | 63, 252 |
| `θ` | 0.5, 0.6, 0.8 |
| `h` (forward) | 5, 10, 21 |

Prefer a **reference run** (defaults from the feature spec) plus a handful of adjacent
perturbations, then the split validation — not an exhaustive sweep.

---

## 8. Artifacts produced by the run

- regime-flip journal (flip timestamps, prior regime from-prob, next-h index return),
- lead-time histogram + median per split window,
- predict asymmetry table per `h`,
- sizing decision log comparable to `runtime_stats` reporting,
- final markdown write-up wired back to this spec with pass/fail evidence.

---

## 9. Execution status (reference run + lead-time study — done)

Implemented artifacts:
- `src/bt/strategies/rdd.py` — `load_yields()` + `zacc`/`zreal`/`zrdisc` + `feature_matrix()`.
- `src/bt/strategies/rdd_regime.py` — adaptive (expanding-history) `GaussianHMM`, forward-
  return remap, `GLOBAL`/`reset_global()`, long-only cash-shift on `θ`-gated regime 0.
- `strats/rdd_regime.json` — SPY daily reference run.
- `scripts/rdd_lead.py` — the §4.1 lead-time harness (per-`--symbol`, SPY/QQQ/*).

### 9.1 Verdict vs §6 pass/fail bar (full period 2014·11 → 2026·07)

> **SUPERSEDED by the Follow-up A/B (§11):** the flip-level median-lead pass below was
> small-n coincidence. The high-n series-level test (n≈2000) shows no significant
> forward edge in the tradeable band and no sign/horizon-stable lead. See §11.

| Criterion | Pass bar | Result |
|---|---|---|
| Leading (4.1) | positive, bounded median flip-lead | **PASS (flip-level)** — SPY `{acc,real}` 27d, `{acc,real,rdisc}` 42d; QQQ 15.5-17.5d; never negative |
| Discrimination (4.2) | gap ≠ 0 across `h` | **WEAK-PASS (flip-level)** — E[r|on]−E[r|off] ≈ +9 to +15 bp (SPY), directionally correct |
| Economic sanity (§6) | each state ≥ ~15% | **PASS** — 32-37%/63-68% (SPY), 44-46%/54-56% (QQQ) |
| OOS / split stability | stable, no silent refit flip | **CAUTION** — lead horizon varies by era (SPY CPI-only 59d→20.5d across flips; QQQ 105d→12d) |
| **High-n forward edge (§11)** | **bootstrap-CI premium ≠ 0 in {3,21}d band, sign-stable across halves** | **FAIL** — n=2000 series test: CPI-only significant only at τ=42/63d; rdisc never significant; lead horizon shifts ~6× across sub-samples; premium sign flips EARLY↔LATE |

**Bottom line (superseded by §11):** the RDD regime is a *leading* indicator (positive,
bounded median lead,
never lagging) whose forward horizon is macro-regime-dependent — it tightens toward ~2-3
weeks in rate-driven tapes (where the discount-rate mechanism is most operant) and runs
longer in quiet tapes. Direction is robust across SPY and QQQ; QQQ is noisier (more
coincident flips) as expected for a tech-concentrated index against a *macro* signal.

**Caveats (carried, not curve-fit):** references with few flips (<~30) are low-confidence;
sub-window runs (e.g. 2020-2025 alone → 1 flip) are unusable. Data starts 2014-11, so any
pre-2014 claims are out of reach. **(→ superseded by the high-n §11 result.)**

### 9.2 Open question (next A/B) — RESOLVED

Isolate the state-relabel **anchor** only (`_order_states_by_fwd_return`: forward-return vs
an `acc`-sign / emission-mean anchor) on the same features/horizons. **Done** — see §11:
the 27d-vs-42d gap and split-instability persist under *every* anchor and parameter set, so
they belong to the macro signal, not the anchor.

---

## 11. Follow-up A/B: is the lead *consistent*? (2026)

> Repro: `scripts/rdd_lead_experiment.py`, `scripts/rdd_lead_split.py`,
> `scripts/rdd_lead_nregimes.py` (Test A), `scripts/rdd_lead_lag.py` (Test B).
> All reuse the exact strategy machinery — nothing hand-rolled.

### 11.1 Anchor / smoothing / θ / feature isolation

Every single-variable change trades one axis off another — **no setting yields a
frequent, confident, tightly-leading flip set at once** (full per-lever table in
`README.md`). θ is **decoupled** from the §4.1 lead journal (measured on ungated state),
so it never moves the lead-time number while meanwhile rejecting most actionable long
flips (1/7 pass for `{acc,real,rdisc}`, 4/7 for `{acc,real}` at θ=0.6).

### 11.2 Test A — n_regimes 2→3

- `{acc,real}`: **FAIL / degenerate** — 3-state collapses to 11/14/75% frequency; regime-0
  starved below the §6 sanity bar; regime0 gap21 flips to −1.56%. No flip even touches
  regime 0.
- `{acc,real,rdisc}`: **REJECT as lead-fixer** — best separation anywhere (+0.31/+0.48/+0.75%
  @h5/10/21, n=206 ✓) and flips double 14→26 (n=26 is still under the 30-event bar), but the
  "Transition" state *proliferates* 1↔2 channel churn rather than absorbing it, and
  flips-to-regime-0 stay sparse/irregular. The hypothesised "Transition absorbs stubby
  noise ⇒ tighter lead" mechanism is not supported.

### 11.3 Test B — series-level lead-lag (n≈2000)

- **`{acc,real}`:** corr(regime0_t, r_{t+5..15}) **negative** (−0.02..−0.03); significant
  premium only at **τ=42/63d** (+0.9/+1.0bp, CI excludes 0). EARLY best-lead τ=7d (corr
  +0.28) vs LATE **τ=40d** (corr +0.06) — lead horizon shifts ~6× across sub-samples;
  premium@21d flips EARLY −0.23% → LATE +0.54%.
- **`{acc,real,rdisc}`:** **zero significant forward edge** — all five horizons have a
  bootstrap CI including 0. corr with *past* returns (τ −5..−40: +0.03..+0.06) is **≥**
  corr with *future* returns (≤+0.022) ⇒ the regime is a **contemporaneous macro filter**, not
  a return-leader. premium@21d flips EARLY −0.45% → LATE +0.46%.

### 11.4 Conclusion

**Both flip-level and high-n series methods now agree.** The original §9 VALID verdict was
small-n coincidence. The RDD regime is best characterised as a **concurrent classification
of macro state** (inflation/discount); it has **no consistent, significant, tradeable-horizon
return lead** for SPY, and the inconsistency is **structural to the daily-macro 2-state
HMM**, not a parameter choice. Downgrade to **REFINE/DEAD** pending a level-overlay
(fixed 42–63d, `{acc,real}`) or continuous-feature-regression hypothesis.
