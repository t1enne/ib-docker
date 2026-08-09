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
