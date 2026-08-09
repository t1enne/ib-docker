# A Novel Leading Indicator for Broad Equity Indexes

**Status:** Reference run implemented + follow-up A/B done. Flip-level lead validated, but
**downgraded REFINE/DEAD** by the high-n series-level test — the regime is a concurrent
macro-state classifier, not a consistent time-leading return predictor for SPY. See
`README.md` → Follow-up A/B and `2024-08_RDD-backtest-spec.md` §11.
**Scope:** Leverage the existing CPI deflator + HMM regime infrastructure
**Design constraint:** Avoid well-documented market-timing ratios (copper/gold, RSP/SPY vs cap-weight, plain HMM-on-CPI, CAMELS).

---

## 1. TL;DR

Propose a **leading, cash/risk-shifting regime** for a broad index (SPY basket) that is
built from macro fundamentals (inflation acceleration, real equity, real discount).

> **⚠ Empirical status (2026):** the *concept* (below) and the feature math are solid, but
> the **"leading" claim as tested was downgraded to REFINE/DEAD.** The high-n series-level
> test (`rdd_lead_lag.py`) shows the regime is a **contemporaneous macro-state filter, not a
> consistent time-leading return predictor** for SPY: no significant forward edge in the
> {3,20}d tradeable band, and the lead horizon shifts ~6× across sub-samples (7d early →
> 40d late). The original flip-level positive median lead was small-n coincidence. See the
> README Follow-up A/B and spec §11 before building on this.

The three inputs are the **CPI-deflated real price** of the index
(`cpi.deflated_log_prices` already exists), an **inflation-acceleration** gauge (a
cost-push proxy), and an inflation-adjusted **"real-discount"** feature that proxies
how much of the index move is a *multiple/discount-rate* repricing vs a *fundamental
earnings* repricing. They are fused by the existing `MarketRegimeHMM` into a small
number of hidden regimes. The **regime index itself is the leading indicator** — it is
designed to turn *ahead of* the index's own realized return, not coincident or lagged.

The distinctive, under-used angle is the **real-vs-nominal yield decomposition**: rather
than timing the index on its own momentum or volatility (both coincident), we classify
the *discount-rate flavor* of the tape — whether the price move is being "paid for" by
inflation (a discounting effect) or by genuine real-earnings re-rating. This is a
**leading** classification because discount-rate shifts lead the earnings and margin
changes that the index eventually prints.

> Why not the standard examples: copper/gold, RSP/SPY, HMM-on-absolute-CPI are all well
> documented and already crowded in the practitioner literature (see §7). This proposal
> deliberately uses a *joint, second-difference* inflation signal fused with a *real
> price gap* — a construction that appears in macro/asset-pricing research but is
> essentially not used as a standalone index-timing rule.

---

## 2. The signal in one sentence

**Detect whether the equity index is trading in a "cost-push / discount-regime" (bearish to
risk exposure) or a "real-earnings / carry-regime" (risk-on) state, and position
accordingly.**

The state is inferred (not asserted) from an HMM over:

- `acc = Δ log(CPI) = ∇²` — inflation *acceleration* (second difference of the price
  level), the textbook cost-push signal;
- `real = log(P_index) − log(CPI)` — the real (inflation-adjusted) equity level, i.e.
  the exact quantity `deflated_log_prices` computes;
- `rdisc = (nominal yield proxy) − (trailing CPI)` — a *real-rate / discount* proxy.

---

## 3. Why it is *leading* (the mechanism)

The claim "discount-rate shifts lead equity returns" is well grounded in asset-pricing
theory. Piazzesi & Schneider's *Equilibrium Yield Curves* (2007) show that inflation
expectations drive the real discount rate, and the discount rate drives asset prices
*before* the cash-flow (earnings) effects materialize. Mechanically:

1. **Cost-push inflation crests first.** Input costs rise → CPI acceleration turns
   positive → the real discount rate (nominal yield minus inflation) is squeezed and the
   *real* price of equities falls — even before earnings revisions hit.
2. **The margin squeeze shows up last.** Corporate margins are squeezed only after the
   input-cost shock has worked its way through. So the *real price gap* (real equity
   falling against still-high nominal equity) is a **lead** on the eventual earnings
   downgrades.
3. **Real price is the cleaner leading object.** `deflated_log_prices` removes the
   nominal illusion: an index that is flat in nominal terms but *falling in real terms*
   is de-risking roughly as fast as CPI is eating purchasing power — a signal nominal
   momentum alone cannot see.

Crucially the **regime is predictive, not reactive**: the HMM classifies the *joint* state
of {acceleration, real-price gap, real-discount}. Realized index returns are then
conditional on being *inside* one regime *before* turning. The lead arises because the
joint features move weeks ahead of the index's own trend/volatility which is what most
timing rules naively read.

---

## 4. Feature construction (formulas only — no code)

Let `CPI(t)` = daily-stepped CPI price-index level (already produced by
`load_cpi_price_index`), `P(t)` = close of the broad index basket, `Y(t)` = a nominal
yield proxy (e.g. 10y UST), all daily.

| Feature | Formula | What it captures | Direction to risk-on |
|---|---|---|---|
| `acc` | `log(CPI_t) − 2·log(CPI_{t-1}) + log(CPI_{t-2})` | Inflation *acceleration* (cost-push) | negative (`acc` easing) |
| `real` | `log(P_t) − log(CPI_t)` | Real (inflation-adjusted) equity | rising |
| `rdisc` | `Y_t − 252·(log CPI_t − log CPI_{t-k})/k` | Real discount / real-rate proxy | falling |

Notes:

- `real` is *literally* `deflated_log_prices(P, CPI)` — reuse as-is; it is z-score
  invariant, so the HMM scaling works without re-normalization.
- `acc` reuses the **same** CPI series, needing only a second difference — no new data
  loader, just a pandas `diff().diff()` on the daily-stepped index.
- `rdisc` is the one genuine new input (a yield series). It can be bootstrapped from
  `assets/` (see §6) or omitted for a CPI-only v0.

Standardize each feature to rolling z-scores (window ~252) before the HMM — this matches
how `MarketRegimeHMM._create_features` standardizes volatility/momentum internally.

---

## 5. The HMM regime → trade mapping

Feed the standardized `{acc, real, rdisc}` into the existing `MarketRegimeHMM`
(`n_regimes=2` or `3`, `covariance_type="diag"`, `random_state` fixed). The raw component
permutation is canonicalised by `rank_states_by_vol` — but note that canonicalisation
anchors by the **return-feature variance** which is not meaningful for these macro
features, so the label assignment must instead be solved by **regime → mean expected
forward return** over the training window (regime 0 is the one whose *forward* return is
best on average). This is the one deliberate deviation from reusing the stock HMM
remapping verbatim (see §9 risk).

Concretely, order states by `E[ r_{t→t+h} | state ]` (h ≈ 5–21) and relabel:

- **State 0 — "Real-carry" (risk-on):** `acc` easing / negative, `real` rising,
  `rdisc` moderate. → **long the index**, full size.
- **State 1 — "Discount compression" (risk-off):** `acc` accelerating, `real` flat/falling,
  `rdisc` rising. → **flat / reduced**, or short.
- **State 2 (optional) — "Transition":** ambiguous joint state. → **stand aside**.

State transitions are read from `predict` / `predict_proba`; use a **threshold on the
state probability** (e.g. `>= 0.6`) before acting — the same `should_trade` confidence
idea already present in the HMM.

---

## 6. Data requirements (gaps vs. current repo)

| Required | Present? | Where |
|---|---|---|
| CPI price-index level, daily-stepped | ✅ | `assets/cpi.csv` + `load_cpi_price_index` |
| Broad index basket OHLCV (SPY or equal proxies) | ✅ | already in local DB (`ibkr data query SPY`), 2014–2026; wire into backtest `universe`/`symbols` |
| Nominal 10y yield (for `rdisc`) | ✅ | `assets/dgs10.csv` (FRED `DGS10`), 1962–2026, FF'd on load like CPI — see `README.md` |

CPI and yields are **monthly/daily observational** but must be stepped and FF'd onto the
daily index grid exactly like `cpi.py` does (`reindex(... freq="D", method="ffill")`) so
the HMM sees a synchronous, cursor-safe long history. **Look-ahead safety:** the fact
that CPI is released with a lag, and that the loader forward-fills at the end, must be
preserved — the signal can only use values known up to the engine cursor. This is
already the library's convention (`state.candles`, cursor-truncation).

---

## 7. Literature anchors (and why the "obvious" ones are excluded)

The proposal is a **composite that is more than the sum of its parts** — each ingredient
is documented, but the fused timing rule is not a standard trade.

- **Lettau & Ludvigson (2001), "Consumption, Aggregate Wealth, and Expected Stock
  Returns" (JF)** — the `cay` variable is the canonical *academic* leading variable for
  equity excess returns; it decomposes price into fundamentals vs discount effects. Our
  `real`+`rdisc` is data-light cousin of the same accounting identity, made tradable
  without consumption/labor data.
- **Piazzesi & Schneider (2007), "Equilibrium Yield Curves"** — inflation *expectations*
  drive real discount rates ahead of cash flows; the direct theoretical backing for
  `rdisc` leading the index.
- **Eckstein (1983), *Core Inflation* / the Triangle model + the cost-push vs demand-pull
  taxonomy** (Okun; Friedman's critique) — `acc` (second difference of CPI) is the
  operational cost-push gauge; using a *second* difference rather than a level is what
  keeps the signal low-frequency/leading instead of coincident.
- **Okun's misery index construct** (unemployment + CPI together) — not a market-timing
  rule; cited only as a conceptual ancestor of fusing *macro rate/level* data into a
  single state.

Why we deliberately avoid:
- **Copper/gold, "Dr. Copper"** — well-documented practitioner timing ratio; commodity
  prices are notoriously noisy and pro-cyclical (coincident-to-leading at best).
- **RSP/SPY (equal vs cap-weight breadth)** — well-documented breadth rotation; it is a
  *relative-value* signal, not a *market*-timing signal, and requires two liquid equity
  ETFs plus careful beta-neutral handling.
- **Plain HMM on CPI level, or HMM on index returns/vol only** — the level is coincident;
  returns/vol HMM is the standard vol-regime, already implemented and not leading.

---

## 8. Backtest design within library idioms (no code — shape only)

- **Strategy module**: new `src/bt/strategies/<name>.py` with `STRATEGY_TYPE`, a frozen
  `Params(StrategyParams)`, `on_candle(...) -> list[TradeSignal]`, and `reset_global()`
  (the `GLOBAL`/`reset_global()` convention) for any cached rolling z-scores.
- **Gating**: reuse the phase of `momentum_regime`/`regime.gates` — read the regime label
  and *size* positions (or go flat) rather than always being in the market. The HMM
  regime is read from a `ModelState`-like field, updated by a model-updater, not hand-
  recomputed inside `on_candle`.
- **Cursor safety**: features read only from `state.candles` (cursor-truncated) plus the
  stepped CPI/yields series aligned to the same grid — no `resample_cache` access.
- **Signal rules**:
  - Enter long only in *Real-carry* state (regime 0) with state-prob ≥ threshold;
  - Go flat in *Transition* (ambiguous prob);
  - Reduce/short in *Discount-compression* state.
- **Validation** (the heart of "is it leading?"):
  - **Lead-time study**: cross-correlate the regime *flip dates* against future index
    returns; verify median flip leads the index drawdown/turn by N days. This is the
    falsifiable claim, not just in-sample Sharpe.
  - **Walk-forward/split**: IS/OOS windows with `_reset_strategy_state()` between them;
    the HMM refits OOS only on data up to window start.
  - **Random seed fixed** for deterministic HMM fits (`random_state=42` pattern).

---

## 9. Risks, limitations & open questions

1. **Regime relabel instability.** Anchoring by *forward return* is better than the
   stock variance-anchor for macro features, but forward-return ranks flip on refits.
   Mitigate with `state-prob` thresholds and short refit-horizon; cross-check agreement
   with the variance anchor.
2. **Low-frequency jumps.** CPI is monthly; stepping to daily makes the series st.step-
   like. The *second difference* amplifies month-boundary jumps. Smooth (e.g. HP filter
   or rolling mean) before z-scoring to avoid HMM flicker.
3. **Lag vs lead tension.** Release-lag of CPI is a *real, unavoidable* delay; if the
   price data ends 2026 but CPI ends 2024, the FF residual window dilutes `acc`. Validate
   the lead on a window where CPI is live.
4. **`rdisc` optionality.** Without a yield series, the model is CPI-only (loses pure
   rate-signal). Decide whether a 10y yield asset file is worth the add.
5. **Regime count.** 2 vs 3 states trade off clarity vs flexibility; test both.
6. **Overfit.** Three features × z-window × HMM regs × threshold is a big grid; keep the
   exploration free-parameter-light and lean on the *lead-time* validation as the guard
   against curve-fitting.
7. **Regime as *leading* is the testable claim.** Everything else (Sharpe, drawdown) is
   secondary; a leading indicator that loses a little Sharpe but demonstrably turns
   before the tape is the deliverable.

---

## 10. Next steps (research artifacts to produce next)

- `RIGDD-features.md` — full feature math, z-scoring, smoothing, and the release-lag
  FF mechanics (concrete, still no code).
- `RIGDD-backtest-spec.md` — strategy params, regime→sizing table, walk-forward design,
  and the lead-time validation metric definitions.
- **Asset status (done):** CPI present, SPY in local DB, `assets/dgs10.csv` downloaded from FRED — see `README.md`.
  Remaining is *engineering*: a `load_yields()` mirroring `load_cpi_price_index`, the `rdisc` feature, and the
  backtest universe wiring. An `assets-data.md` note captures the loader contract.

*(Wait for review of this proposal before proceeding to the deeper specs.)*
