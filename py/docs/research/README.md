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

### Still to wrap (engineering, not collection)
1. Broad index universe: wire the existing local SPY bars into the backtest
   symbols/universe (already queryable, needs a config).
2. Loaders: add a `load_yields()` mirroring `load_cpi_price_index` (read + step + FF),
   and the `rdisc` feature in the RDD feature pipeline.
3. Without yields, the **CPI-only variant** (`{acc, real}`) is live immediately.

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

## Next step

Review `2024-08_leading-indicator.md`; on approval, close the two data gaps and build the
reference run per `2024-08_RDD-backtest-spec.md`.
