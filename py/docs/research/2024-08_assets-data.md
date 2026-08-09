# Collected Assets — Loader Contract

**Research artifact #4** — exact formats and loader requirements for the data now on disk.
No code, but the precise contract a `load_*()` mirror of `cpi.py` must satisfy.

## Current state of `assets/`

| File | Series | Header | Rows | Range | Missing |
|---|---|---|---|---|---|
| `cpi.csv` | World Bank CPI annual rate | `Country,Country Code,Year,CPI` | 11,183 | all countries, to 2024 | n/a (annual) |
| `dgs10.csv` | FRED 10y UST constant maturity (pct) | `date,yld10` | 16,853 | 1962-01-02 → 2026-08-06 | 719 dates blank |

Index OHLCV (SPY) is **not** a file — it lives in the local IBKR DB (`ibkr data query SPY`,
20,513 × 1h bars, 2014-11 → 2026-07).

---

## `assets/dgs10.csv` contract

- Two columns: `date` (YYYY-MM-DD), `yld10` (yield in percent, float).
- Dates strictly unique; no duplicate rows.
- Missing prints are **blank** (`''`), not `NaN`-strings and not `' . '`.
- **Loader must:**
  1. parse `date` (→ `datetime`), coerce non-blank `yld10` to `float`;
  2. drop/FF blanks — **reindex to the target grid with `method="ffill"`** exactly like
     `load_cpi_price_index` steps CPI; DB-style daily gaps are single-day holidays;
  3. cache the parsed frame and expose the FF'd **daily level series** for reuse.
- Look-ahead stance is inherited: the series is *forward-filled*, so at cursor `τ` it
  naturally carries the latest known print — no peek.

## Reuse of the existing CPI loader

`load_cpi_price_index` already yields a daily-stepped, base-1.0 CPI level index —
reuse it unchanged for both the `acc` (2nd diff) and `real` (deflator) features. The new
`load_yields()` is the only new loader, and it is a structural clone (read + step + FF).

## Alignment rule (both series)

Both macro series must be reindexed to the **backtest date grid** (the SPY candle dates)
with `method="ffill"` before differencing, so `acc`, `real`, and `rdisc` are synchronous.
Any date in the backtest window before the earliest macro observation → treated as warmup
(NaN), consistent with the 252-window z-scoring warmup.

## Verification commands (already run)

```bash
uv run python - <<'PY'
import pandas as pd
df = pd.read_csv('assets/dgs10.csv', parse_dates=['date'])
assert df['date'].is_unique, "dates must be unique"
assert (df['date']>='2014-01-01').any() and (df['date']>='2026-07-01').any()
print(df.shape, df['yld10'].min(), df['yld10'].max())
PY
```
Checked: unique dates ✓; backtest-window coverage ✓; max null run in 2014+ = 1 day ✓.
