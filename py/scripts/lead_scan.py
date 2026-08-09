"""Research harness: scan macro/constructed signals for a *consistent leading*
component against the SPY.

Context
-------
The prior RDD line of research (docs/research/README.md) showed that an HMM over
inflation/discount features (``{acc, real, rdisc}``) was a *contemporaneous* macro
classifier, not a time-leading return predictor: no significant forward edge in the
tradeable {3,21}d band, sign/horizon unstable across time splits. That failure was
diagnosed as structural to the *macro signal family* (nominal/price/inflation), not
to the method.

This harness tests the complementary, newly-collected **real-economy** family
(employment, output, capacity, GDP, labour-slack) for a consistent leading component,
using the same series-level discipline that caught the RDD small-n coincidence:

  - continuous predictors (no HMM discretisation loss), each standardised;
  - forward-return correlation with a stationary bootstrap CI (n large);
  - the tradeable-band check h in {3,5,10,21} (42/63 flagged as *not*tradeable);
  - the leader-vs-filter check: corr(past returns) vs corr(future returns);
  - sign/horizon stability across an EARLY/LATE split.

Release-lag honesty: the FRED CSVs are dated at the observation *print* date and the
loader forward-fills from that date. To guard against a subtle data-release artifact
where a monthly print dated t actually became public later, every predictor is also
re-scored with an extra ``drop_burn`` X-day shift (predictor withheld X extra days),
and the lead must survive that shift to be trusted.

Usage:  uv run python scripts/lead_scan.py  [--min-days 63]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj))

from src.bt.strategies.rdd import load_yields  # noqa: E402
from src.indicators.macro._shared import (  # noqa: E402
    load_cpi_price_index,
    load_daily,
)

FRED_DAILY = ["dgs10", "payems", "unrate", "indpro", "tcu", "gdpc1", "gdp", "bopgstb"]
TRADEABLE = {3, 5, 10, 21}
LONG_ONLY_HORIZONS = {42, 63}  # prior significant band; reported as non-tradeable


# --------------------------------------------------------------------------- #
# autocorrelation-honest statistics (the *primary* significance test)
# --------------------------------------------------------------------------- #


def block_bootstrap_p(
    a: np.ndarray,
    b: np.ndarray,
    block: int = 63,
    iter: int = 4000,
    seed: int = 7,
) -> float:
    """Stationary block bootstrap p-value respecting serial autocorrelation.

    Naive pairwise reshuffling overstates significance when monthly/quarterly
    macros are over-resampled to a daily grid (n balloons from ~48 to ~4000 and
    overlapping prints masquerade as independent observations). A pairing that
    sweeps blocks of *consecutive* observations preserves the dependence and is
    the honest significance test for low-frequency macro signals.
    """
    r_obs = float(np.corrcoef(a, b)[0, 1])
    n = len(a)
    if n < 2 * block:
        return 1.0
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / block))
    exceed = 0
    for _ in range(iter):
        starts = rng.integers(0, n - block, size=nblk)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        rr = float(np.corrcoef(a[idx], b[idx])[0, 1])
        if abs(rr) >= abs(r_obs):
            exceed += 1
    return exceed / iter


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #


def load_spy_daily(bar: str = "1d") -> pd.Series:
    """SPY daily close series (resampled from the 1h local DB), 2014-11 → 2026-07."""
    import sqlite3

    con = sqlite3.connect("../data/db.sqlite")
    q = """select c.timestamp, c.close from candle c
           join symbol s on c.conid=s.conid
           where s.ticker='SPY' order by c.timestamp"""
    df = pd.read_sql_query(q, con)
    con.close()
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    s = df.set_index("Date")["close"].sort_index()
    return s.resample("1D").last().dropna() if bar == "1d" else s


def macro_frame() -> pd.DataFrame:
    """Daily-grid frame of raw macro levels, forward-filled (cursor-safe)."""
    idx_start = pd.Timestamp("2014-01-01")
    idx_end = pd.Timestamp("2026-08-31")
    grid = pd.date_range(idx_start, idx_end, freq="D")
    cols: dict[str, pd.Series] = {}
    for name in FRED_DAILY:
        if name == "dgs10":
            cols["yld10"] = load_yields().reindex(grid, method="ffill")
        else:
            cols[name] = load_daily(name, source_dir="assets").reindex(
                grid, method="ffill"
            )
    cpi = load_cpi_price_index().reindex(grid, method="ffill")
    cols["cpi"] = cpi
    return pd.DataFrame(cols).dropna(how="all")


# --------------------------------------------------------------------------- #
# feature transforms -> standardized predictor per bar
# --------------------------------------------------------------------------- #


def _z(series: pd.Series, lookback: int = 252, minp: int = 63) -> pd.Series:
    mu = series.rolling(lookback, min_periods=minp).mean()
    sd = series.rolling(lookback, min_periods=minp).std()
    return (series - mu) / sd


def build_features(m: pd.DataFrame, a: int = 63, k: int = 252) -> pd.DataFrame:
    """Return standardised candidate predictors (one column each, cursor-safe)."""
    f = pd.DataFrame(index=m.index)
    lp = np.log(m["cpi"].replace(0, np.nan))

    # ---- real-economy family (the new data) ----
    # payrolls: 12-mo % growth then z
    f["payems_gr_z"] = _z(m["payems"].pct_change(12).replace([np.inf, -np.inf], np.nan))
    # payrolls acceleration (2nd diff of log)
    f["payems_acc_z"] = _z(np.log(m["payems"]).diff().diff())
    # industrial production 12-mo growth + acceleration
    f["indpro_gr_z"] = _z(m["indpro"].pct_change(12).replace([np.inf, -np.inf], np.nan))
    f["indpro_acc_z"] = _z(np.log(m["indpro"]).diff().diff())
    # capacity utilization level (already a 0-100 bound series) + momentum
    f["tcu_level_z"] = _z(m["tcu"].diff(12))
    f["tcu_lev"] = _z(m["tcu"])
    # unemployment: inverse-ish -> 12-mo *change* (rising slack = bearish)
    f["unrate_d12_z"] = _z(m["unrate"].diff(12))
    # Sahm-style: 3-mo avg minus 12-mo min of unemployment (recession trigger)
    u = m["unrate"]
    sahm = (u.rolling(3).mean()) - (u.rolling(12).min())
    f["sahm_z"] = _z(sahm)
    # output gap proxy: (GDP index - trend) recession gauge
    g = np.log(m["gdpc1"])
    f["gdp_gap_z"] = _z(g - g.rolling(252).mean())
    # trade balance (monthly, noisy) * 1 to show it's weak
    f["bopgstb_z"] = _z(m["bopgstb"])

    # ---- prior nominal/inflation family (baseline controls) ----
    _r = np.log(m["yld10"])
    f["acc_z"] = _z(lp.diff().diff().rolling(a).mean())  # inflation accel
    f["rdisc_z"] = _z(m["yld10"] - 252 * (lp - lp.shift(k)) / k)  # real discount
    return f


def _forward_returns(close: pd.Series, h: int) -> pd.Series:
    _ = np.log(close).diff()  # confirm log-prices; unused (shift-lag already log)
    return np.log(close.shift(-h)) - np.log(close)


# --------------------------------------------------------------------------- #
# series-level lead / lag statistics with stationary bootstrap
# --------------------------------------------------------------------------- #


def score_predictor(
    X: pd.Series,
    close: pd.Series,
    horizons: tuple[int, ...],
    drop_burn: int = 0,
) -> pd.DataFrame:
    """For each horizon, corr(X_t, r_{t+h}) and the block-bootstrap p-value.

    Uses the autocorrelation-honest block bootstrap (see ``block_bootstrap_p``).
    """
    close = close.dropna()
    v = pd.Series(close).reindex(X.index).ffill()
    rows = []
    for h in horizons:
        fr = _forward_returns(v, h)
        d = pd.concat([X, fr, v], axis=1, keys=["X", "r", "c"]).dropna()
        if drop_burn:
            d = d.iloc[drop_burn:] if len(d) > drop_burn else d.head(0)
        min_obs = 30
        if len(d) < min_obs:
            rows.append((h, np.nan, 1.0, len(d)))
            continue
        a = d["X"].to_numpy(dtype=float)
        b = d["r"].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1])
        p = block_bootstrap_p(a, b)
        rows.append((h, r, p, len(d)))
    return pd.DataFrame(rows, columns=["h", "corr", "pval", "n"])


def _do_split(
    close: pd.Series, X: pd.Series, horizons: tuple[int, ...]
) -> pd.DataFrame:
    t = X.index[int(len(X) * 0.5)]
    early = score_predictor(X[X.index <= t], close, horizons)
    late = score_predictor(X[X.index > t], close, horizons)
    early["split"], late["split"] = "EARLY", "LATE"
    return pd.concat([early, late], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap  # noqa  (options reserved for future sensitivity knobs)

    close = load_spy_daily()
    m = macro_frame()
    f = build_features(m)

    print(
        f"SPY daily bars: {len(close)}  ({close.index[0].date()} → {close.index[-1].date()})"
    )
    print(f"macro grid: {len(m)} days; feature cols: {list(f.columns)}\n")

    horizons = (3, 5, 10, 21, 42, 63)

    # FIRST PASS: no extra release-lag shift, full sample, per feature
    print("=" * 88)
    print(
        "FULL-SAMPLE series-level lead (tradeable {3,21}d | long-only flagged at 42/63d)"
    )
    print(
        "  corr>0 = forecasts HIGHER future return | sig *=+, !=-  under BLOCK-bootstrap (p<0.05)"
    )
    print(
        "  NOTE: naive high-n significance is misleading for low-freq macros; * & ! are honest."
    )
    print("=" * 88)
    header = f"{'feature':18s}" + "".join(f" h={h:<4d}corr  sig " for h in horizons)
    print(header)
    summary: dict[str, dict] = {}
    for col in f.columns:
        res = score_predictor(f[col], close, horizons)
        summary[col] = res
        cells = []
        for _, r in res.iterrows():
            h = int(r["h"])
            if r["n"] < 30:
                cells.append(f"    h={h} <30n ")
            else:
                corr = float(r["corr"])
                pv = float(r["pval"])
                sig = (
                    "*"
                    if (corr > 0 and pv < 0.05)
                    else ("!" if (corr < 0 and pv < 0.05) else ".")
                )
                cells.append(f" {corr:+.3f}  {sig}  ")
        print(f"{col:18s}" + "".join(cells))

    # LEADER-vs-FILTER: corr(X_t, past return) should NOT dominate corr(X_t, future)
    print("=" * 88)
    print("LEADER-vs-FILTER check @ h=21 (must be future > |past| to be a leader)")
    print("=" * 88)
    v = close.dropna()
    for col in f.columns:
        X = f[col]
        fr = _forward_returns(v, 21)
        past = np.log(v).diff().shift(1)
        d = pd.concat([X, fr, past], axis=1, keys=["X", "fut", "past"]).dropna()
        if len(d) < 30:
            print(f"  {col:16s}  too few obs")
            continue
        cf = float(np.corrcoef(d["X"], d["fut"])[0, 1])
        cp = float(np.corrcoef(d["X"], d["past"])[0, 1])
        marker = "LEADER" if cf > abs(cp) else "FILTER"
        print(
            f"  {col:16s}  corr(future@21)={cf:+.3f}  corr(past)={cp:+.3f}   → {marker}"
        )

    # SECOND PASS: release-lag robustness (drop burn) on the top tradeable candidates
    def best_sig(df: pd.DataFrame) -> float:
        """Smallest p-value across the tradeable band (block-bootstrap p)."""
        band = df[df["h"].isin(TRADEABLE)]
        if band.empty or band["n"].min() < 30:
            return 1.0
        return float(band["pval"].min())

    # Nominal significance threshold after autocorrelation-honest bootstrap.
    SIG_BAR = 0.05
    top = [c for c in f.columns if best_sig(summary[c]) < SIG_BAR]
    top = list(dict.fromkeys(top))  # dedupe, preserve order
    if top:
        print("=" * 88)
        print(
            "RELEASE-LAG ROBUSTNESS (drop_burn in trading days: signal withheld X extra days)"
        )
        print("=" * 88)
        for col in top:
            row = []
            for db in (0, 20, 40):
                r = score_predictor(f[col], close, (21,), drop_burn=db).iloc[0]
                row.append(f"db={db}: corr_h21={r['corr']:+.3f} (n={r['n']})")
            print(f"  {col:16s}  " + "  ".join(row))

    # THIRD PASS: EARLY/LATE sign stability on the top candidate
    if top:
        print("=" * 88)
        print(
            "EARLY/LATE SPLIT stability — leading requires same SIGN and similar horizon"
        )
        print("=" * 88)
        for col in top:
            s = _do_split(close, f[col], horizons)
            e21 = float(s[(s.split == "EARLY") & (s.h == 21)]["corr"].iloc[0])
            l21 = float(s[(s.split == "LATE") & (s.h == 21)]["corr"].iloc[0])
            stable = (e21 > 0) == (l21 > 0)
            print(
                f"  {col:16s}  EARLY@21={e21:+.3f}  LATE@21={l21:+.3f}  "
                f"{'STABLE' if stable else 'UNSTABLE'}"
            )


if __name__ == "__main__":
    main()
