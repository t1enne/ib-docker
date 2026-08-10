"""Re-run of the documented RDD "HMM-over-macro" leading-indicator study on the
full SPY 2004+ history.

Context
-------
The original project (docs/research/README.md, 2024-08_RDD-backtest-spec.md)
fit a 2-state GaussianHMM over standardized macro features ``{acc, real,
(rdisc)}`` and claimed the resulting regime *leads* index turning points. The
2014+ run reported positive flip-level median leads (27–42d) but was later
downgraded to REFINE/DEAD by a high-n series-level test (concurrent macro
filter, not return-leader). The runnable harness (``scripts/rdd_lead.py``) was
never committed — only the pure feature module ``src/bt/strategies/rdd.py`` and
the ``MarketRegimeHMM`` survived.

This script reimplements the documented method faithfully and runs it on the
newly extended SPY daily sample (2004-01 → 2026-08, ~5660 bars — double the
macro cycles of the original 2014+ window):

  - expanding-history 2-state GaussianHMM over the feature frame, refit every
    ``retrain_interval`` bars on data up to the cursor only (no lookahead);
  - forward-return state relabel (regime 0 = higher mean forward return at
    ``h_anchor``), the documented deviation from ``rank_states_by_vol``;
  - persistent-flip extraction with a ``min_run`` flicker screen;
  - flip-level lead-time study (spec §4.1): median lead to the next index
    turning point + distribution + EARLY/LATE split stability;
  - discrimination gap (spec §4.2) per horizon;
  - series-level lead-lag (README Follow-up Test B) with the
    autocorrelation-honest block bootstrap (mirrors ``scripts/lead_scan.py``).

Deterministic: fixed ``random_state`` for every HMM fit.

Usage::

    uv run python scripts/rdd_lead_2004.py [--fit-mode expanding|full]
                                           [--min-run 5] [--pivot-r 5]
                                           [--h-anchor 10] [--features all|cpi|cpi+rdisc]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn import hmm

_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj))

from src.bt.strategies.rdd import load_yields, zacc, zrdisc, zreal  # noqa: E402
from src.data.db import query_candles  # noqa: E402
from src.indicators.macro._shared import load_cpi_price_index  # noqa: E402
from scripts.lead_scan import block_bootstrap_p  # noqa: E402

HORIZONS = (3, 5, 10, 21, 42, 63)
W_A = 63
K = 252
W_Z = 252
MIN_TRAIN_SIZE = 252
RETRAIN_INTERVAL = 50
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# data + features
# --------------------------------------------------------------------------- #


def load_spy_daily() -> pd.Series:
    """SPY daily close series 2004+ from the local candle DB."""
    df = query_candles("SPY", None, None, "1D")
    return df["close"].dropna()


def build_aligned_features(
    close: pd.Series,
    cpi: pd.Series,
    yld: pd.Series,
    with_rdisc: bool,
    w_a: int = W_A,
    k: int = K,
    w_z: int = W_Z,
) -> pd.DataFrame:
    """Standardized feature frame aligned to the ``close`` index (NaN warmup kept).

    Mirrors ``feature_matrix`` but preserves the price grid so the expanding
    refit can slice by cursor position without losing index alignment.
    """
    grid = close.index
    cpi_g = cpi.reindex(grid, method="ffill")
    cols: dict[str, pd.Series] = {
        "z_acc": zacc(cpi_g, w_a=w_a, w_z=w_z),
        "z_real": zreal(close, cpi_g, w_z=w_z),
    }
    if with_rdisc and yld is not None and not yld.empty:
        cols["z_rdisc"] = zrdisc(yld, cpi_g, k=k, w_z=w_z)
    return pd.DataFrame(cols, index=grid)


# --------------------------------------------------------------------------- #
# HMM regime inference (expanding refit, forward-return relabel, cursor-safe)
# --------------------------------------------------------------------------- #


def _remap_by_fwd_return(
    states: np.ndarray, fwd: np.ndarray, n_regimes: int
) -> dict[int, int]:
    """Relabel raw HMM states by mean forward return (regime 0 = higher mean).

    ``fwd`` must be date-aligned to ``states`` (same length/order). NaN forward
    returns (rows whose h-window is not observable inside the training window)
    are excluded. Deterministic tie-break by state index.
    """
    means: list[float] = []
    for s in range(n_regimes):
        vals = fwd[states == s]
        vals = vals[np.isfinite(vals)]
        means.append(float(vals.mean()) if vals.size else -np.inf)
    order = sorted(range(n_regimes), key=lambda s: (-means[s], s))
    return {s: r for r, s in enumerate(order)}


def _match_new_to_old(
    new_states: np.ndarray, old_states: np.ndarray, n_regimes: int
) -> dict[int, int]:
    """Match new-fit raw states to previous-fit raw states by overlap.

    hmmlearn component indices are arbitrary and may permute between refits.
    Recover identity by the most co-occurring old raw state over the shared
    dates (greedy, largest agreement first; deterministic tie-break).
    """
    cont = np.zeros((n_regimes, n_regimes), dtype=int)
    for o, nw in zip(old_states, new_states):
        if 0 <= int(o) < n_regimes and 0 <= int(nw) < n_regimes:
            cont[int(o), int(nw)] += 1
    mapping: dict[int, int] = {}
    used_old: set[int] = set()
    order = sorted(
        ((nw, o) for o in range(n_regimes) for nw in range(n_regimes)),
        key=lambda t: (-int(cont[t[1], t[0]]), t[1], t[0]),
    )
    for nw, o in order:
        if nw in mapping or o in used_old or cont[o, nw] == 0:
            continue
        mapping[nw] = o
        used_old.add(o)
    for nw in range(n_regimes):
        if nw not in mapping:
            for o in range(n_regimes):
                if o not in used_old:
                    mapping[nw] = o
                    used_old.add(o)
                    break
    return mapping


def _fwd_at_cursor(close: pd.Series, cursor: pd.Timestamp, h: int) -> pd.Series:
    """Forward return series observable at ``cursor`` (NaN past the cursor-h)."""
    trunc = close.loc[:cursor]
    return trunc.shift(-h) / trunc - 1.0


def infer_regime_series(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    h_anchor: int = 10,
    n_regimes: int = 2,
    min_train_size: int = MIN_TRAIN_SIZE,
    retrain_interval: int = RETRAIN_INTERVAL,
    random_state: int = RANDOM_STATE,
    fit_mode: Literal["expanding", "full"] = "expanding",
) -> tuple[pd.Series, dict]:
    """Infer a 0/1 regime series over ``features`` via a GaussianHMM.

    Cursor-safe: a fit at cursor t uses only rows with date <= t, and the
    forward-return relabel uses only returns observable inside that training
    window (``_fwd_at_cursor`` returns NaN past cursor−h). State identity is
    tracked across refits by overlap matching so the forward-return anchor
    (decided at the first fit) is not silently inverted by arbitrary component
    permutations.

    Returns ``(regime_series, meta)`` where ``regime_series`` is aligned to
    ``features.index`` (NaN warmup) and ``meta`` holds diagnostics.
    """
    idx = features.index
    n = len(features)
    regimes = pd.Series(np.nan, index=idx, dtype=float)
    meta: dict = {"fit_mode": fit_mode, "n_refits": 0, "relabel_conflicts": 0}

    if fit_mode == "full":
        X = features.dropna()
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            random_state=random_state,
            n_iter=200,
            tol=1e-3,
        )
        model.fit(X.to_numpy())
        states = model.predict(X.to_numpy())
        fwd = _fwd_at_cursor(close, X.index[-1], h_anchor).reindex(X.index)
        remap = _remap_by_fwd_return(states, fwd.to_numpy(dtype=float), n_regimes)
        for pos, raw in zip(X.index, states):
            regimes.loc[pos] = float(remap[int(raw)])
        meta["n_refits"] = 1
        return regimes, meta

    model: hmm.GaussianHMM | None = None
    raw_to_regime: dict[int, int] = {}
    raw_hist: pd.Series = pd.Series(np.nan, index=idx, dtype=float)
    last_fit = -1

    for i in range(n):
        if i < min_train_size:
            continue
        cursor = idx[i]
        X = features.iloc[: i + 1].dropna()
        if len(X) < min_train_size:
            continue
        if model is None or i - last_fit >= retrain_interval:
            new_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                random_state=random_state,
                n_iter=200,
                tol=1e-3,
            )
            new_model.fit(X.to_numpy())
            new_states = new_model.predict(X.to_numpy())
            fwd_train = _fwd_at_cursor(close, cursor, h_anchor).reindex(X.index)
            if model is None:
                raw_to_regime = _remap_by_fwd_return(
                    new_states, fwd_train.to_numpy(dtype=float), n_regimes
                )
            else:
                shared = raw_hist.dropna().index.intersection(X.index)
                if len(shared):
                    old_raw = raw_hist.reindex(shared).to_numpy(dtype=int)
                    new_raw = (
                        pd.Series(new_states, index=X.index)
                        .reindex(shared)
                        .to_numpy(dtype=int)
                    )
                    match = _match_new_to_old(new_raw, old_raw, n_regimes)
                    raw_to_regime = {
                        nw: raw_to_regime.get(ow, 0) for nw, ow in match.items()
                    }
                    # stability diagnostic: does fwd-return ordering agree with
                    # the tracked labels (regime0 = higher mean fwd return)?
                    fw = fwd_train.to_numpy(dtype=float)
                    fwd_means = {
                        s: float(fw[(new_states == s) & np.isfinite(fw)].mean())
                        if (new_states == s).any()
                        else -np.inf
                        for s in range(n_regimes)
                    }
                    r0_state = next((s for s, r in raw_to_regime.items() if r == 0), 0)
                    r1_state = next((s for s, r in raw_to_regime.items() if r == 1), 1)
                    if fwd_means[r1_state] > fwd_means[r0_state]:
                        meta["relabel_conflicts"] += 1
            model = new_model
            last_fit = i
            meta["n_refits"] += 1
        assert model is not None
        st = int(model.predict(X.to_numpy())[-1])
        raw_hist.iloc[i] = st
        if st in raw_to_regime:
            regimes.iloc[i] = float(raw_to_regime[st])
    return regimes, meta


# --------------------------------------------------------------------------- #
# flips + lead-time study
# --------------------------------------------------------------------------- #


def persistent_flips(
    regimes: pd.Series, min_run: int = 5
) -> list[tuple[pd.Timestamp, int, int]]:
    """Detect persistent regime flips (new state held >= ``min_run`` bars).

    Screens out short flickers exactly as documented: a transition between two
    *persistent* runs is a flip; a short run bridging two runs of the same
    state is ignored. Returns ``(flip_date, from_state, to_state)``.
    """
    s = regimes.dropna().astype(int)
    flips: list[tuple[pd.Timestamp, int, int]] = []
    prev_state: int | None = None
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and int(s.iloc[j + 1]) == int(s.iloc[i]):
            j += 1
        run_len = j - i + 1
        if run_len >= min_run:
            state = int(s.iloc[i])
            if prev_state is not None and state != prev_state:
                flips.append((s.index[i], prev_state, state))
            prev_state = state
        i = j + 1
    return flips


def pivot_mask(close: pd.Series, r: int, kind: Literal["high", "low"]) -> pd.Series:
    """Boolean mask of local pivots (±r bars window) of the given kind."""
    win = 2 * r + 1
    if kind == "high":
        roll = close.rolling(win, center=True, min_periods=win).max()
    else:
        roll = close.rolling(win, center=True, min_periods=win).min()
    return (close == roll) & roll.notna()


def significant_pivots(
    close: pd.Series, r: int = 5, min_swing: float = 0.03
) -> tuple[pd.Index, pd.Index]:
    """Alternating pivot highs/lows that clear a minimum swing vs the previous pivot.

    A pivot only counts as a "turning point" (drawdown start / rally start) if the
    move from the previous opposite pivot is at least ``min_swing`` (fraction), so
    minor ±r-bar wiggles are not mistaken for real turns. Returns ``(highs, lows)``.
    """
    highs_mask = pivot_mask(close, r, "high")
    lows_mask = pivot_mask(close, r, "low")
    highs = list(close.index[highs_mask])
    lows = list(close.index[lows_mask])
    # Interleave chronologically, enforcing alternation + min swing.
    seq: list[tuple[pd.Timestamp, str]] = []
    for d in close.index:
        is_h = d in highs
        is_l = d in lows
        if is_h and is_l:
            continue  # degenerate flat bar; skip
        if not (is_h or is_l):
            continue
        kind = "high" if is_h else "low"
        if not seq:
            seq.append((d, kind))
            continue
        last_d, last_kind = seq[-1]
        if kind == last_kind:
            # same direction: keep the more extreme of the two
            if (kind == "high" and close[d] > close[last_d]) or (
                kind == "low" and close[d] < close[last_d]
            ):
                seq[-1] = (d, kind)
            continue
        swing = abs(close[d] / close[last_d] - 1.0)
        if swing >= min_swing:
            seq.append((d, kind))
    highs_out = pd.Index([d for d, k in seq if k == "high"])
    lows_out = pd.Index([d for d, k in seq if k == "low"])
    return highs_out, lows_out


def lead_to_turning_point(
    close: pd.Series,
    flip_date: pd.Timestamp,
    to_state: int,
    r: int = 5,
    max_horizon: int = 252,
    min_swing: float = 0.03,
) -> int | None:
    """Trading-day lead from a flip to the next index turning point.

    to_state==1 (risk-off started) → next local MAX (drawdown start).
    to_state==0 (risk-on started)  → next local MIN (rally start).
    Returns None when no pivot exists within ``max_horizon`` trading days.
    """
    highs, lows = significant_pivots(close, r=r, min_swing=min_swing)
    pivots = highs if to_state == 1 else lows
    pos = close.index.get_loc(flip_date)
    for p in pivots:
        pp = close.index.get_loc(p)
        if pp > pos:
            if pp - pos > max_horizon:
                return None
            return pp - pos
    return None


def lead_study(
    close: pd.Series,
    flips: list[tuple[pd.Timestamp, int, int]],
    r: int = 5,
    min_swing: float = 0.03,
) -> pd.DataFrame:
    """Lead-time study per flip: lead days + forward return at h=10 and h=21."""
    rows: list[dict] = []
    for date, frm, to in flips:
        lead = lead_to_turning_point(close, date, to, r=r, min_swing=min_swing)
        pos = close.index.get_loc(date)
        r10 = close.iloc[min(pos + 10, len(close) - 1)] / close.iloc[pos] - 1.0
        r21 = close.iloc[min(pos + 21, len(close) - 1)] / close.iloc[pos] - 1.0
        rows.append(
            {
                "flip_date": date,
                "from": frm,
                "to": to,
                "lead_days": lead,
                "r10": r10,
                "r21": r21,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# discrimination + series-level lead-lag
# --------------------------------------------------------------------------- #


def discrimination_gap(
    regimes: pd.Series, close: pd.Series, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Mean forward return per regime and the gap (regime0 − regime1), per h."""
    r = regimes.dropna().astype(int).rename("regime")
    rows: list[dict] = []
    for h in horizons:
        fwd = (close.shift(-h) / close - 1.0).reindex(r.index).rename("fwd")
        d = pd.concat([r, fwd], axis=1).dropna()
        if len(d) < 30 or d["regime"].nunique() < 2:
            rows.append(
                {
                    "h": h,
                    "regime0": np.nan,
                    "regime1": np.nan,
                    "gap": np.nan,
                    "n": len(d),
                }
            )
            continue
        g0 = float(d.loc[d["regime"] == 0, "fwd"].mean())
        g1 = float(d.loc[d["regime"] == 1, "fwd"].mean())
        rows.append({"h": h, "regime0": g0, "regime1": g1, "gap": g0 - g1, "n": len(d)})
    return pd.DataFrame(rows)


def _block_bootstrap_premium_p(
    regime: np.ndarray,
    fwd: np.ndarray,
    block: int = 63,
    iters: int = 4000,
    seed: int = 7,
) -> float:
    """Block-bootstrap p-value for the regime0-vs-regime1 forward premium."""
    a = regime.astype(float)
    b = fwd.astype(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2 * block or a.sum() < 5 or (n - a.sum()) < 5:
        return 1.0

    def prem(x: np.ndarray, y: np.ndarray) -> float:
        return float(y[x == 1].mean() - y[x == 0].mean())

    obs = prem(a, b)
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / block))
    exceed = 0
    for _ in range(iters):
        starts = rng.integers(0, n - block, size=nblk)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        if abs(prem(a[idx], b[idx])) >= abs(obs):
            exceed += 1
    return exceed / iters


def series_level(
    regimes: pd.Series, close: pd.Series, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Series-level lead-lag: corr(regime0, fwd), corr(past), premium + bootstrap.

    ``prem_bp`` is the mean log-forward-return difference (regime0 − regime1)
    in basis points; ``p_prem`` is its block-bootstrap p-value. ``p_corr`` is
    the block-bootstrap p-value of the correlation (mirrors lead_scan).
    """
    r0 = (regimes.dropna() == 0).astype(float).rename("r0")
    logc = pd.Series(np.log(close), index=close.index, name="logc")
    rows: list[dict] = []
    for h in horizons:
        fwd = (logc.shift(-h) - logc).reindex(r0.index).rename("fwd")
        past = (logc - logc.shift(h)).reindex(r0.index).rename("past")
        d = pd.concat([r0, fwd, past], axis=1).dropna()
        if len(d) < 60 or d["r0"].nunique() < 2:
            rows.append(
                {
                    "h": h,
                    "corr_fwd": np.nan,
                    "corr_past": np.nan,
                    "prem_bp": np.nan,
                    "p_prem": 1.0,
                    "p_corr": 1.0,
                    "n": len(d),
                }
            )
            continue
        cf = float(np.corrcoef(d["r0"], d["fwd"])[0, 1])
        cp = float(np.corrcoef(d["r0"], d["past"])[0, 1])
        prem_bp = 1e4 * (
            float(d.loc[d["r0"] == 1, "fwd"].mean())
            - float(d.loc[d["r0"] == 0, "fwd"].mean())
        )
        p_prem = _block_bootstrap_premium_p(d["r0"].to_numpy(), d["fwd"].to_numpy())
        p_corr = block_bootstrap_p(d["r0"].to_numpy(), d["fwd"].to_numpy())
        rows.append(
            {
                "h": h,
                "corr_fwd": cf,
                "corr_past": cp,
                "prem_bp": prem_bp,
                "p_prem": p_prem,
                "p_corr": p_corr,
                "n": len(d),
            }
        )
    return pd.DataFrame(rows)


def _band_corr(
    regimes: pd.Series, close: pd.Series, lo: int = 5, hi: int = 15
) -> float:
    """corr(regime0_t, r_{t+lo..t+hi}) — the README Test B band statistic."""
    r0 = (regimes.dropna() == 0).astype(float)
    logc = pd.Series(np.log(close), index=close.index, name="logc")
    band = (logc.shift(-hi) - logc.shift(-lo)).reindex(r0.index)
    d = pd.concat([r0, band], axis=1).dropna()
    if len(d) < 30:
        return float("nan")
    return float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "  n/a"
    return f"{100 * x:+6.2f}%"


def _sig(p: float) -> str:
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return " "


def _report_feature_set(
    name: str,
    close: pd.Series,
    regimes: pd.Series,
    meta: dict,
    fit_mode: str,
    min_run: int,
    pivot_r: int,
    min_swing: float,
) -> None:
    print("=" * 92)
    print(
        f"FEATURE SET: {name}   (fit={fit_mode}, min_run={min_run}, pivot_r={pivot_r}, min_swing={min_swing:.0%})"
    )
    print(f"  refits={meta['n_refits']}  relabel_conflicts={meta['relabel_conflicts']}")
    rs = regimes.dropna()
    valid = rs.index
    if len(rs):
        print(
            f"  regime series: n={len(rs)}  {valid[0].date()} → {valid[-1].date()}  "
            f"regime0={100 * (rs == 0).mean():.1f}%  regime1={100 * (rs == 1).mean():.1f}%"
        )
    flips = persistent_flips(regimes, min_run=min_run)
    yrs = max((valid[-1] - valid[0]).days / 365.25, 1e-9)
    print(
        f"  persistent flips (min_run={min_run}): {len(flips)}  "
        f"({len(flips) / yrs:.2f}/yr)  "
        f"0→1: {sum(1 for f in flips if f[2] == 1)}  1→0: {sum(1 for f in flips if f[2] == 0)}"
    )

    ls = lead_study(close, flips, r=pivot_r, min_swing=min_swing)
    leads = ls["lead_days"].dropna()
    if len(leads):
        q1, med, q3 = leads.quantile([0.25, 0.5, 0.75])
        print(
            f"  LEAD-TIME: n={len(leads)}/{len(flips)}  median={med:.1f}d  Q1={q1:.1f}  Q3={q3:.1f}  "
            f"lead<=20d: {100 * (leads <= 20).mean():.0f}%  "
            f"coincident(<3d): {100 * (leads < 3).mean():.0f}%"
        )
        print(
            f"    mean fwd return after flip: h=10 {_pct(float(ls['r10'].mean()))}  "
            f"h=21 {_pct(float(ls['r21'].mean()))}"
        )
    else:
        print("  LEAD-TIME: no flips with a forward turning point")

    mid = valid[int(len(valid) * 0.5)]
    early_r = regimes.loc[regimes.index <= mid]
    late_r = regimes.loc[regimes.index > mid]
    for label, sub in (("EARLY", early_r), ("LATE", late_r)):
        fl = persistent_flips(sub, min_run=min_run)
        l2 = lead_study(close, fl, r=pivot_r, min_swing=min_swing)["lead_days"].dropna()
        g = discrimination_gap(sub, close, (21,))
        gap21 = float(g.loc[g["h"] == 21, "gap"].iloc[0])
        med_txt = f"{l2.median():.1f}d" if len(l2) else "n/a"
        print(
            f"    SPLIT {label}: {len(fl)} flips  median_lead={med_txt}  gap21={_pct(gap21)}"
        )

    print("  DISCRIMINATION (mean fwd return, regime0 − regime1):")
    for _, row in discrimination_gap(regimes, close).iterrows():
        print(
            f"    h={int(row['h']):>2d}  r0={_pct(float(row['regime0']))}  "
            f"r1={_pct(float(row['regime1']))}  gap={_pct(float(row['gap']))}  n={int(row['n'])}"
        )

    print("  SERIES-LEVEL lead-lag (corr(regime0_t, fwd_t→t+h), block bootstrap):")
    for _, row in series_level(regimes, close).iterrows():
        h = int(row["h"])
        print(
            f"    h={h:>2d}  corr_fwd={row['corr_fwd']:+.3f}  corr_past={row['corr_past']:+.3f}  "
            f"prem={row['prem_bp']:+.2f}bp{_sig(float(row['p_prem']))}  "
            f"p_prem={row['p_prem']:.2f}  n={int(row['n'])}"
        )

    band = _band_corr(regimes, close)
    print(f"    corr(regime0, r_+5..+15) = {band:+.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-mode", choices=["expanding", "full"], default="expanding")
    ap.add_argument(
        "--min-run", type=int, default=5, help="persistence screen for flips (bars)"
    )
    ap.add_argument(
        "--pivot-r", type=int, default=5, help="±r bar window for turning-point pivots"
    )
    ap.add_argument(
        "--min-swing",
        type=float,
        default=0.03,
        help="min swing (fraction) for a significant pivot",
    )
    ap.add_argument(
        "--h-anchor", type=int, default=10, help="forward horizon for state relabel"
    )
    ap.add_argument("--features", choices=["all", "cpi", "cpi+rdisc"], default="all")
    args = ap.parse_args()

    close = load_spy_daily()
    cpi = load_cpi_price_index()
    yld = load_yields()
    print(
        f"SPY daily: {len(close)} bars  {close.index[0].date()} → {close.index[-1].date()}"
    )

    combos: list[tuple[str, bool]] = [
        ("{acc, real}", False),
        ("{acc, real, rdisc}", True),
    ]
    if args.features == "cpi":
        combos = [("{acc, real}", False)]
    elif args.features == "cpi+rdisc":
        combos = [("{acc, real, rdisc}", True)]

    for name, with_rdisc in combos:
        features = build_aligned_features(close, cpi, yld, with_rdisc=with_rdisc)
        regimes, meta = infer_regime_series(
            features, close, h_anchor=args.h_anchor, fit_mode=args.fit_mode
        )
        _report_feature_set(
            name,
            close,
            regimes,
            meta,
            args.fit_mode,
            args.min_run,
            args.pivot_r,
            args.min_swing,
        )


if __name__ == "__main__":
    main()
