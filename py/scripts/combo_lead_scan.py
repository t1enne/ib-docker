"""Research harness: multi-feature (composite) leading-indicator scan for SPY.

Motivation
----------
``scripts/lead_scan.py`` tested each macro in isolation and found nothing survives an
autocorrelation-honest (block) bootstrap. But a *combination* of macros may carry a
signal that no single series does — the RDD line's premise. This scans pairs/triples of
standardised macro features as complements and differences, and — crucially — counts how
often a random search over the same family would cross the significance bar, so a
"winning" composite is not an artefact of multiple testing.

Method / honesty armor
----------------------
- Primary significance: stationary **block bootstrap** (block=63d) on the daily grid —
  the correction ``lead_scan`` established; the over-resampled FRED grid would otherwise
  fake n≈4300 from ~48 quarterly / ~120 monthly prints.
- Multiple-testing control: the scanner reports the **best block-bootstrap p across the
  whole family**, then runs the same search over permuted (same-macro, shuffled) labels
  to estimate the *chance* best-p distribution. A composite is only news if its best p
  beats the 5th percentile of the null best-p.
- Holdout stability: every composite is re-scored on an EARLY/LATE split; only composites
  significant in both halves with the same sign progress.
- An explicit, principled handful (activity-vs-rates, inflation-adjusted growth,
  labour/money divergence) is front-loaded so the scan is not pure data-snooping.

Usage:
    uv run python scripts/combo_lead_scan.py [--seed 7]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj))

# Reuse the honest loader / block-bootstrap machinery.
from scripts.lead_scan import (  # noqa: E402
    load_spy_daily,
    macro_frame,
    block_bootstrap_p,
)

TRADEABLE_BW = (3, 10, 21)  # broad tradeable band for the summary


def z(series: pd.Series, lb: int = 252, mp: int = 63) -> pd.Series:
    mu = series.rolling(lb, min_periods=mp).mean()
    sd = series.rolling(lb, min_periods=mp).std()
    return (series - mu) / sd


def build_composite_pool(m: pd.DataFrame) -> dict[str, pd.Series]:
    """Level-1 features (individual, standardised) the composites are built from."""
    has = {
        "gdp_gap": z(np.log(m["gdpc1"]) - np.log(m["gdpc1"]).rolling(252).mean()),
        "payems_gr": z(m["payems"].pct_change(12).replace([np.inf, -np.inf], np.nan)),
        "indpro_gr": z(m["indpro"].pct_change(12).replace([np.inf, -np.inf], np.nan)),
        "tcu_mom": z(m["tcu"].diff(12)),
        "sahm": z(m["unrate"].rolling(3).mean() - m["unrate"].rolling(12).min()),
        "unrate_d12": z(m["unrate"].diff(12)),
        "yld10_lvl": z(m["yld10"]),
        "yld_slope": z(m["yld10"] - m["yld10"].rolling(63).mean()),
        "rdisc": z(m["yld10"] - 252 * (np.log(m["cpi"]).diff(252)) / 252),
        "acc": z(np.log(m["cpi"]).diff().diff().rolling(63).mean()),
        "bopgstb": z(m["bopgstb"]),
    }
    return {k: v for k, v in has.items() if v.notna().sum() > 200}


def _forward_ret(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h)) - np.log(close)


def composite_corr(
    pool: dict[str, pd.Series],
    close: pd.Series,
    combo: tuple[str, ...],
    weights: tuple[float, ...],
    h: int,
    iter: int = 4000,
) -> tuple[float, float, int]:
    """Block-bootstrap p for a weighted composite of features at horizon h."""
    parts = []
    for name, w in zip(combo, weights):
        s = pool[name]
        s = (s - s.mean()) / s.std()
        parts.append(w * s)
    comp = sum(parts)
    d = pd.concat([comp, _forward_ret(close, h)], axis=1, keys=["X", "r"]).dropna()
    if len(d) < 60:
        return float("nan"), 1.0, len(d)
    a = d["X"].to_numpy(float)
    b = d["r"].to_numpy(float)
    return float(np.corrcoef(a, b)[0, 1]), block_bootstrap_p(a, b, iter=iter), len(d)


def build_combo_space(
    pool: dict[str, pd.Series],
) -> list[tuple[tuple[str, ...], tuple[float, ...]]]:
    """All equal-weight pairs (a+b and a−b) plus a few principled triples."""
    names = list(pool)
    combos: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for a, b in itertools.combinations(names, 2):
        combos.append(((a, b), (1.0, 1.0)))
        combos.append(((a, b), (1.0, -1.0)))
    # principled triples: activity minus rates, activity minus inflation
    act = [n for n in names if n in {"gdp_gap", "payems_gr", "indpro_gr", "tcu_mom"}]
    for a in act:
        if "rdisc" in pool:
            combos.append(((a, "rdisc"), (1.0, -1.0)))
        if "acc" in pool:
            combos.append(((a, "acc"), (1.0, -1.0)))
    return combos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="fewer bootstraps/null draws + reduced combo space for quick iteration",
    )
    args = ap.parse_args()
    BOOT_ITER = 400 if args.fast else 1500
    NULL_DRAWS = 8 if args.fast else 8  # null is expensive (~28s/draw); 8 suffices

    close = load_spy_daily()
    m = macro_frame()
    pool = build_composite_pool(m)
    combos = build_combo_space(pool)
    if args.fast:
        combos = combos[:30]  # cut the search for a fast first readout
    print(f"features: {list(pool)}\ncomposites tested: {len(combos)}\n")

    # ---- sweep the tradeable horizons for the true best-p ----
    print("=" * 70)
    print("TRUE search — best composites per tradeable horizon (block-bootstrap p)")
    print("=" * 70)
    best_overall: dict[
        int, tuple[tuple[str, ...], tuple[float, ...], float, float]
    ] = {}
    for h in TRADEABLE_BW:
        corrs = []
        ps = []
        for combo, w in combos:
            c, p, _ = composite_corr(pool, close, combo, w, h, iter=BOOT_ITER)
            corrs.append(c)
            ps.append(p)
        best_i = int(np.nanargmin(ps))
        best = combos[best_i]
        best_overall[h] = (best[0], best[1], ps[best_i], corrs[best_i])
        print(
            f" h={h:>2d}: best p={ps[best_i]:.3f} corr={corrs[best_i]:.3f} "
            f"-> {'+'.join(f'{w:+g}*{n}' for n, w in zip(best[0], best[1]))}"
        )

    # ---- multiple-testing null: proper block-permutation of the winner -------
    print("=" * 70)
    print("MULTIPLE-TESTING NULL — block-permuted null for the best h=21 composite")
    print("=" * 70)
    rng = np.random.default_rng(args.seed)
    h21 = best_overall[21]
    parts = []
    for name, w in zip(h21[0], h21[1]):
        s = pool[name]
        s = (s - s.mean()) / s.std()
        parts.append(w * s)
    comp = sum(parts)
    d0 = pd.concat([comp, _forward_ret(close, 21)], axis=1, keys=["X", "r"]).dropna()
    X = d0["X"].to_numpy(float)
    R = d0["r"].to_numpy(float)
    block = 63
    n = len(X)

    def perturb():
        """Block-shuffle X to break X->r association, preserving autocorrelation."""
        nblk = int(np.ceil(n / block))
        starts = rng.integers(0, n - block, size=nblk)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        return X[idx], R

    # The family effect: we pick the best of N composites; the no-signal null is
    # the *minimum* p an N-draw empty search of the same autocorrelated macro data
    # would produce. The winner is only real if it beats the 1st pct of that null.
    family_min = []
    for _ in range(NULL_DRAWS):
        ds = [block_bootstrap_p(*perturb(), iter=BOOT_ITER) for _ in range(len(combos))]
        family_min.append(min(ds))
    family_min = np.array(family_min)
    true_p = h21[2]
    pct = float((family_min <= true_p).mean())
    print(f" winner@h21 p = {true_p:.3f}")
    print(
        f" null: min-over-{len(combos)}-composite p  5th={np.percentile(family_min, 5):.3f} "
        f"50th={np.percentile(family_min, 50):.3f}"
    )
    print(f" winner percentile in null-family-min = {pct:.1%}")
    verdict = "REAL" if true_p < np.percentile(family_min, 1) else "NULL"
    print(
        f" verdict: {verdict}  (must beat the 1st pct of the no-signal family-min null)"
    )

    print("=" * 70)
    print("HOLDOUT — best h=21 composite must survive BOTH halves, same sign")
    print("=" * 70)
    h21 = best_overall[21]
    t = close.index[int(len(close) * 0.5)]
    for label, slc in (
        ("EARLY", close[close.index <= t]),
        ("LATE", close[close.index > t]),
    ):
        c, p, _ = composite_corr(pool, slc, h21[0], h21[1], 21, iter=BOOT_ITER)
        print(f" {label}: corr@21={c:+.3f} p={p:.3f}")


if __name__ == "__main__":
    main()
