"""Pure weight-construction functions for the correlation-driven pf_* strategies.

Ported from the removed ``src.bt.pf.core`` module (the `bt pf` CLI is gone; only
the strategies remain) so the strategies are fully self-contained. Every function
here is a pure function over arrays/DataFrames — no side effects, no I/O.

All methods take a trailing-return window and return a ``pd.Series`` of portfolio
weights over the same columns (summing to ~1, long-only by default). They must be
well-defined on all-NaN / empty windows (return all-zero weights summing to 0).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class WeightMethodFn(Protocol):
    """Weight-construction protocol: trailing returns -> portfolio weights.

    Given a trailing return window, return a Series of target portfolio weights
    over the same columns. All methods must be well-defined on all-NaN / empty
    windows (return all-zero weights summing to 0 to signal unallocated).
    """

    def __call__(self, returns: pd.DataFrame, long_only: bool = True) -> pd.Series: ...


def _usable_columns(returns: pd.DataFrame) -> list[str]:
    """Columns with >= 2 usable (non-NaN, finite) observations for covariance."""
    cols = returns.columns
    clean = returns[cols].replace([np.inf, -np.inf], np.nan)
    return [c for c in cols if clean[c].notna().sum() >= 2]


def _position_weights(
    weights: np.ndarray,
    usable: list[str],
    all_cols: pd.Index,
    long_only: bool,
) -> pd.Series:
    """Position weight array into a full-column Series, optionally clipping short."""
    full = np.zeros(len(all_cols))
    idx = {c: i for i, c in enumerate(all_cols)}
    for c, wt in zip(usable, weights):
        full[idx[c]] = wt

    if long_only and full.min() < 0:
        full = np.clip(full, 0.0, None)
        total = float(np.sum(full))
        if total > 1e-12:
            full = full / total

    return pd.Series(full, index=all_cols)


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    """Enforce a per-asset maximum weight, redistributing any excess.

    Iteratively clips every weight above ``max_weight`` and spreads the freed
    allocation proportionally across the remaining (uncapped) assets, so the
    result still sums to 1 and no single asset exceeds the cap. A ``max_weight``
    <= 0 or >= 1 leaves the weights unchanged.
    """
    if max_weight <= 0.0 or max_weight >= 1.0:
        return weights

    w = np.asarray(weights.values, dtype=float)
    n = len(w)
    for _ in range(n + 1):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w - max_weight)[over].sum())
        w = np.where(over, max_weight, w)
        room = w < max_weight
        pool = float(w[room].sum()) if room.any() else 0.0
        if pool <= 1e-12:
            # Degenerate (all assets at/over the cap): clip and split evenly.
            if room.any():
                w[room] = max_weight
            break
        w[room] += excess * (w[room] / pool)

    return pd.Series(w, index=weights.index)


def min_variance_weights(
    returns: pd.DataFrame,
    long_only: bool = True,
) -> pd.Series:
    """Global Minimum-Variance portfolio weights from a trailing-return window.

    ``w = (Σ⁻¹ · 1) / (1ᵀ · Σ⁻¹ · 1)`` where Σ is the sample covariance of the
    trailing returns (the correlation structure scaled by per-asset vol). This
    minimises portfolio variance — the standard "correlation determines weight"
    allocation. Correlated assets are jointly downweighted; diversifiers are
    upweighted.
    """
    if returns.empty:
        return pd.Series(0.0, index=returns.columns)

    usable = _usable_columns(returns)
    if not usable:
        return pd.Series(0.0, index=returns.columns)

    clean = returns[usable].replace([np.inf, -np.inf], np.nan)
    cov = clean.cov()
    try:
        inv = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError:
        # Singular covariance (e.g. duplicates): fall back to equal weight.
        w = np.ones(len(usable)) / len(usable)
        return _position_weights(w, usable, returns.columns, long_only)

    ones = np.ones(len(usable))
    w = inv @ ones
    denom = float(w.sum())
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        w = np.ones(len(usable)) / len(usable)
    else:
        w = w / denom

    return _position_weights(w, usable, returns.columns, long_only)


def inverse_vol_weights(
    returns: pd.DataFrame,
    long_only: bool = True,
) -> pd.Series:
    """Inverse-volatility weights: ``wᵢ ∝ 1/σᵢ`` then normalised.

    The classic low-vol allocation. Weights by per-asset risk only (ignores
    correlation), so weights only move when trailing volatilities move — typically
    much lower turnover than covariance-based schemes.
    """
    if returns.empty:
        return pd.Series(0.0, index=returns.columns)

    usable = _usable_columns(returns)
    if not usable:
        return pd.Series(0.0, index=returns.columns)

    clean = returns[usable].replace([np.inf, -np.inf], np.nan)
    vol = clean.std()
    inv = 1.0 / vol
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 1e-12:
        w = np.ones(len(usable)) / len(usable)
    else:
        w = inv.values / total

    return _position_weights(w, usable, returns.columns, long_only)


def risk_parity_weights(
    returns: pd.DataFrame,
    long_only: bool = True,
    max_iters: int = 200,
    tol: float = 1e-9,
) -> pd.Series:
    """Equal-Risk-Contribution (ERC / risk parity) weights from trailing returns.

    Iteratively finds weights such that every asset contributes the same share
    of total portfolio variance: ``w · (Σw)ₜ ∝ 1/N``. Correlations matter here
    (a correlated/high-vol asset is endogenously downweighted) but the result is
    far less extreme than GMV because diversification is rewarded across all risk
    dimensions. Update rule ``wᵢ ← β / (N · (Σw)ᵢ)`` with ``β = N/2``.
    """
    if returns.empty:
        return pd.Series(0.0, index=returns.columns)

    usable = _usable_columns(returns)
    if not usable:
        return pd.Series(0.0, index=returns.columns)

    clean = returns[usable].replace([np.inf, -np.inf], np.nan)
    cov = clean.cov().values
    cov = np.asarray(cov, dtype=float)
    n = len(usable)
    cov = np.maximum(cov, cov.T)  # enforce symmetry for numerical safety

    # Damped update for the symmetric ERC solution.
    w = np.full(n, 1.0 / n)
    beta = n / 2.0
    for _ in range(max_iters):
        sw = cov @ w
        nnz = np.abs(sw) > 1e-12
        if not nnz.any():
            break
        w_new = np.zeros(n)
        w_new[nnz] = beta / (n * sw[nnz])
        w_new = 0.5 * (w + w_new)
        denom = float(np.sum(w_new))
        if not np.isfinite(denom) or denom <= 1e-12:
            w_new = np.ones(n) / n
        else:
            w_new = w_new / denom
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return _position_weights(
        np.asarray(w, dtype=float), usable, returns.columns, long_only
    )


def fixed_alloc_weights(
    targets: dict[str, float],
) -> WeightMethodFn:
    """Return a weight function holding constant target allocations.

    The returned ``WeightMethodFn`` ignores the trailing-return window (there is
    no estimation) and emits ``targets`` (normalised to sum 1, filtered to the
    columns present). This gives a classic fixed-weight, periodically-rebalanced
    portfolio (PRP): weights drift with price between rebalances, then snap back
    to the targets on cadence.

    Args:
        targets: {symbol: target weight}, e.g. ``{"SPY": 0.5, "TLT": 0.1}``.

    Raises:
        ValueError: If ``targets`` is empty or does not sum to a positive value.
    """
    total = float(sum(targets.values()))
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("allocations must sum to a positive amount")
    normalised = {k: v / total for k, v in targets.items()}

    def _fixed(
        returns: pd.DataFrame,
        long_only: bool = True,  # noqa: ARG001 -- fixed targets are already long-only
    ) -> pd.Series:
        cols = list(returns.columns)
        out = pd.Series(0.0, index=cols)
        for sym, wt in normalised.items():
            if sym in out.index:
                out.loc[sym] = wt
        total_out = float(out.sum())
        if total_out > 1e-12:
            out = out / total_out
        return out

    return _fixed


def resolve_rebalance_period(raw: str) -> str:
    """Normalise a rebalance string into a pandas offset alias.

    Accepts named cadences (daily/weekly/monthly/quarterly/yearly) or a
    multiplicative shorthand ``Nd``/``Nw``/``Nm`` (N calendar days/weeks/months).

    Raises:
        ValueError: For unrecognised inputs.
    """
    key = raw.strip().lower()
    named = {
        "daily": "1D",
        "weekly": "W-MON",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE",
    }
    if key in named:
        return named[key]

    if len(key) >= 2 and key[-1] in {"d", "w", "m"} and key[:-1].isdigit():
        n = int(key[:-1])
        if n <= 0:
            raise ValueError(f"rebalance period must be positive: {raw!r}")
        suffix = key[-1]
        unit = {"d": "D", "w": "W", "m": "ME"}[suffix]
        return f"{n}{unit}"

    raise ValueError(
        f"invalid {raw!r}; expected one of daily/weekly/monthly/quarterly/yearly "
        f"or 'Nd'/'Nw'/'Nm'"
    )
