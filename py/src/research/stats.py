"""Pure cross-sectional research statistics.

All functions are pure over aligned pandas/numpy inputs — no I/O — so they are
unit-testable and reusable when a user graduates a relationship into a
strategy. Each returns a frozen result object from ``src.research.types``.

Conventions
-----------
- ``residual`` returns = daily member return minus beta times benchmark daily
  return, beta from a rolling OLS fit (63 trading days) where each day uses
  only the *preceding* ``_BETA_WINDOW`` bars (shifted one) — no lookahead.
- Sample sizes always accompany a number; nothing is silently hidden.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from src.research.types import (
    CatalystResult,
    CatalystSide,
    DispersionResult,
    MomentumCell,
    RegimeCell,
    VolClusterResult,
)

_BETA_WINDOW = 63
_MOM_LOOKBACKS = (5, 21)
_MOM_HORIZONS = (5, 21)
_VOL_LAGS = (1, 5, 20)
_VOL_WINDOW = 20
_SPIKE_MULT = 1.5
_SLOPE_DAYS = 21
_VOL_HALFLIFE = 20


# ── Residual construction ─────────────────────────────────────


def residual_returns(
    ret: pd.DataFrame, bench_ret: pd.Series, window: int = _BETA_WINDOW
) -> pd.DataFrame:
    """Idiosyncratic daily return of each member versus the benchmark.

    Args:
        ret: member daily returns (DatetimeIndex, one column per member).
        bench_ret: benchmark daily returns sharing the index intersection.
        window: rolling OLS window in trading days (default 63).

    Returns:
        Same-shaped residual return frame (NaN through warmup).
    """
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    x = bench_ret.reindex(ret.index).astype(float)
    for member in ret.columns:
        resid = _residual_series(ret[member].astype(float), x, window)
        out[member] = resid.reindex(ret.index)
    return out


def _residual_series(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Residual series of one name vs the benchmark using prior-window beta."""
    pair = pd.concat([y, x], axis=1).dropna()
    if len(pair) < window + 2:
        return y.mul(float("nan"))
    a, b = pair.iloc[:, 0], pair.iloc[:, 1]
    beta = a.rolling(window).cov(b).div(b.rolling(window).var())
    alpha = a.rolling(window).mean() - beta * b.rolling(window).mean()
    prev = pd.concat([alpha, beta], axis=1).shift(1)
    alpha_prev = prev.iloc[:, 0].dropna()
    beta_prev = prev.iloc[:, 1].dropna()
    idx = alpha_prev.index.intersection(beta_prev.index).intersection(a.index)
    resid = a.loc[idx] - (alpha_prev.loc[idx] + beta_prev.loc[idx] * b.loc[idx])
    return resid


# ── Family 1: residual dispersion ─────────────────────────────


def residual_dispersion(resid: pd.DataFrame) -> DispersionResult:
    """Mean pairwise residual rho, PC1 variance share, effective independent count.

    Reports low effective independent bets for a correlated "one factor" panel.
    """
    if resid.shape[1] < 2 or resid.shape[0] < 10:
        return DispersionResult(
            None, None, None, int(resid.shape[1]), int(resid.shape[0])
        )
    stripped = resid.dropna()
    if stripped.shape[0] < 10:
        return DispersionResult(
            None, None, None, int(resid.shape[1]), int(stripped.shape[0])
        )

    corr = stripped.corr()
    n = int(corr.shape[0])
    upper = corr.values[np.triu_indices(n, k=1)]
    mean_rho = float(np.nanmean(upper)) if len(upper) else None

    eigen = np.sort(np.linalg.eigvalsh(corr.to_numpy(dtype=float)))[::-1]
    trace = float(eigen.sum())
    pc1_share = float(eigen[0]) / trace if trace > 0 else None

    eff_n: float | None = None
    if mean_rho is not None:
        denom = 1.0 + mean_rho * (n - 1)
        if denom > 0:
            eff_n = float(n / denom)

    return DispersionResult(
        mean_pairwise_rho=mean_rho,
        pc1_var_share=pc1_share,
        effective_n=eff_n,
        n_members=n,
        n_rows=int(stripped.shape[0]),
    )


# ── Family 2: cross-sectional momentum / reversal sweep ───────


def momentum_sweep(
    ret: pd.DataFrame,
    resid: pd.DataFrame,
    lookbacks: Sequence[int] = _MOM_LOOKBACKS,
    horizons: Sequence[int] = _MOM_HORIZONS,
) -> tuple[MomentumCell, ...]:
    """Spearman rank forecast + decile spread per (lookback, horizon).

    Reports the *net-of-benchmark* spread (residual returns) plus the raw
    spread so benchmark contamination is visible.
    """
    cells: list[MomentumCell] = []
    for lookback in lookbacks:
        for horizon in horizons:
            prior_net = _cum_signal(resid, lookback)
            fwd_net = _cum_forward(resid, horizon)
            net_rho, net_t, net_spread, net_n = _rank_forecast(prior_net, fwd_net)

            prior_raw = _cum_signal(ret, lookback)
            fwd_raw = _cum_forward(ret, horizon)
            _, _, raw_spread, raw_n = _rank_forecast(prior_raw, fwd_raw)

            # De-overlapped independent-sample significance for the net side.
            # The pooled ``net_t`` treats every overlapped window as an
            # independent row; the effective t on the stride-decimated
            # subsample is the honest gate for a strategy decision.
            s_eff, t_eff, n_eff, too_small = _effective_rank_forecast(
                prior_net, fwd_net, lookback, horizon
            )

            cells.append(
                MomentumCell(
                    lookback=lookback,
                    horizon=horizon,
                    spearman=net_rho,
                    t_stat=net_t,
                    n_rows=net_n,
                    net_decile_bps=net_spread,
                    net_n=net_n,
                    raw_decile_bps=raw_spread,
                    raw_n=raw_n,
                    spearman_effective=s_eff,
                    t_effective=t_eff,
                    n_effective=n_eff,
                    n_effective_too_small=too_small,
                )
            )
    return tuple(cells)


def _cum_signal(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    """Cumulative return over ``days`` bars strictly before the current one."""
    return returns.shift(1).rolling(days).sum()


def _cum_forward(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    """Cumulative return over the next ``days`` bars starting at a given row."""
    return returns.rolling(days).sum().shift(-(days - 1))


def _independent_stride(lookback: int, horizon: int) -> int:
    """Trading-day stride between windows treated as near-independent samples.

    Consecutive trailing/forward windows overlap by ``max(lookback, horizon)
    - 1`` days, so two observations starting on adjacent days share ~all their
    bars. Sampling one start-day per ``max(lookback, horizon)`` trading days
    roughly removes the redundancy (the very share the pooled t-stat wrongly
    counts as additional independent rows).
    """
    return max(int(lookback), int(horizon))


def _effective_rank_forecast(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> tuple[float | None, float | None, int, bool]:
    """Spearman rho/t on a stride-decimated, near-independent subsample.

    Returns (spearman_eff, t_eff, n_eff, n_eff_too_small). A single ``stride``
    phase (anchored at the first day that carries an informative window) makes
    the subsample a clean, auditable partition of the tail — not the "best" of
    several phases, so it adds no multiple-comparison optimism. Point estimates
    (the pooled ``_rank_forecast``) are untouched; this only restates
    significance without overlap inflation.
    """
    stride = _independent_stride(lookback, horizon)
    if stride <= 1 or signal.shape[1] == 0:
        return None, None, 0, True
    alive = signal.notna().any(axis=1) & target.notna().any(axis=1)
    if not bool(alive.any()):
        return None, None, 0, True
    rows = np.flatnonzero(alive.to_numpy())
    sampled = rows[::stride]
    sub_sig = signal.iloc[sampled]
    sub_tgt = target.iloc[sampled]
    rho, t_stat, _spread, n = _rank_forecast(sub_sig, sub_tgt)
    return rho, t_stat, int(n), bool(t_stat is None or n < 30)


# ── Family 3: vol clustering / conditional spike (residual) ───


def vol_clustering(
    resid: pd.DataFrame,
    lags: Sequence[int] = _VOL_LAGS,
    vol_window: int = _VOL_WINDOW,
    spike_mult: float = _SPIKE_MULT,
) -> VolClusterResult:
    """Autocorr of realized residual vol across names + next-day response to a spike.

    Realized vol_t = rolling std of residual returns over ``vol_window`` days.
    A spike at t means realized vol_t > ``spike_mult`` times the prior EWMA-vol
    (halflife 20 days). Forward response is next-day pooled residual return.
    """
    if resid.shape[1] == 0 or resid.shape[0] < vol_window + 5:
        return VolClusterResult(None, None, None, int(resid.shape[1]), None, 0, None)

    realized = resid.rolling(vol_window).std()
    ac: dict[int, float | None] = {}
    for lag in lags:
        vals: list[float] = []
        for member in resid.columns:
            s = realized[member].dropna()
            if len(s) > lag + 5:
                vals.append(_corr_lag(s, lag))
        ac[lag] = float(np.mean(vals)) if vals else None

    alpha = _ewma_alpha(_VOL_HALFLIFE)
    spike_fwd: list[float] = []
    for member in resid.columns:
        col = resid[member]
        rv = col.rolling(vol_window).std()
        ewma = rv.ewm(alpha=alpha, adjust=False).mean().shift(1)
        spike = (rv > spike_mult * ewma) & ewma.notna()
        fwd = col.shift(-1)
        spike_fwd.extend(fwd[spike].dropna().tolist())

    fwd_all = resid.shift(-1).stack().dropna().to_numpy(dtype=float)
    return VolClusterResult(
        mean_ac1=ac.get(1),
        mean_ac5=ac.get(5),
        mean_ac20=ac.get(20),
        n_members=int(resid.shape[1]),
        spike_fwd_mean_resid=float(np.mean(spike_fwd)) if spike_fwd else None,
        spike_n=len(spike_fwd),
        unconditional_resid_mean=float(np.mean(fwd_all)) if len(fwd_all) else None,
    )


def _corr_lag(series: pd.Series, lag: int) -> float:
    """Autocorrelation at ``lag`` of a Series (one-pass, NaN-free input)."""
    d = series.to_numpy(dtype=float)
    demean = d - d.mean()
    denom = demean @ demean
    if denom == 0:
        return float("nan")
    return float((demean[:-lag] @ demean[lag:]) / denom)


def _ewma_alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / halflife)


# ── Family 5: benchmark regime splits ─────────────────────────


def bench_regimes(bench_ret: pd.Series, slope_days: int = _SLOPE_DAYS) -> pd.Series:
    """Bucket each day 'down'/'flat'/'up' by benchmark 21-day log-slope tertile."""
    close = (1.0 + bench_ret.fillna(0.0)).cumprod()
    slope = np.log(close / close.shift(slope_days)).dropna()
    if len(slope) < 30:
        return pd.Series("flat", index=bench_ret.index)
    edges = pd.qcut(slope, 3, labels=False, duplicates="drop")
    bucket = pd.Series("flat", index=slope.index)
    codes = edges.astype(float)
    bucket.loc[codes[codes == 0].index] = "down"
    bucket.loc[codes[codes == 2].index] = "up"
    return bucket.reindex(bench_ret.index).fillna("flat")


def regime_stats(
    bucket: pd.Series,
    resid: pd.DataFrame,
    lookback: int = 21,
    horizon: int = 21,
) -> tuple[RegimeCell, ...]:
    """Per-regime momentum decile spread and pairwise rho for the residual panel."""
    cells: list[RegimeCell] = []
    aligned = bucket.reindex(resid.index)
    for regime in ("down", "flat", "up"):
        subset = resid[aligned == regime]
        cells.append(_regime_cell(regime, subset, lookback, horizon))
    return tuple(cells)


def _regime_cell(
    regime: str, subset: pd.DataFrame, lookback: int, horizon: int
) -> RegimeCell:
    if subset.shape[0] < 15 or subset.shape[1] < 2:
        return RegimeCell(regime, None, None, int(subset.shape[0]), None, None)
    signal = _cum_signal(subset, lookback)
    target = _cum_forward(subset, horizon)
    _, _, spread, n = _rank_forecast(signal, target)
    # Regime rows pool overlapping 21->21 windows too, so de-overlap the
    # confidence on a stride-21 subsample just like the whole-panel momentum
    # cells (see _effective_rank_forecast).
    s_eff, t_eff, n_eff, too_small = _effective_rank_forecast(
        signal, target, lookback, horizon
    )
    stripped = subset.dropna()
    rho: float | None = None
    if stripped.shape[1] > 1 and stripped.shape[0] > 5:
        corr = stripped.corr()
        upper = corr.values[np.triu_indices(corr.shape[0], k=1)]
        vals = upper[~np.isnan(upper)]
        rho = float(np.mean(vals)) if len(vals) else None
    return RegimeCell(
        regime=regime,
        bench_slope_lo=None,
        bench_slope_hi=None,
        n_rows=n,
        net_decile_bps=spread,
        mean_pairwise_rho=rho,
        spearman_effective=s_eff,
        t_effective=t_eff,
        n_effective=n_eff,
        n_effective_too_small=too_small,
    )


# ── Shared rank/decile helper ─────────────────────────────────


def _rank_forecast(
    signal: pd.DataFrame, target: pd.DataFrame
) -> tuple[float | None, float | None, float | None, int]:
    """Pooled Spearman rho, t-stat and top-minus-bottom decile spread (bps)."""
    sig = signal.stack().dropna()
    tgt = target.stack().dropna()
    idx = sig.index.intersection(tgt.index)
    if len(idx) < 30:
        return None, None, None, len(idx)
    x = sig.loc[idx].to_numpy(dtype=float)
    y = tgt.loc[idx].to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return None, None, None, len(x)

    rho, _p = _scipy_stats.spearmanr(x, y)
    rho = float(rho) if not np.isnan(rho) else None
    t_stat = None
    if rho is not None and abs(rho) < 1.0:
        t_stat = float(rho * np.sqrt((len(x) - 2.0) / (1.0 - rho * rho)))

    labels = _quantile_labels(x, 10)
    spread: float | None = None
    if labels is not None and len(set(labels[~np.isnan(labels)])) >= 2:
        means = {}
        for q in np.unique(labels[~np.isnan(labels)]):
            means[q] = float(np.mean(y[labels == q]))
        spread = float((means[max(means)] - means[min(means)]) * 1e4)
    return rho, t_stat, spread, len(x)


def _quantile_labels(x: np.ndarray, bins: int) -> np.ndarray | None:
    """Quantile-bucket integer labels (NaN for ties below the minimum bucket)."""
    s = pd.Series(x)
    try:
        labels = pd.qcut(s, bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    vals = labels.astype("float").to_numpy()
    nan = np.isnan(s.to_numpy())
    return np.where(nan, np.nan, vals)


# ── Family 4: intraday catalyst drift / fade ──────────────────


def catalyst_intraday(
    hourly_logret: Mapping[str, pd.Series],
    threshold_sigma: float = 2.5,
) -> CatalystResult:
    """Drift vs fade after an extreme ≤1h move, split by direction.

    Flags |hourly log return| > ``threshold_sigma`` x prior EWMA-vol (halflife
    24 hours) of that name, prior meaning before the current bar. Records the
    forward 1h and forward-24h log-return totals in bps.
    """
    alpha = _ewma_alpha(24.0)
    up: list[tuple[float | None, float | None]] = []
    down: list[tuple[float | None, float | None]] = []
    handled: int = 0
    for series in hourly_logret.values():
        lr = series.dropna().astype(float)
        if len(lr) < 250:
            continue
        sig2 = (lr**2).ewm(alpha=alpha, adjust=False).mean()
        vol_prior = sig2.shift(1).pow(0.5)
        mask = lr.abs() > threshold_sigma * vol_prior
        positions = np.flatnonzero(mask.to_numpy())
        values = lr.to_numpy(dtype=float)
        for p in positions:
            handled += 1
            fwd_1h: float | None = (
                float(values[p + 1]) * 1e4 if p + 1 < len(values) else None
            )
            fwd_24h: float | None = None
            if p + 25 <= len(values):
                fwd_24h = float(values[p + 1 : p + 25].sum()) * 1e4
            bucket = up if values[p] > 0 else down
            bucket.append((fwd_1h, fwd_24h))
    return CatalystResult(
        sides=(_catalyst_side("up", up), _catalyst_side("down", down)),
        total_events=handled,
        threshold_sigma=threshold_sigma,
        n_names=len(hourly_logret),
    )


def _catalyst_side(
    label: str, events: Sequence[tuple[float | None, float | None]]
) -> CatalystSide:
    f1 = [e[0] for e in events if e[0] is not None]
    f24 = [e[1] for e in events if e[1] is not None]
    return CatalystSide(
        direction=label,
        events=len(events),
        fwd_1h_mean_bps=float(np.mean(f1)) if f1 else None,
        fwd_24h_mean_bps=float(np.mean(f24)) if f24 else None,
    )


__all__ = [
    "residual_returns",
    "residual_dispersion",
    "momentum_sweep",
    "vol_clustering",
    "bench_regimes",
    "regime_stats",
    "catalyst_intraday",
]
