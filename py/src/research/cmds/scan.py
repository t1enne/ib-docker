"""`ibkr research` command — cross-sectional statistics engine.

Thin CLI orchestration: loads a universe + benchmark from the local candle DB,
builds the daily panel, runs the pure statistic families (``src.research.stats``),
then renders a plain-text research report. No fills/costs/backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.db import query_candles
from src.data.symbols import load_universe_config
from src.research.panel import load_daily_panel
from src.research.stats import (
    bench_regimes,
    catalyst_intraday,
    momentum_sweep,
    regime_stats,
    residual_dispersion,
    residual_returns,
    vol_clustering,
)
from src.research.types import (
    CatalystResult,
    DispersionResult,
    MomentumCell,
    PanelInfo,
    RegimeCell,
    VolClusterResult,
)


def run_scan(
    universe_path: str,
    bench: str,
    from_date: str | None,
    to_date: str | None,
) -> str:
    """Build the panel, compute every family, return the rendered report."""
    conf = load_universe_config(universe_path)
    names: list[str] = [s for s in conf.symbols if s.upper() != bench.upper()]

    close_frame, bench_close, info = load_daily_panel(
        conf.symbols, bench, from_date=from_date, to_date=to_date
    )

    ret = close_frame.pct_change(fill_method=None)
    bench_ret = bench_close.pct_change(fill_method=None)
    resid = residual_returns(ret, bench_ret)

    disp = residual_dispersion(resid)
    mom = momentum_sweep(ret, resid)
    vol = vol_clustering(resid)
    bucket = bench_regimes(bench_ret)
    regimes = regime_stats(bucket, resid)

    cataly = _catalyst_scan(names, info, from_date, to_date)

    lines: list[str] = []
    lines.append(f"# Research scan — {universe_path}")
    lines.append(f"Panel benchmark: {bench}")
    _push_panel(lines, info)
    lines.append("")
    lines.append("## 1. Residual diversification")
    lines.append(_fmt_dispersion(disp))
    lines.append("  > " + _dispersion_hint(disp))
    lines.append("")
    lines.append("## 2. Cross-sectional momentum / reversal forecast")
    for cell in mom:
        lines.append(_fmt_momentum(cell))
    lines.append(
        "     > net spread is residual-separated; raw spread shows beta contamination"
    )
    lines.append("")
    lines.append("## 3. Vol clustering & spike response (residual daily)")
    lines.append(_fmt_vol(vol))
    lines.append("")
    lines.append("## 4. Intraday catalyst drift-vs-fade (native 1h)")
    lines.append(_fmt_catalyst(cataly))
    lines.append("")
    lines.append("## 5. Benchmark-regime stability")
    for cell in regimes:
        lines.append(_fmt_regime(cell))
    lines.append("")
    lines.append(_summary_block(disp, mom, vol, cataly, regimes))
    return "\n".join(lines)


def _push_panel(lines: list[str], info: PanelInfo) -> None:
    members = ", ".join(info.members)
    lines.append(f"Panel: {info.n_members} names, {info.n_common_rows} common days")
    lines.append(f"Common window: {info.common_start} -> {info.common_end}")
    lines.append(f"Members: {members}")
    if info.dropped:
        lines.append(f"Dropped short-history stragglers: {', '.join(info.dropped)}")


def _fmt_dispersion(disp: DispersionResult) -> str:
    rho = disp.mean_pairwise_rho
    pc1 = disp.pc1_var_share
    eff = disp.effective_n
    return (
        f"mean pairwise residual rho={_pf(rho)} pc1_share={_pf(pc1)} "
        f"effective_n={_pf(eff)} (panel {disp.n_members}x{disp.n_rows})"
    )


def _pf(v: float | None, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_momentum(cell: MomentumCell) -> str:
    caveat = "  <n<30 low confidence>" if (cell.n_rows and cell.n_rows < 30) else ""
    if cell.n_effective_too_small:
        eff = "t_eff<n30 n_eff={cidx}".format(cidx=_n_int(cell.n_effective))
    else:
        eff = f"t_eff={_pf(cell.t_effective)} (n_eff={_n_int(cell.n_effective)})"
    return (
        f"k={cell.lookback:>2}d h={cell.horizon:>2}d  "
        f"spearman={_pf(cell.spearman)} "
        f"t_pooled(overlap)={_pf(cell.t_stat)} "
        f"{eff} "
        f"netTopBot={_bp(cell.net_decile_bps)} "
        f"rawTopBot={_bp(cell.raw_decile_bps)} "
        f"n_pooled={_n_int(cell.n_rows or 0)}{caveat}"
    )


def _n_int(v: float | int | None) -> str:
    return "n/a" if v is None else f"{int(v):,}"


def _bp(bps: float | None) -> str:
    return "n/a" if bps is None else f"{bps:+.1f}bps"


def _fmt_vol(vol: VolClusterResult) -> str:
    caveat = " <spike n<30 low confidence>" if vol.spike_n < 30 else ""
    ac = f"({_pf(vol.mean_ac1)}, {_pf(vol.mean_ac5)}, {_pf(vol.mean_ac20)})"
    # Read the conditional-minus-unconditional CONTRAST, not the raw spike mean:
    # the unconditional base itself moves with the sector, so a lone condition
    # number can mislead (e.g. biotech spike=-10 vs nsdq spike=+3 differ largely
    # because their unconditional bases differ).
    contrast: str
    if (
        vol.spike_fwd_mean_resid is not None
        and vol.unconditional_resid_mean is not None
    ):
        diff = (vol.spike_fwd_mean_resid - vol.unconditional_resid_mean) * 1e4
        contrast = f"spike-minus-uncond={diff:+.2f}bps/d"
    else:
        contrast = "spike-minus-uncond=n/a"
    return (
        f"mean realized-vol AC(l1|5|20)={ac} "
        f"spike_n={vol.spike_n} fwd1d_resid={_daily_bps(vol.spike_fwd_mean_resid)} "
        f"uncond={_daily_bps(vol.unconditional_resid_mean)} "
        f"({contrast}){caveat}"
    )


def _daily_bps(ret: float | None) -> str:
    """A daily return expressed in basis points per day for a readable report."""
    return "n/a" if ret is None else f"{ret * 1e4:+.2f}bps/d"


def _fmt_catalyst(cataly: CatalystResult) -> str:
    out: list[str] = [
        f"events(n={cataly.total_events}) threshold={cataly.threshold_sigma:g}σ names={cataly.n_names}"
    ]
    for side in cataly.sides:
        label = side.fwd_1h_mean_bps
        out.append(
            f"  {side.direction:<5} n={side.events}".ljust(26)
            + f"fwd1h={_bp(label)} fwd24h={_bp(side.fwd_24h_mean_bps)}"
        )
    # fwd24h windows are counted per (name,event); when extreme hourly events for a
    # name are less than ~24h apart their forward windows overlap, so fwd24h treats
    # cluster members once each and can overrepresent a single sustained catalyst.
    out.append(
        "  note: fwd24h windows may overlap when a name repeats extreme events <24h "
        "apart -> read fwd24h as drift, not a per-independent-event expectation"
    )
    return "\n".join(out)


def _fmt_regime(cell: RegimeCell) -> str:
    if cell.n_effective_too_small:
        te = "t_eff<n30"
    else:
        te = f"t_eff={_pf(cell.t_effective, 2)} (n_eff={_n_int(cell.n_effective)})"
    return (
        f"regime={cell.regime:<5} n_pooled={cell.n_rows:>5} "
        f"net21x21TopBot={_bp(cell.net_decile_bps)} residualRho={_pf(cell.mean_pairwise_rho)} "
        f"spearman_eff={_pf(cell.spearman_effective)} {te}"
    )


def _catalyst_scan(
    names: list[str],
    info: PanelInfo,
    from_date: str | None,
    to_date: str | None,
) -> CatalystResult:
    """Load native 1h log-returns for retained members, run the drift/fade test."""
    hourly: dict[str, pd.Series] = {}
    for ticker in names:
        if ticker not in info.members:
            continue
        try:
            df = query_candles(ticker.upper(), bar="1h")
        except Exception:
            continue
        if df.empty:
            continue
        if from_date:
            df = df.loc[pd.Timestamp(from_date) :]
        if to_date:
            df = df.loc[: pd.Timestamp(to_date)]
        close = df["close"].astype(float)
        hourly[ticker] = np.log(close).diff().dropna()
    return catalyst_intraday(hourly)


def _dispersion_hint(disp: DispersionResult) -> str:
    if disp.effective_n is None or disp.effective_n >= 8.0:
        return "effective independent count is high -> real single-name dispersion to build on"
    return "low effective independent count (one-blob residual) -> limited single-name dispersion money"


def _summary_block(
    disp: DispersionResult,
    mom: tuple[MomentumCell, ...],
    vol: VolClusterResult,
    cataly: CatalystResult,
    regimes: tuple[RegimeCell, ...],
) -> str:
    """Short, honest read of which families offer strategy-buildable edge."""
    momentum_read = _momentum_verdict(mom)
    if (vol.mean_ac1 or 0) > 0.9:
        vol_read = "vol clustering strong; "
    else:
        vol_read = ""
    # Judge the conditional-vs-unconditional contrast (see _fmt_vol).
    contrast = None
    if (
        vol.spike_fwd_mean_resid is not None
        and vol.unconditional_resid_mean is not None
    ):
        contrast = vol.spike_fwd_mean_resid - vol.unconditional_resid_mean
    if contrast is not None and contrast < 0:
        vol_read += (
            "after a vol spike residual next-day leans mean-reverting (spike<uncond)"
        )
    elif contrast is not None and contrast > 0:
        vol_read += "after a vol spike residual next-day drifts positive (spike>uncond)"
    else:
        vol_read += "no spike-vs-unconditional contrast in residual vol response"
    down = [c for c in regimes if c.regime == "down"][0]
    up = [c for c in regimes if c.regime == "up"][0]
    stable = bool(
        down.net_decile_bps is not None
        and up.net_decile_bps is not None
        and down.net_decile_bps < 0
        and up.net_decile_bps < 0
    ) or bool(
        down.net_decile_bps is not None
        and up.net_decile_bps is not None
        and down.net_decile_bps > 0
        and up.net_decile_bps > 0
    )
    if stable:
        regime_hint = "reversal/momentum sign is stable across up/down regimes"
    else:
        regime_hint = "sign differs across up/down regimes -> regime-dependent (handle per-regime)"
    lines = [
        "## Summary & method notes",
        f"dispersion: {_dispersion_hint(disp)} ({_pf(disp.effective_n)} effective names)",
        f"momentum: {momentum_read}",
        f"vol: {vol_read}",
        f"regime: {regime_hint}",
        _catalyst_verdict(cataly),
        "-",
        "Notes: beta removed via rolling-63d OLS vs the single benchmark.",
        "Momentum/regime rows pool every overlapping (member,day) window, so the pooled",
        "t-stat counts ~max(k,h)-fold redundant rows as independent and OVERSTATES",
        "confidence. Gate strategy decisions on t_effective (stride-decimated, near-",
        "independent subsample), NOT the pooled t. Point estimates (spearman, decile",
        "bps) are unchanged. Regime rows show t_effective too. Family 3 'fwd1d_resid'",
        "is a conditional mean; read the spike-vs-unconditional CONTRAST, not the raw",
        "number, because the unconditional base itself shifts with the sector. Family 4",
        "fwd24h windows can overlap when adjacent extreme hourly events are <24h apart,",
        "so fwd24h counts cluster-derived events once each. No fees/slippage/backtest.",
    ]
    return "\n".join(lines)


def _momentum_verdict(cells: tuple[MomentumCell, ...]) -> str:
    by_horizon: dict[int, list[MomentumCell]] = {}
    for c in cells:
        by_horizon.setdefault(c.horizon, []).append(c)
    out: list[str] = []
    for horizon in sorted(by_horizon):
        span = by_horizon[horizon]
        sign = None
        for c in span:
            if c.net_decile_bps is not None and abs(c.net_decile_bps) > 0:
                sign = c.net_decile_bps
                break
        # strongest de-overlapped significance across lookbacks of this horizon.
        # Only cells with a real effective t count toward the confidence gate.
        te = [c.t_effective for c in span if c.t_effective is not None]
        max_abs_te = max((abs(x) for x in te), default=0.0)
        te_tag = f"|t_eff|max={max_abs_te:.1f}" if te else "t_eff=n/a(<30 eff sample)"
        if sign is None:
            out.append(f"h{horizon}d: n/a ({te_tag})")
            continue
        direction = "mean-reversion candidate" if sign < 0 else "momentum candidate"
        confidence = "gated(clear)" if max_abs_te >= 2.0 else "NOT gated(|t_eff|<2)"
        has_pos = any(
            (c.net_decile_bps or 0) > 0 for c in span if c.net_decile_bps is not None
        )
        has_neg = any(
            (c.net_decile_bps or 0) < 0 for c in span if c.net_decile_bps is not None
        )
        stability = (
            "sign-stable across lookbacks"
            if has_pos != has_neg
            else "mixed across lookbacks"
        )
        out.append(f"h{horizon}d: {direction} ({te_tag}; {confidence}; {stability})")
    return "; ".join(out) or "n/a"


def _catalyst_verdict(cataly: CatalystResult) -> str:
    up = next((s for s in cataly.sides if s.direction == "up"), None)
    neg1 = bool(up and up.fwd_1h_mean_bps is not None and up.fwd_1h_mean_bps < 0)
    if neg1:
        return "intraday: big up catalyst fades in next hour (reversal); check 24h drift separately"
    return "intraday: no clear first-hour fade on catalyst moves"
