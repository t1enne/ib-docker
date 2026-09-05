"""Immutable result types for the `research` cross-sectional statistics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PanelInfo:
    """The rectangular daily panel produced for a scan."""

    members: tuple[str, ...]
    bench: str
    n_members: int
    n_common_rows: int
    common_start: str
    common_end: str
    dropped: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispersionResult:
    """Residual diversification structure (family 1)."""

    mean_pairwise_rho: float | None
    pc1_var_share: float | None
    effective_n: float | None
    n_members: int
    n_rows: int


@dataclass(frozen=True)
class MomentumCell:
    """One (lookback, horizon) cross-sectional momentum/reversal probe.

    ``spearman``/``t_stat``/``n_rows`` are the *pooled* estimates over every
    trailing/forward  (member, day) window. Because consecutive windows overlap
    by ``max(lookback, horizon) - 1`` days of every ``max(lookback, horizon)``,
    the pooled ``t_stat`` treats ~N-fold-redundant rows as independent and thus
    overstates confidence. ``t_effective`` restates significance on a
    stride-decimated (near-independent) subsample and is the number a strategy
    decision should be gated on; it may be ``None`` when the effective sample is
    too small (<30 rows). Point estimates (``spearman``, decile bps) are
    unchanged.
    """

    lookback: int
    horizon: int
    spearman: float | None
    t_stat: float | None  # pooled (overlap-inflated) significance
    n_rows: int  # pooled sample size
    net_decile_bps: (
        float | None
    )  # residual-return top-minus-bottom decile, bps/day of horizon
    net_n: int
    raw_decile_bps: (
        float | None
    )  # raw-return top-minus-bottom decile (beta contamination view)
    raw_n: int
    # de-overlapped independent-sample significance (the honest confidence gate)
    spearman_effective: float | None = None
    t_effective: float | None = None
    n_effective: int = 0
    n_effective_too_small: bool = False


@dataclass(frozen=True)
class VolClusterResult:
    """Per-name vol autocorrelation + conditional-spike forward move."""

    mean_ac1: float | None
    mean_ac5: float | None
    mean_ac20: float | None
    n_members: int
    spike_fwd_mean_resid: float | None
    spike_n: int
    unconditional_resid_mean: float | None


@dataclass(frozen=True)
class CatalystSide:
    """One direction's catalyst-drift/fade statistics (intraday 1h)."""

    direction: str
    events: int
    fwd_1h_mean_bps: float | None
    fwd_24h_mean_bps: float | None


@dataclass(frozen=True)
class CatalystResult:
    """Extreme 1-hour move drift-vs-fade (family 4)."""

    sides: tuple[CatalystSide, ...] = field(default_factory=tuple)
    total_events: int = 0
    threshold_sigma: float = 2.5
    n_names: int = 0


@dataclass(frozen=True)
class RegimeCell:
    """One benchmark-regime subset's key statistics.

    The regime decile spread pools overlapping 21->21 windows (one per trading
    day), so the row count ``n_rows`` overstates independence for the spread's
    confidence. ``t_effective`` restates the cross-sectional reversal/momentum
    significance of that regime on a stride-decimated (near-independent)
    subsample and is what a strategy gate should use.
    """

    regime: str  # up / flat / down
    bench_slope_lo: float | None
    bench_slope_hi: float | None
    n_rows: int
    net_decile_bps: float | None  # 21->21 residual top-minus-bottom
    mean_pairwise_rho: float | None
    spearman_effective: float | None = None
    t_effective: float | None = None
    n_effective: int = 0
    n_effective_too_small: bool = False
