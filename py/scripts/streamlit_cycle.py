#!/usr/bin/env -S uv run
"""
Cycle Screener Dashboard — Visual regime ratio explorer.

Replaces the arbitrary-threshold terminal screener with an interactive
Streamlit dashboard. Plots the full history of each cross-asset ratio
with rolling statistics so you can visually assess regime state.

Usage:
    uv run streamlit run scripts/streamlit_cycle.py

Layers (top → bottom): (1) Market-Health composite verdict as a *smoothed wave* —
the per-day mean of the fresh ratio legs' trailing-z scores (not a single
number), set against RISK-ON / NEUTRAL / RISK-OFF threshold bands; (2) a Macro
Cycle / FRED backdrop drawn from the packaged lookahead-safe macro assets (GDP,
unemployment, payrolls, capacity, industrial), each tile showing its real
as-of release month; (3) a staleness warning when any leg that would otherwise
vote stopped being current; then the usual per-ratio history charts and
summary table.
"""

from __future__ import annotations
from src.data import resample_ohlcv

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from src.utils import get_local_candles  # noqa: E402  (after repo-root shim)

# ── Page config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Cycle & Market Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ratio definitions ordered by signal hierarchy ──
# A deliberately lean set — each ratio earns its place on a DISTINCT axis of
# risk, no two-leg redundancy. 1. Credit risk, the classic leading regime
# signal (HYG/TLT); 2. Yield-curve slope (IEF/TLT), the private recession
# clock; 3. Equity breadth (IWM/SPY); 4. Real-economy copper demand (CPER/GLD);
# 5. Credit transmission into financials (XLF/XLU). Deliberately slimmed from
# the over-fitted set (see git history): dropped the ÷GLD commodity redundancy
# (SLV, USO), the secular QQQ/SPY drift, discretionary XLY/XLP (breadth already
# spans it), and the second credit ratio HYG/LQD (HYG/TLT carries the signal).
# SPY/TLT was also dropped: at the useful (126d+) forward-equity horizon it is a
# NEGATIVE predictor (the SPY numerator makes it a trailing-equity-momentum
# proxy, contaminating a cross-asset regime read); the equity outcome is better
# signalled by HYG, IEF/TLT curve and XLF credit legs (see scripts/check_legs*).
_RATIO_DEFS: list[tuple[str, str, str]] = [
    ("HYG / TLT (credit risk)", "HYG", "TLT"),
    ("IEF / TLT (curve slope)", "IEF", "TLT"),
    ("IWM / SPY (equity breadth)", "IWM", "SPY"),
    ("CPER / GLD (copper demand)", "CPER", "GLD"),
    ("XLF / XLU (credit-financials)", "XLF", "XLU"),
]
# Composite = the full (fresh) ratio set: every signal shown votes on the
# health verdict, so the verdict is transparently the plotted legs.
_COMPOSITE_LEGS: tuple[tuple[str, str, str], ...] = tuple(_RATIO_DEFS)

RiskPhase = Literal["RISK-ON", "NEUTRAL", "RISK-OFF", "INSUFFICIENT"]

# Minimum trading days a ratio leg must have before it may vote in the health
# composite (keeps blink-and-gone series from tilting the verdict).
MIN_HISTORY_DAYS: int = 90
# Confidence the composite needs across voting legs before we trust the phase.
MIN_QUORUM_FRACTION: float = 0.30
# A leg is "stale" when its most recent bar trails the freshest series by more
# than this many trading days — stale legs never vote in *today's* health read.
STALE_TRADING_DAYS: int = 7
# Rolling window (fast-settling proxy) used to normalise a ratio level to a
# z-score for the health composite.
HEALTH_Z_WINDOW: int = 126


# Composite phase decision boundaries on the mean z of the voting legs.
HEALTH_RISKON_Z: float = 0.5


# ── Data fetch ─────────────────────────────────────────────────


@st.cache_resource
def fetch_daily_close(symbol: str) -> pd.Series:
    """Fetch daily close prices for a symbol."""
    df = get_local_candles(symbol, bar="1h")
    if df.empty:
        return pd.Series(dtype=float)
    daily = resample_ohlcv(df, "1d")
    if daily.empty:
        return pd.Series(dtype=float)
    return daily["close"]


def compute_ratio_series(
    num_series: pd.Series,
    den_series: pd.Series,
) -> pd.Series:
    """Compute ratio of two aligned price series."""
    combined = pd.DataFrame({"num": num_series, "den": den_series}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    return combined["num"] / combined["den"]


def compute_rolling_stats(
    ratio: pd.Series,
    window: int = 63,
) -> dict[str, pd.Series]:
    """Compute rolling mean and ±2σ bands for a ratio series."""
    roll = ratio.rolling(window, min_periods=max(5, window // 4)).agg(["mean", "std"])
    roll_mean, roll_std = roll["mean"], roll["std"]
    return {
        "sma": roll_mean,
        "upper": roll_mean + 2 * roll_std,
        "lower": roll_mean - 2 * roll_std,
    }


def band_z(ratio: pd.Series, stats: dict[str, pd.Series]) -> float | None:
    """Latest value as band-width z: (cur - sma) / sigma (sigma = band/4).

    ``None`` when the band is degenerate (unavailable/zero width) — e.g. a
    level ratio pinned near a constant whose z cannot be read.
    """
    upper = stats["upper"].iloc[-1]
    lower = stats["lower"].iloc[-1]
    if pd.isna(upper) or pd.isna(lower) or upper - lower <= 0:
        return None
    sigma = (upper - lower) / 4.0
    return (ratio.iloc[-1] - stats["sma"].iloc[-1]) / sigma


# A leg is skipped as "flat" when its trailing dispersion is ~zero (a level
# ratio pinned near a constant), because its z cannot be read meaningfully.
@dataclass(frozen=True)
class LegReading:
    """One candidate ratio leg and its health vote (used or skipped)."""

    label: str
    z: float
    n_points: int
    last_date: pd.Timestamp
    used: bool
    skip_reason: str | None


@dataclass(frozen=True)
class HealthSnapshot:
    """Composite risk-on/off verdict plus the raw leg readings that built it."""

    phase: RiskPhase
    mean_z: float
    confidence: float
    used_count: int
    candidate_count: int
    skipped_labels: tuple[str, ...]
    legs: tuple[LegReading, ...]


def ratio_health_z(ratio: pd.Series, window: int = HEALTH_Z_WINDOW) -> float | None:
    """Signed z-score of the latest ratio value against its own trailing window.

    Returns ``None`` when there is not enough history to compute a stable
    trailing mean/std (fewer than ``MIN_HISTORY_DAYS`` points) or the trailing
    dispersion is ~zero. Each leg is normalised on its *own* trailing window so
    a weak 1.2 and a strong 0.9 ratio both land on a shared scale the composite
    can average.
    """
    if len(ratio) < MIN_HISTORY_DAYS:
        return None
    agg = ratio.rolling(window, min_periods=MIN_HISTORY_DAYS).agg(["mean", "std"])
    z_series = (ratio - agg["mean"]) / agg["std"].replace(0.0, np.nan)
    latest = z_series.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)


def _skim_closes(
    closes: Mapping[str, pd.Series], num_ticker: str, den_ticker: str
) -> pd.DataFrame:
    """Align a num/den pair onto a shared (non-null) daily frame, empty-safe."""
    num_s = closes.get(num_ticker)
    den_s = closes.get(den_ticker)
    if num_s is None or den_s is None or num_s.empty or den_s.empty:
        return pd.DataFrame()
    return pd.DataFrame({"num": num_s, "den": den_s}).dropna()


def _dt_ix_last(index: pd.DatetimeIndex) -> pd.Timestamp:
    """Last timestamp of a non-empty DatetimeIndex.

    ``index[-1]`` is always a real printed bar in our data; pandas stubs type it
    ``Timestamp | NaTType`` so we strip that with an explicit cast (the union's
    NaT member is unreachable for a resampled daily calendar).
    """
    assert len(index) > 0, "index must be non-empty"
    return cast(pd.Timestamp, index[-1])


def count_trailing_days(index: pd.DatetimeIndex, after: pd.Timestamp) -> int:
    """Number of entries in ``index`` strictly later than ``after``."""
    return int((index > after).sum())


def _classify_leg(
    closes: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
    label: str,
    num_ticker: str,
    den_ticker: str,
    freshest: pd.Timestamp,
) -> tuple[LegReading, float | None]:
    """Classify one ratio leg: a reading plus the z to vote (None = no vote)."""
    joint = _skim_closes(closes, num_ticker, den_ticker)
    n_points = len(joint)
    if n_points == 0:
        return (LegReading(label, 0.0, 0, freshest, False, "no joint data"), None)
    if n_points < MIN_HISTORY_DAYS:
        last_date = _dt_ix_last(pd.DatetimeIndex(joint.index))
        return (
            LegReading(
                label, 0.0, n_points, last_date, False, f"history < {MIN_HISTORY_DAYS}d"
            ),
            None,
        )
    ratio = (joint["num"] / joint["den"]).dropna()
    if ratio.empty:
        return (LegReading(label, 0.0, n_points, freshest, False, "empty ratio"), None)
    last_date = _dt_ix_last(pd.DatetimeIndex(ratio.index))
    stale_days = count_trailing_days(calendar, last_date)
    if stale_days > STALE_TRADING_DAYS:
        return (
            LegReading(
                label,
                0.0,
                len(ratio),
                last_date,
                False,
                f"stale {stale_days}d > {STALE_TRADING_DAYS}d",
            ),
            None,
        )
    z = ratio_health_z(ratio)
    if z is None:
        return (
            LegReading(
                label, 0.0, len(ratio), last_date, False, "flat / unstable dispersion"
            ),
            None,
        )
    return (LegReading(label, z, len(ratio), last_date, True, None), z)


def _decode_votes(used_z: list[float]) -> tuple[float, float, RiskPhase]:
    """Mean z, agreement confidence, and phase from the voting leg z's.

    ``used_z`` is non-empty (callers only reach this after quorum passes). Phase
    is RISK-ON above ``HEALTH_RISKON_Z``, RISK-OFF below its negation, else
    NEUTRAL. Confidence is the fraction of legs whose sign matches the mean.
    """
    mean_z = float(np.mean(used_z))
    agreement = sum(1.0 for z in used_z if np.sign(z) == np.sign(mean_z))
    confidence = agreement / len(used_z)
    if mean_z > HEALTH_RISKON_Z:
        phase: RiskPhase = "RISK-ON"
    elif mean_z < -HEALTH_RISKON_Z:
        phase = "RISK-OFF"
    else:
        phase = "NEUTRAL"
    return mean_z, confidence, phase


def build_health_snapshot(
    closes: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> HealthSnapshot:
    """Assemble the composite risk-on/off snapshot from the fresh ratio legs.

    A leg votes only when it (a) has >= ``MIN_HISTORY_DAYS`` of joint history,
    (b) carries a normalisable z, and (c) is not stale (its last bar trails the
    freshest series by more than ``STALE_TRADING_DAYS``). Skipped legs are
    surfaced on ``skipped_labels`` so a healthy-looking verdict can never hide a
    leg that stopped reporting. With <30% of candidates voting, ``INSUFFICIENT``.
    """
    candidate_count: int = len(_COMPOSITE_LEGS)
    freshest = _dt_ix_last(calendar)
    readings: list[LegReading] = []
    used_z: list[float] = []

    for label, num_ticker, den_ticker in _COMPOSITE_LEGS:
        reading, vote = _classify_leg(
            closes, calendar, label, num_ticker, den_ticker, freshest
        )
        readings.append(reading)
        if vote is not None:
            used_z.append(vote)

    used_count = len(used_z)
    skipped_labels: tuple[str, ...] = tuple(r.label for r in readings if not r.used)
    quorum_ok = used_count / candidate_count > MIN_QUORUM_FRACTION
    if not quorum_ok or used_count == 0:
        return HealthSnapshot(
            phase="INSUFFICIENT",
            mean_z=0.0,
            confidence=0.0,
            used_count=used_count,
            candidate_count=candidate_count,
            skipped_labels=skipped_labels,
            legs=tuple(readings),
        )

    mean_z, confidence, phase = _decode_votes(used_z)
    return HealthSnapshot(
        phase=phase,
        mean_z=mean_z,
        confidence=confidence,
        used_count=used_count,
        candidate_count=candidate_count,
        skipped_labels=skipped_labels,
        legs=tuple(readings),
    )


# ── Sidebar ────────────────────────────────────────────────────

st.sidebar.title("Cycle Ratios")

lookback = st.sidebar.slider(
    "Lookback (trading days)",
    min_value=60,
    max_value=126,
    value=504,
    step=21,
)

rolling_window = st.sidebar.slider(
    "Rolling window (days)",
    min_value=21,
    max_value=252,
    value=63,
    step=21,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Each plot shows the ratio history (blue), a rolling mean (orange), "
    "and ±2σ bands (shaded). Use these to visually assess where the "
    "current ratio sits relative to its own history — no arbitrary thresholds."
)

# ── FRED macro layer (cycle / health backdrop) ────────────────

# Which packaged FRED asset files surface as health/cycle KPIs. Cadence and
# coverage live server-side in assets/<name>.csv (see scripts/fetch_macro_fred.py
# and src.indicators.macro); discovery here is by the loader's accepted names.
_MACRO_KPI_NAMES: tuple[str, ...] = (
    "gdpc1",  # real GDP, quarterly level
    "unrate",  # unemployment rate, %
    "payems",  # nonfarm payrolls, monthly level
    "indpro",  # industrial production index
    "tcu",  # capacity utilisation, %
    "dgs2",  # 2-year treasury constant-maturity yield, % daily
    "t10y2y",  # 10Y minus 2Y treasury spread, pp daily (recession clock)
)


@dataclass(frozen=True)
class MacroKpi:
    """One FRED indicator presented as a health/cycle tile."""

    name: str
    title: str
    value: float
    unit: str
    as_of: str
    read: str


def _load_macro_series(name: str, cursor: pd.Timestamp) -> pd.Series:
    """Packaged daily FF'd FRED series truncated to prints at/before ``cursor``.

    ``load_daily`` is the same loader that backs ``init_macro_indicator`` for
    FRED two-column names, so its values are byte-identical to what the
    indicator callable returns at the cursor — keeps the macro layer on the
    library's single source of truth without reaching into private state.
    """
    from src.indicators.macro._shared import load_daily

    series = load_daily(name, source_dir="assets")
    return series[series.index <= cursor]


def _print_index(series: pd.Series) -> pd.DatetimeIndex:
    """Index positions of the real (non-ffilled) FRED prints within ``series``.

    A daily FF'd series is piecewise-constant between prints; a new print shows
    as the first row whose value differs from the prior row, which the loader
    emits at the actual observation (month/quarter) date.
    """
    arr = series.to_numpy()
    change = np.diff(arr, prepend=arr[:1]) != 0
    positions = series.index[change]
    # The first kept row is also a genuine print when the history begins there.
    return pd.DatetimeIndex(positions)


def _latest_two_values(series: pd.Series) -> tuple[float, float] | None:
    """The two most recent real print values (oldest, newest) or None."""
    starts = _print_index(series)
    if len(starts) < 2:
        return None
    last_ts, prev_ts = starts[-1], starts[-2]
    latest = float(series.loc[last_ts])
    previous = float(series.loc[prev_ts])
    return (previous, latest)


def _asof(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """Value in effect at/just before ``ts``, else None outside coverage."""
    if series.empty or ts < series.index[0]:
        return None
    found = series.index.asof(ts)
    if pd.isna(found):
        return None
    val = float(series.loc[found])
    return val if pd.notna(val) else None


def _annualised_qoq(prev_q: float, curl_q: float) -> float | None:
    """Annualised quarter-over-quarter growth in %. None when base invalid."""
    if prev_q <= 0 or curl_q <= 0:
        return None
    return (pow(curl_q / prev_q, 4.0) - 1.0) * 100.0


def _yoy_pct(series: pd.Series) -> float | None:
    """Latest print vs the same month ~1 year earlier, in %."""
    starts = _print_index(series)
    if len(starts) == 0:
        return None
    latest_ts = starts[-1]
    latest_v = float(series.loc[latest_ts])
    a_year_ago = latest_ts - pd.DateOffset(months=12)
    prev_v = _asof(series, a_year_ago)
    if prev_v is None or prev_v == 0 or pd.isna(prev_v):
        return None
    return (latest_v / prev_v - 1.0) * 100.0


def _rate_dir(series: pd.Series, months: int) -> str:
    """One-line direction read for a rate-style level series (unrate/TCU)."""
    starts = _print_index(series)
    if len(starts) == 0:
        return "no recent print"
    latest_ts = starts[-1]
    latest_v = float(series.loc[latest_ts])
    earlier_v = _asof(series, latest_ts - pd.DateOffset(months=months))
    if earlier_v is None:
        return "recent print only"
    move = latest_v - earlier_v
    if move > 0.05:
        phrase = "rising"
    elif move < -0.05:
        phrase = "falling"
    else:
        phrase = "~flat"
    return f"{phrase} over {months}mo"


def _gdp_kpi(series: pd.Series) -> tuple[float, str, str]:
    """(value, unit, read) for the real-GDP quarterly growth tile."""
    vals = _latest_two_values(series)
    ann = _annualised_qoq(vals[0], vals[1]) if vals is not None else None
    if ann is None:
        return (float("nan"), "% ann.q/q", "insufficient GDP prints")
    state = "in expansion" if ann >= 0 else "in contraction"
    return (ann, "% ann. q/q", f"real GDP {state}")


def _unrate_kpi(series: pd.Series) -> tuple[float, str, str]:
    """(value, unit, read) for the unemployment-rate tile."""
    starts = _print_index(series)
    if len(starts) == 0:
        return (float("nan"), "%", "no print")
    level = float(series.loc[starts[-1]])
    return (level, "%", _rate_dir(series, 3))


def _util_kpi(series: pd.Series) -> tuple[float, str, str]:
    """(value, unit, read) for the capacity-utilisation tile."""
    starts = _print_index(series)
    if len(starts) == 0:
        return (float("nan"), "%", "no print")
    level = float(series.loc[starts[-1]])
    return (level, "%", _rate_dir(series, 12))


def _curve_kpi(name: str, series: pd.Series) -> tuple[float, str, str]:
    """Read for a treasury daily micro-series: 2Y level (``dgs2``) or 2s10s.

    ``dgs2`` is an absolute constant-maturity yield (%). ``t10y2y`` is the
    10Y−2Y spread in pp — already pre-computed by FRED so its sign is the
    inversion clock directly (negative = inverted, an historic recession lead).
    Direction spans the ~3-month window because the series is daily, not
    monthly/quarterly like the other KPI tiles.
    """
    starts = _print_index(series)
    if len(starts) == 0:
        return (float("nan"), "%" if name == "dgs2" else "pp", "no print")
    latest_ts = starts[-1]
    latest_v = float(series.loc[latest_ts])
    if name == "dgs2":
        return (latest_v, "%", f"2Y yield {_rate_dir(series, 3)}")
    # t10y2y — combine the spread's sign (inverted or not) with its 3mo move.
    earlier_v = _asof(series, latest_ts - pd.DateOffset(months=3))
    if earlier_v is None:
        return (latest_v, "pp", "recent curve print only")
    moved = latest_v - earlier_v
    if latest_v < 0:
        shade = "inverted" + (" & deepening" if moved < 0 else " & recovering")
    else:
        shade = "positive" + (" & steepening" if moved > 0 else " & flattening")
    return (latest_v, "pp", f"2s10s {shade} over 3mo")


def _momentum_kpi(series: pd.Series, title: str) -> tuple[float, str, str]:
    """(value, unit, read) for a growth-series tile (payrolls / industrial)."""
    yoy = _yoy_pct(series)
    if yoy is None:
        return (float("nan"), "% y/y", f"{title}: insufficient history")
    direction = "rising" if yoy >= 0 else "contracting"
    return (yoy, "% y/y", f"{title} {direction}")


def _as_series(name: str, cursor: pd.Timestamp) -> pd.Series | None:
    """Packaged daily FF'd FRED series at/for ``name`` as-of ``cursor``."""
    try:
        series = _load_macro_series(name, cursor)
    except FileNotFoundError:
        return None
    return None if series.empty else series


def build_fred_kpis(cursor: pd.Timestamp) -> tuple[MacroKpi, ...]:
    """Read each packaged FRED series as-of ``cursor`` into labelled KPIs."""
    titles: dict[str, str] = {
        "gdpc1": "Real GDP",
        "payems": "Nonfarm payrolls",
        "indpro": "Industrial output",
        "unrate": "Unemployment rate",
        "tcu": "Capacity utilisation",
        "dgs2": "2Y Treasury yield",
        "t10y2y": "2s10s curve",
    }
    kpis: list[MacroKpi] = []
    for name in _MACRO_KPI_NAMES:
        series = _as_series(name, cursor)
        if series is None:
            continue
        starts = _print_index(series)
        if len(starts) == 0:
            continue
        title = titles[name]
        if name == "gdpc1":
            value, unit, read = _gdp_kpi(series)
        elif name == "unrate":
            value, unit, read = _unrate_kpi(series)
        elif name == "tcu":
            value, unit, read = _util_kpi(series)
        elif name in ("dgs2", "t10y2y"):
            value, unit, read = _curve_kpi(name, series)
        else:
            value, unit, read = _momentum_kpi(series, title)
        as_of = starts[-1].strftime("%Y-%m")
        kpis.append(MacroKpi(name, title, value, unit, as_of, read))
    return tuple(kpis)


def build_cycle_narrative(kpis: Sequence[MacroKpi]) -> tuple[str, ...]:
    """A short, directional macro-cycle readout from the FRED tiles.

    Descriptive only: these are monthly/quarterly *lagging* prints the dashboard
    cannot see revised, so the text reports direction + momentum and overlays an
    explicit recession-clock/bear gauges disclaimer rather than a hard phase tag.
    """
    by: dict[str, MacroKpi] = {k.title: k for k in kpis}
    lines: list[str] = []
    if "Real GDP" in by:
        lines.append(f"GDP: {by['Real GDP'].read} — as-of {by['Real GDP'].as_of}")
    if "Unemployment rate" in by:
        lines.append(
            f"Labour: {by['Unemployment rate'].read} — as-of {by['Unemployment rate'].as_of}"
        )
    if "Industrial output" in by:
        lines.append(
            f"Industrial: {by['Industrial output'].read} — as-of {by['Industrial output'].as_of}"
        )
    curve = by.get("2s10s curve") or by.get("2Y Treasury yield")
    if curve is not None:
        lines.append(f"Curve/Yields: {curve.read} — as-of {curve.as_of}")
    return tuple(lines)


# ── Fetch all needed data ──────────────────────────────────────

_NEEDED_TICKERS: list[str] = sorted(
    {den for _, _, den in _RATIO_DEFS} | {num for _, num, _ in _RATIO_DEFS}
)

with st.spinner("Fetching data for all cross-asset tickers..."):
    closes: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in _NEEDED_TICKERS:
        s = fetch_daily_close(ticker)
        if s.empty:
            missing.append(ticker)
        else:
            closes[ticker] = s

if missing:
    st.warning(f"No data for: {', '.join(missing)}")

if not closes:
    st.error("No ticker data available. Check database.")
    st.stop()

# ── Apply lookback ─────────────────────────────────────────────

cutoff_date: pd.Timestamp | None = None
if lookback > 0:
    all_dates = sorted({d for s in closes.values() for d in s.index})
    if len(all_dates) >= lookback:
        cutoff_date = all_dates[-lookback]
        closes = {
            k: v[v.index >= cutoff_date]  # type: ignore[misc]
            for k, v in closes.items()
        }

# ── Compute ratios & stats ─────────────────────────────────────

st.title("Macro Cycle Ratio Explorer")


def _calendar_from(closes: Mapping[str, pd.Series]) -> pd.DatetimeIndex:
    """Sorted unique union of every daily bar date across the tickers used."""
    all_dates: list[pd.Timestamp] = [ts for s in closes.values() for ts in s.index]
    return pd.DatetimeIndex(sorted(set(all_dates)))


def _leg_z_series(
    closes: Mapping[str, pd.Series],
    num_ticker: str,
    den_ticker: str,
) -> pd.Series:
    """Full trailing-z series of one ratio, defined only on days with history.

    Mirrors :func:`ratio_health_z`'s normalisation per bar (rolling mean/std over
    ``HEALTH_Z_WINDOW``, ``NaN`` until ``MIN_HISTORY_DAYS`` and on flat/zero
    dispersion). ``NaN`` wherever the leg cannot yet speak — those days the leg
    simply does not contribute to the composite.
    """
    joint = _skim_closes(closes, num_ticker, den_ticker)
    if joint.empty or len(joint) < MIN_HISTORY_DAYS:
        return pd.Series(dtype=float)
    ratio = (joint["num"] / joint["den"]).dropna()
    if ratio.empty:
        return pd.Series(dtype=float)
    agg = ratio.rolling(HEALTH_Z_WINDOW, min_periods=MIN_HISTORY_DAYS).agg(
        ["mean", "std"]
    )
    return (ratio - agg["mean"]) / agg["std"].replace(0.0, np.nan)


def _alive_mask(n_contributors: pd.Series) -> pd.Series:
    """Days where enough legs hold a reading to justify a composite mean."""
    frac = n_contributors / len(_COMPOSITE_LEGS)
    return frac > MIN_QUORUM_FRACTION


def composite_wave(closes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Composite risk as a per-day smoothed wave over the window.

    Returns a DataFrame indexed by the union trading ``calendar`` with
    ``mean_z`` (cross-leg mean of each leg's trailing-126d z, leg contributes
    only where it has >=MIN_HISTORY_DAYS history and non-flat dispersion) and
    ``smooth`` (``mean_z`` exponentially smoothed). Days below the quorum
    threshold, or with no contributing leg, are NaN (no area to call a verdict).
    Each leg's full-history z is used, so the wave is lookahead-free by
    construction; the terminal value aligns with ``build_health_snapshot``.
    """
    calendar = _calendar_from(closes)
    if len(calendar) == 0:
        return pd.DataFrame()
    z_by_leg: dict[str, pd.Series] = {}
    for label, num_ticker, den_ticker in _COMPOSITE_LEGS:
        z = _leg_z_series(closes, num_ticker, den_ticker)
        # Carry a fresh leg's value forward up to STALE_TRADING_DAYS so a leg
        # that printed within the freshness window still votes on the following
        # global days — mirror of the "7 trading days" staleness gate. No
        # forward-fill beyond that limit keeps the wave as-of honest.
        z_by_leg[label] = z.reindex(calendar).ffill(limit=STALE_TRADING_DAYS)
    stack = pd.DataFrame(z_by_leg)  # columns = legs, index = calendar
    n_contrib = stack.notna().sum(axis=1)
    mean_z = stack.mean(axis=1, skipna=True)
    mean_z = mean_z.where(_alive_mask(n_contrib))
    smooth = mean_z.ewm(span=7, adjust=False, min_periods=3).mean()
    return pd.DataFrame(
        {"mean_z": mean_z, "smooth": smooth, "legs": n_contrib}, index=calendar
    )


def _risk_wave_figure(wave: pd.DataFrame, current_phase: RiskPhase) -> go.Figure | None:
    """Plot the smoothed composite risk wave with phase bands.

    Adds horizontal fills at ``±HEALTH_RISKON_Z`` (the RISK-OFF / RISK-ON
    threshold band) so a glance reads over-tilt vs neutral. ``None`` when the
    window has no usable composite readings.
    """
    if wave.empty or wave["smooth"].dropna().empty:
        return None
    fig = go.Figure()
    fig.add_hline(y=HEALTH_RISKON_Z, line_dash="dot", line_color="#0f7a3d", opacity=0.5)
    fig.add_hline(
        y=-HEALTH_RISKON_Z, line_dash="dot", line_color="#b00020", opacity=0.5
    )
    fig.add_trace(
        go.Scatter(
            x=wave.index,
            y=wave["smooth"],
            mode="lines",
            line=dict(color="#2e4a7d", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}<extra></extra>",
        )
    )
    # neutral band (between thresholds) shaded lightly behind the line
    fig.add_hrect(
        y0=-HEALTH_RISKON_Z,
        y1=HEALTH_RISKON_Z,
        fillcolor="rgba(128,128,128,0.08)",
        line_width=0,
        layer="below",
    )
    zone = None
    if current_phase == "RISK-ON":
        zone = "above the RISK-ON line"
        accent = "#0f7a3d"
    elif current_phase == "RISK-OFF":
        zone = "below the RISK-OFF line"
        accent = "#b00020"
    elif current_phase == "NEUTRAL":
        zone = "inside the neutral band"
        accent = "#8a6d1c"
    else:
        zone = "no reading (insufficient data)"
        accent = "#5b6472"
    fig.update_layout(
        title=dict(
            text=(
                "Composite Risk Wave"
                f"<br><sup>current {zone} · smoother = EWMA of per-leg trailing-126d z</sup>"
            ),
            x=0.01,
            font=dict(size=14, color=accent),
        ),
        xaxis_title=None,
        yaxis_title="composite z",
        template="plotly_white",
        hovermode="x unified",
        height=260,
        margin=dict(l=10, r=10, t=52, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


def _render_health(snapshot: HealthSnapshot, wave: pd.DataFrame) -> None:
    """Top-of-page market-health verdict: a smoothed risk wave, not a number."""
    st.subheader("Market Health · RISK-ON vs RISK-OFF")
    fig = _risk_wave_figure(wave, snapshot.phase)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption(
            "Only stale/short ratio legs present; no risk wave to draw "
            "(refresh credit/fixed-income data to power the composite)."
        )
    caption = (
        "Wave = cross-leg mean of each fresh leg's trailing-126d z, exponentially "
        "smoothed (EWMA span 7). A leg contributes only with ≥90 trading days "
        "history and while its last bar sits within 7 trading days of the freshest "
        "series. Above the top dashed line → RISK-ON; below the bottom → RISK-OFF; "
        "between them → NEUTRAL. Grey band marks the neutral zone."
    )
    if snapshot.used_count == 0:
        caption += " Current verdict: INSUFFICIENT (no fresh legs)."
    st.caption(caption)


def _macro_print_frame(name: str, cursor: pd.Timestamp) -> pd.DataFrame | None:
    """Real FRED observations as a (date, value) frame for ``name`` as-of cursor.

    Returns None when the series is absent or carries no printed history. Only
    real prints are kept (via :func:`_print_index`) — not the flat ffilled daily
    grid — so a chart shows one marker per monthly/quarterly release instead of
    thousands of redundant points.
    """
    series = _as_series(name, cursor)
    if series is None:
        return None
    starts = _print_index(series)
    if len(starts) == 0:
        return None
    return pd.DataFrame({"date": starts, "value": series.loc[starts].to_numpy()})


def _macro_figure(kpi: MacroKpi, frame: pd.DataFrame) -> go.Figure:
    """Small lines+markers chart of a macro series' real prints."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["value"],
            mode="lines+markers",
            name=kpi.title,
            line=dict(color="#5a6b86", width=2),
            marker=dict(color="#2e4a7d", size=4),
            hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{kpi.title}<br>"
                f"<sup>current: {kpi.value:.2f}{kpi.unit} · as of {kpi.as_of}</sup>"
            ),
            x=0.01,
            font=dict(size=12),
        ),
        xaxis_title=None,
        yaxis_title=None,
        template="plotly_white",
        hovermode="x unified",
        height=210,
        margin=dict(l=10, r=10, t=46, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


_MIN_CHART_PRINTS: int = 6


def _emit_macro_panel(col: DeltaGenerator, kpi: MacroKpi) -> None:
    """Render one kpi into a column as a trend chart (or a fallback tile)."""
    frame = _macro_print_frame(kpi.name, _macro_cursor)
    if frame is None or len(frame) < _MIN_CHART_PRINTS:
        with col:
            st.metric(
                label=f"{kpi.title}\n{('as of ' + kpi.as_of)}",
                value=(f"{kpi.value:.2f} {kpi.unit}" if pd.notna(kpi.value) else "n/a"),
                help=kpi.read,
            )
            st.caption("insufficient history to chart")
        return
    with col:
        st.plotly_chart(_macro_figure(kpi, frame), width="stretch")


def _render_macro_layer(kpis: Sequence[MacroKpi], narrative: Sequence[str]) -> None:
    """Macro/cycle backdrop: FRED history charts + a short directional readout."""
    st.subheader("Macro Cycle · FRED backdrop")
    if not kpis:
        st.caption(
            "No FRED macro series found — run scripts/fetch_macro_fred.py first."
        )
        return
    cols = st.columns(2)
    for i, kpi in enumerate(kpis):
        _emit_macro_panel(cols[i % 2], kpi)
        if i % 2 == 1:
            cols = st.columns(2)  # fresh row every two panels
    for line in narrative:
        st.markdown(f"- {line}")
    st.caption(
        "Macro series are monthly/quarterly prints (curve yields daily) shown as-of each "
        "series' own release month — not a forecast. The 2s10s spread is an historic "
        "recession lead, not a timing call."
    )


def _render_staleness_warning(snapshot: HealthSnapshot) -> None:
    """Flag stale composite legs so a fresh-looking verdict can't hide them."""
    stale: list[str] = []
    for leg in snapshot.legs:
        if (
            not leg.used
            and leg.skip_reason is not None
            and leg.skip_reason.startswith("stale")
        ):
            stale.append(
                f"- {leg.label}: last bar {leg.last_date.date()} ({leg.skip_reason})"
            )
    if stale:
        st.warning(
            "⚠️ Stale ratio legs excluded from the composite above — present on "
            "the charts below only as history:\n\n" + "\n".join(stale)
        )


_calendar_idx: pd.DatetimeIndex = _calendar_from(closes)
_health: HealthSnapshot = build_health_snapshot(closes, _calendar_idx)
_health_wave: pd.DataFrame = composite_wave(closes)
_macro_cursor: pd.Timestamp = (
    _calendar_idx.max() if len(_calendar_idx) else pd.Timestamp.today()
)
_macro_kpis: tuple[MacroKpi, ...] = build_fred_kpis(_macro_cursor)
_macro_narrative: tuple[str, ...] = build_cycle_narrative(_macro_kpis)

_render_health(_health, _health_wave)
st.markdown("---")
_render_macro_layer(_macro_kpis, _macro_narrative)
st.markdown("---")
_render_staleness_warning(_health)


def _add_band_fill(
    fig: go.Figure,
    ratio: pd.Series,
    stats: dict[str, pd.Series],
) -> None:
    """Draw the shaded ±2σ band as a self-filling region, when bands exist."""
    if stats["upper"].notna().any() and stats["lower"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=ratio.index.tolist() + ratio.index.tolist()[::-1],
                y=stats["upper"].tolist() + stats["lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_ratio_trace(
    fig: go.Figure,
    ratio: pd.Series,
    label: str,
) -> None:
    """Draw the ratio line trace (royalblue, named by the ratio ``label``)."""
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio,
            mode="lines",
            name=label,
            line=dict(color="royalblue", width=1.5),
        )
    )


def _add_mean_trace(
    fig: go.Figure,
    stats: dict[str, pd.Series],
) -> None:
    """Draw the dashed rolling-mean trace over the ratio's indices."""
    fig.add_trace(
        go.Scatter(
            x=stats["sma"].index,
            y=stats["sma"],
            mode="lines",
            name=f"SMA({rolling_window})",
            line=dict(color="darkorange", width=1.2, dash="dash"),
        )
    )


def _set_ratio_layout(
    fig: go.Figure,
    label: str,
    current: float,
    sma_val: float,
    z_score: float,
) -> None:
    """Apply the chart title, template, and axis styling for a ratio plot."""
    fig.update_layout(
        title=dict(
            text=(
                f"{label}<br>"
                f"<sup>current: {current:.4f} | z≈{z_score:+.1f} "
                f"| sma: {sma_val:.4f}</sup>"
            ),
            x=0.01,
        ),
        xaxis_title=None,
        template="plotly_white",
        hovermode="x unified",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")


def _plot_ratio(
    label: str,
    num_ticker: str,
    den_ticker: str,
    col: DeltaGenerator,
) -> None:
    """Plot a single ratio with rolling statistics into a Streamlit column."""
    num_s = closes.get(num_ticker)
    den_s = closes.get(den_ticker)

    if num_s is None or den_s is None:
        with col:
            st.caption(f"{label} — data missing")
        return

    ratio = compute_ratio_series(num_s, den_s)
    if ratio.empty or len(ratio) < 5:
        with col:
            st.caption(f"{label} — insufficient data")
        return

    stats = compute_rolling_stats(ratio, rolling_window)
    current = ratio.iloc[-1]
    sma_val = stats["sma"].iloc[-1]
    z_score = band_z(ratio, stats) or 0.0

    # Build plot
    fig = go.Figure()

    # ±2σ band (shaded), ratio line, then rolling mean
    _add_band_fill(fig, ratio, stats)
    _add_ratio_trace(fig, ratio, label)
    _add_mean_trace(fig, stats)
    _set_ratio_layout(fig, label, current, sma_val, z_score)

    with col:
        st.plotly_chart(fig, width="stretch")


# ── Layout: 2 columns per row ──────────────────────────────────

for i in range(0, len(_RATIO_DEFS), 2):
    cols = st.columns(2)
    for j, (label, num, den) in enumerate(_RATIO_DEFS[i : i + 2]):
        _plot_ratio(label, num, den, cols[j])  # type: ignore[arg-type]

# ── Summary table ──────────────────────────────────────────────

st.markdown("---")
st.subheader("Current Summary")


def _fmt_value(v: float | None) -> str:
    """Format a number to 4dp, ``N/A`` for None/NaN."""
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.4f}"


def _na_row(label: str) -> dict[str, object]:
    """Summary row where no data / insufficient history is present."""
    return {
        "Ratio": label,
        "Current": "N/A",
        f"SMA({rolling_window})": "N/A",
        "Z≈": "N/A",
        "Trend (21d)": "N/A",
    }


rows: list[dict[str, object]] = []
for label, num, den in _RATIO_DEFS:
    num_s = closes.get(num)
    den_s = closes.get(den)
    if num_s is None or den_s is None:
        rows.append(_na_row(label))
        continue
    ratio = compute_ratio_series(num_s, den_s)
    if ratio.empty or len(ratio) < 5:
        rows.append(_na_row(label))
        continue
    stats = compute_rolling_stats(ratio, rolling_window)
    current_v = ratio.iloc[-1]
    sma_v = stats["sma"].iloc[-1]
    z_v = band_z(ratio, stats)
    trend_21 = (
        (ratio.iloc[-1] / ratio.iloc[-22] - 1.0) * 100.0 if len(ratio) >= 22 else np.nan
    )
    rows.append(
        {
            "Ratio": label,
            "Current": _fmt_value(current_v),
            f"SMA({rolling_window})": _fmt_value(sma_v),
            "Z≈": "N/A" if z_v is None else f"{z_v:+.1f}",
            "Trend (21d)": "N/A" if pd.isna(trend_21) else f"{trend_21:+.2f}%",
        }
    )

summary_df = pd.DataFrame(rows)

st.dataframe(
    summary_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Ratio": st.column_config.TextColumn("Ratio"),
        "Current": st.column_config.TextColumn("Current"),
        f"SMA({rolling_window})": st.column_config.TextColumn(f"SMA({rolling_window})"),
        "Z≈": st.column_config.TextColumn("Z≈"),
        "Trend (21d)": st.column_config.TextColumn("Trend (21d)"),
    },
)

st.caption(
    f"Z≈ = (current − sma) / (σ)  where σ = band_width / 4. "
    f"Data window: last {lookback} trading days. Charts use {rolling_window}-day rolling stats."
)
