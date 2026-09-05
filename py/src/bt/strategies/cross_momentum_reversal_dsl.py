"""Cross-sectional short-lookback mean reversion (DSL, daily).

Built from a cross-sectional research scan over the nsdq universe (88 names,
daily, beta removed via rolling-63d OLS vs the single benchmark ``QQQ``). The
scan found sign-stable SHORT-HORIZON MEAN REVERSION:

    k=5d h=5d   spearman -0.017 t_eff -3.68
    k=21d h=21d spearman -0.118 t_eff -9.01 net -483bps (residual)

CLAIM TESTED: "Recent-losers (past ~21d, beta-removed) mean-revert up over the
next month cross-sectionally. Buy worst-residual decile, short/hold-light best
decile. Strongest in down/flat benchmark regimes, positive but weakest in up."

Each panel day (once per timestamp, fired on the benchmark symbol) this
strategy:

  * computes a *residualized* lookback return per panel member =
    ``raw_ret(sym) - beta * bench_lookback_ret``, where ``raw_ret`` is the
    ``lookback``-day log-return and ``beta`` is the trailing ``ols_window``-day
    OLS slope of the member's daily log-returns on the benchmark's (de-beta-ing
    is ESSENTIAL: in the scan the raw spread was -112bps vs -483bps net, so the
    raw single-name signal is beta-masked);
  * ranks members cross-sectionally by that residual;
  * rebalances a ~monthly hold: ``tail_n`` WORST-decile residual names long; and,
    when ``use_short``, ``tail_n`` BEST-decile residual names short;
  * holds the book ``hold_days`` bars, then refreshes. Fills occur at the next
    bar's open, so there is no look-ahead between a signal close and its fill.

All cross-call bookkeeping (refresh cadence, the open-book legs) lives in
``ctx.shared`` via ``@strategy(stateful=True)``; pure helpers operate only on
cursor-truncated OHLCV views (never future rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "cross_momentum_reversal"

# ctx.shared key (module-private; minted fresh per run/window).
_CTX_KEY = "cross_mr_ctx"


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- reversal lookback / hold ---------------------------------------------
    lookback: int = 21  # trailing daily bars over which the raw move is measured
    hold_days: int = 21  # bars between refresh (rebalance cadence)
    # -- de-beta (OLS vs benchmark) -------------------------------------------
    ols_window: int = 63  # daily bars of the beta regression
    benchmark: str = "QQQ"  # index whose history de-betas each member
    # -- portfolio legs -------------------------------------------------------
    tail_n: int = 9  # members per side (worst long, best short when enabled)
    long_share: float = 0.5  # gross equity fraction on the long leg
    short_share: float = 0.5  # gross equity fraction shorted
    use_short: bool = True  # short the top-residual decile?
    # -- gating ---------------------------------------------------------------
    warmup_bars: int = 84  # visible daily bars (63 OLS + 21 lookback) before first book
    min_total_daily_history: int = 130  # member needs >= this many daily bars to rank


@dataclass
class _Ctx:
    """Cross-call bookkeeping for one run/window (held in ``ctx.shared``)."""

    bars_to_refresh: int = 0  # countdown; when 0 the book is due to refresh
    initialized: bool = False  # True once the first book has formed


# ---------------------------------------------------------------------------
# pure helpers (typed; operate only on cursor-truncated OHLCV views)
# ---------------------------------------------------------------------------


def _closes(ctx: StrategyContext, sym: str) -> np.ndarray | None:
    """Visible (cursor-truncated) daily close array for ``sym``, else None."""
    try:
        o = ctx.ohlcv(sym)
    except KeyError:
        return None
    return o.close.to_array()


def _log_returns(closes: np.ndarray) -> np.ndarray:
    """Daily log-returns of a close array (length n-1; NaN-safe)."""
    if closes.size < 2:
        return np.array([], dtype=np.float64)
    positive = np.where(closes > 0, closes, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(positive))
    return np.asarray(np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0))


def _tail_log_ret(closes: np.ndarray, window: int) -> float:
    """``ln(c[-1] / c[-1-window])`` from the visible tail, NaN if too short."""
    if closes.size <= window:
        return float("nan")
    a, b = closes[-1], closes[-(window + 1)]
    if a <= 0 or b <= 0 or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(np.log(a / b))


def _tail_beta(member_lr: np.ndarray, bench_lr: np.ndarray, window: int) -> float:
    """OLS slope of trailing ``window`` member returns on the benchmark's.

    Member and benchmark share the business-day cadence over the tail, so the
    tail return alignment is date-aligned. NaN when degenerate (constant bench).
    """
    w = min(window, member_lr.size, bench_lr.size)
    if w < 2:
        return float("nan")
    xm = bench_lr[-w:] - bench_lr[-w:].mean()
    ym = member_lr[-w:] - member_lr[-w:].mean()
    denom = float(np.dot(xm, xm))
    if denom <= 1e-16:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


def _resid_return(
    member: np.ndarray, bench: np.ndarray, lookback: int, ols_window: int
) -> float:
    """Residualized lookback return: ``raw_member - beta * bench_lookback_move``."""
    raw = _tail_log_ret(member, lookback)
    bench_move = _tail_log_ret(bench, lookback)
    if not (np.isfinite(raw) and np.isfinite(bench_move)):
        return float("nan")
    member_lr = _log_returns(member)
    bench_lr = _log_returns(bench)
    beta = _tail_beta(member_lr, bench_lr, ols_window)
    if not np.isfinite(beta):
        return float("nan")
    return raw - beta * bench_move


def _is_benchmark(sym: str, bench: str) -> bool:
    """Case-insensitive benchmark/observer check (benchmarks are not traded)."""
    return sym.lower() == bench.lower()


def _net_qty(ctx: StrategyContext, sym: str) -> float:
    """Signed net quantity for ``sym`` across all lots (0 when flat)."""
    return float(ctx.quantity(sym))


def _side_of(qty: float) -> str | None:
    """Canonical side string for a signed net quantity (None when flat)."""
    if qty > 1e-6:
        return "long"
    if qty < -1e-6:
        return "short"
    return None


def _leg_size(params: Params, side: str) -> float:
    """Per-name 0..1 capital fraction for one book side (gross split over tail)."""
    share = params.long_share if side == "long" else params.short_share
    denom = max(1, params.tail_n)
    return float(max(0.0, min(share / denom, 1.0)))


def _rank_targets(ctx: StrategyContext, params: Params) -> dict[str, str]:
    """Map ``sym -> desired side`` from the cross-sectional residual ranking.

    Members that lack enough daily history, or whose residual is not finite,
    drop out of the ranking (never traded this refresh).
    """
    bench_c = _closes(ctx, params.benchmark)
    scores: dict[str, float] = {}
    for sym in ctx.symbols:
        if _is_benchmark(sym, params.benchmark):
            continue
        m = _closes(ctx, sym)
        if m is None or bench_c is None:
            continue
        if m.size < params.min_total_daily_history:
            continue
        r = _resid_return(m, bench_c, params.lookback, params.ols_window)
        if np.isfinite(r):
            scores[sym] = r

    ordered = sorted(scores, key=lambda s: scores[s])  # ascending residual
    targets: dict[str, str] = {}
    # tail_n WORST residual -> LONG (expect reversion)
    for sym in ordered[: min(params.tail_n, len(ordered))]:
        targets[sym] = "long"
    # tail_n BEST residual -> SHORT (hedge light) when enabled
    if params.use_short:
        for sym in ordered[-params.tail_n :]:
            if sym not in targets:  # never both long and short in one book
                targets[sym] = "short"
    return targets


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    st: _Ctx = shared.setdefault(_CTX_KEY, _Ctx())

    # Benchmark must be visible for the de-beta and warmup windows.
    bench = _closes(ctx, params.benchmark)
    if bench is None:
        return
    if bench.size < params.warmup_bars:
        return

    # Count down the hold window. The first book forms immediately on the first
    # warm day; after that we only refresh every ``hold_days`` trading bars.
    if st.initialized and st.bars_to_refresh > 0:
        st.bars_to_refresh -= 1
    # Refresh when the first book is due, or the hold countdown has elapsed.
    if st.initialized and st.bars_to_refresh > 0:
        return

    _refresh(ctx, st, params)


def _refresh(ctx: StrategyContext, st: _Ctx, params: Params) -> None:
    """Rebalance the whole cross-sectional book from the current residual ranking.

    Closes prior open legs whose side rotated out / flipped, then opens fresh
    worst/best-decile targets. All fills occur at the next bar's open.
    """
    current: dict[str, str] = {}
    for sym in ctx.symbols:
        if _is_benchmark(sym, params.benchmark):
            continue
        side = _side_of(_net_qty(ctx, sym))
        if side is not None:
            current[sym] = side

    targets = _rank_targets(ctx, params)

    # (a) close rotated members (side changed or rotated out of the book)
    for sym, side in current.items():
        if targets.get(sym) != side:
            ctx.close(sym, reason=f"[xmr] rotate out of {side} {sym}")

    # (b) open fresh targets absent a same-side current position
    for sym, side in targets.items():
        if current.get(sym) == side:
            continue
        size = _leg_size(params, side)
        if size <= 0:
            continue
        if side == "long":
            ctx.long(sym, size=size, reason=f"[xmr] long worst-resid {sym}")
        else:
            ctx.short(sym, size=size, reason=f"[xmr] short top-resid {sym}")

    # (c) reset the hold cadence and mark initialized
    st.bars_to_refresh = params.hold_days
    st.initialized = True
