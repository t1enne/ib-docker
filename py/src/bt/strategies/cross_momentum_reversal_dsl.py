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

An optional **adaptive-entropy regime bias** (``params.ae_bias``) feeds a
``OnlineAdaptiveEntropy`` tracker on the benchmark symbol and, at each refresh,
deploys a SINGLE side selected by its raw quantised ``trend``: +1 buys the
worst-residual names, -1 shorts the top-residual names (when ``use_short``),
and 0 skips the period. This is a side *bias*, never an exposure scale. With
``ae_bias`` off (default), behavior is byte-identical to the dual-leg book
below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams
from src.indicators.adaptive_entropy import AdaptiveEntropyConfig, OnlineAdaptiveEntropy

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
    # -- market regime bias (adaptive-entropy SIDE selector) ------------------
    # A *regime bias*, not exposure sizing. When ``ae_bias`` is on, a dedicated
    # ``OnlineAdaptiveEntropy`` tracker on the benchmark symbol (feed-only,
    # never traded) reads the raw quantised AE ``trend`` of QQQ at each refresh
    # and picks which SINGLE side of the residual book is deployed that period:
    #   * AE trend == +1  -> buy the ``tail_n`` WORST-residual names (long),
    #   * AE trend == -1  -> short the ``tail_n`` TOP-residual names (short,
    #     only when ``use_short`` also permits shorting),
    #   * AE trend ==  0  -> skip (rotates any prior book shut for the period).
    # Notional per deployed side is NOT regime-scaled -- it stays the leg's own
    # ``long_share``/``short_share`` (equity-compounded as the code already
    # does). Default ``False`` => byte-identical to the ungated legacy behavior
    # (never feeds AE, never changes the dual-leg book).
    ae_bias: bool = False
    # OnlineAdaptiveEntropy hyper-parameters (only read when ``ae_bias``).
    entropy_lookback: int = 25  # AE entropy lookback (Shannon histogram + ATR)
    entropy_num_bins: int = 10  # AE log-return histogram bins


@dataclass
class _Ctx:
    """Cross-call bookkeeping for one run/window (held in ``ctx.shared``)."""

    bars_to_refresh: int = 0  # countdown; when 0 the book is due to refresh
    initialized: bool = False  # True once the first book has formed
    # AE regime-bias state (only used when ``ae_bias`` is on).
    ae: OnlineAdaptiveEntropy | None = None  # QQQ adaptive-entropy tracker
    ae_trend: int = 0  # latest raw quantised QQQ AE trend (-1/0/1); 0 if not warm


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


def _leg_share(params: Params, side: str) -> float:
    """0..1 live-equity fraction each name on a book side gets (gross/tail split).

    Combined with ``ctx.long/short(..., size_mode="equity")`` this sizes every
    name off the current MTM book, so positions compound with PnL and capital
    utilization stays high rather than drifting down off fixed seed capital.
    """
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
    long_tail = ordered[: min(params.tail_n, len(ordered))]
    for sym in long_tail:
        targets[sym] = "long"
    # tail_n BEST residual -> SHORT (hedge light) when enabled
    if params.use_short:
        short_tail = ordered[-params.tail_n :]
        for sym in short_tail:
            if sym not in targets:  # never both long and short in one book
                targets[sym] = "short"
    return targets


def _ae_side_for_trend(trend: int, params: Params) -> str | None:
    """Map the benchmark AE quantised trend to a ONE-SIDED book orientation.

    A regime *bias* (side selector), never an exposure scale:
      * +1 -> "long"  (buy worst-residual names; expect reversion up),
      * -1 -> "short" if ``use_short`` else "None" (short the winners that
        must give back in a down-tape; disallowed short exposes nothing),
      *  0 -> "None"  (indeterminate regime -> no directional book).
    Returns None also when the AE is not yet warm (trend reads 0). The caller
    treats None as "skip this period".
    """
    if trend >= 1:
        return "long"
    if trend <= -1:
        if params.use_short:
            return "short"
        return None
    return None


def _ae_targets(
    ctx: StrategyContext, params: Params, side: str | None
) -> dict[str, str]:
    """Single-side residual targets the AE regime selects for one refresh.

    ``side`` is the orientation chosen by :func:`_ae_side_for_trend`. Returns
    the ``tail_n`` members on that side ("). When ``side`` is None (skip) the
    empty dict is returned, which rotates every currently-open leg shut.
    """
    if side != "long" and side != "short":
        return {}
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
    if not scores:
        return {}
    ordered = sorted(scores, key=lambda s: scores[s])  # ascending residual
    cut = min(params.tail_n, len(ordered))
    members = ordered[:cut] if side == "long" else ordered[-cut:]
    return {sym: side for sym in members}


# ---------------------------------------------------------------------------
# adaptive-entropy regime bias (OnlineAdaptiveEntropy on the benchmark symbol)
# ---------------------------------------------------------------------------


def _feed_ae(ctx: StrategyContext, st: _Ctx, params: Params) -> None:
    """Feed QQQ into the AE tracker and cache the raw quantised ``trend`` on
    ``st.ae_trend`` (-1/0/+1).

    Fed on every bar the strategy runs (including the pre-trade warmup), so the
    AE state is warm by the time the first book forms. Reads the RAW adaptive-
    entropy ``trend`` (adaptive-EMA / ATR-band breakout) -- deliberately NOT an
    SMA-anchored two-layer predicate -- as the regime-bias source. State lives
    on ``st`` (shared per-run ``_Ctx``), so split/sweep windows never leak.
    """
    if st.ae is None:
        st.ae = OnlineAdaptiveEntropy(
            AdaptiveEntropyConfig(
                lookback=params.entropy_lookback,
                num_bins=params.entropy_num_bins,
            )
        )
    try:
        o = ctx.ohlcv(params.benchmark)
    except KeyError:
        return
    if len(o.close) == 0:
        return
    result = st.ae.observe(float(o.close[-1]), float(o.high[-1]), float(o.low[-1]))
    st.ae_trend = int(getattr(result, "trend", 0))


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    st: _Ctx = shared.setdefault(_CTX_KEY, _Ctx())

    # When ``ae_bias`` is on, feed the benchmark AE tracker this bar so
    # ``st.ae_trend`` is warm and fresh whenever a refresh reads it. Feeds run
    # even over the pre-trade warmup (before the warmup early-return below) so
    # the chosen side is grounded before the first book forms.
    if params.ae_bias:
        _feed_ae(ctx, st, params)

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

    # Regime-bias mode refreshes toward the AE-selected single side; baseline
    # mode rotates the full dual-leg book. Same cadence semantics.
    if params.ae_bias:
        _refresh_ae(ctx, st, params)
    else:
        _refresh(ctx, st, params)


def _refresh(ctx: StrategyContext, st: _Ctx, params: Params) -> None:
    """Rotate toward a fresh `targets` set (fills at next bar's open).

    Closes any currently-open leg whose side changed or rotated out of the set,
    then opens targets absent a same-side current position, sized ``size_mode=
    "equity"`` (live-equity compounding off the MTM book)._refresh resolves the
    target set from the full dual-leg ranking by default.
    """
    _apply_targets(ctx, st, params, _rank_targets(ctx, params))


def _refresh_ae(ctx: StrategyContext, st: _Ctx, params: Params) -> None:
    """Regime-bias rotation: targets = ONE side chosen by the QQQ AE trend.

    ``side`` = :func:`_ae_side_for_trend(st.ae_trend)` -- long in an AE +1
    regime, short in an AE -1 regime (only if ``use_short``), or skip
    (empty targets -> rotate the prior side shut) when AE reads 0 / not warm.
    """
    side = _ae_side_for_trend(st.ae_trend, params)
    _apply_targets(ctx, st, params, _ae_targets(ctx, params, side))


def _apply_targets(
    ctx: StrategyContext,
    st: _Ctx,
    params: Params,
    targets: dict[str, str],
) -> None:
    """Rotate the current open book to ``targets`` and reset the hold cadence."""
    current: dict[str, str] = {}
    for sym in ctx.symbols:
        if _is_benchmark(sym, params.benchmark):
            continue
        side = _side_of(_net_qty(ctx, sym))
        if side is not None:
            current[sym] = side

    # (a) close rotated members (side changed or rotated out of the book)
    for sym, side in current.items():
        if targets.get(sym) != side:
            ctx.close(sym, reason=f"[xmr] rotate out of {side} {sym}")

    # (b) open fresh targets absent a same-side current position. Sizes target
    # a ``size_mode="equity"`` live fraction of the book (compounds with PnL).
    for sym, side in targets.items():
        if current.get(sym) == side:
            continue
        size = _leg_share(params, side)
        if size <= 0:
            continue
        if side == "long":
            ctx.long(
                sym,
                size=size,
                size_mode="equity",
                reason=f"[xmr] long worst-resid {sym}",
            )
        else:
            ctx.short(
                sym,
                size=size,
                size_mode="equity",
                reason=f"[xmr] short top-resid {sym}",
            )

    # (c) reset the hold cadence and mark initialized
    st.bars_to_refresh = params.hold_days
    st.initialized = True
