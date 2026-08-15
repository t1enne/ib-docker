"""MACD + MFI divergence strategy — signal-then-confirm (DSL).

Backtests the two existing *screening* layers — ``macd_divergence`` and
``mfi_divergence`` (``src/bt/screen/screens/``) — as a tradeable engine
strategy. It reuses each screen's exact pure divergence detector verbatim (the
same fractal swing pivots, the same price-vs-oscillator divergence test, the
same direction encoding), so the backtest can never disagree with the manual
screen about whether a divergence fired.

Momentum axes (both imported straight from the screens):

  * **MACD** — price vs. the MACD *histogram*. Bullish: lower price low while
    the histogram prints a higher low below ``bull_hist_floor`` (weak histogram
    at the price low). Bearish: higher price high while the histogram prints a
    lower high above ``bear_hist_ceiling``.
  * **MFI** — price vs. Money Flow Index (RSI weighted by volume). Bullish:
    lower price low while MFI prints a higher low below ``mfi_floor``.
    Bearish: higher price high while MFI prints a lower high above
    ``mfi_ceiling``.

Signal-then-confirm (the ``rsi_divergence_dsl`` pattern): a fresh divergence is
*stored*, not entered — a bullish divergence prints a price low that sits below
any EMA by construction, so demanding ``close > EMA`` on the same bar is
impossible. On a later bar within ``max_confirm_latency`` that confirms (close
back on the right side of the fast EMA, oscillator not yet over-extended,
volume participation, optional slow-EMA trend gate) the position is entered with
an ATR-risk-sized stop. Stale, unconfirmed divergences are dropped.

Two screens, one position policy (per symbol, set by ``entry_mode``):

  * ``"either"`` (default) — trade whichever of the two screens fires a fresh
    divergence. Gives both screens independent opportunity (highest trade count).
  * ``"both"`` — require *both* the MACD and MFI screens to print a divergence
    on the same symbol, agreeing on direction, before entering (highest
    conviction, far fewer trades). This is the cross-confirmation filter that
    uses the two *different* momentum axes (MACD histogram = trend-break
    sensitivity, MFI = volume-backed money flow) as mutual confirmation.

Direction is gated by ``direction`` (``"long"`` / ``"short"`` / ``"both"``) and
optional slow-EMA trend filters per side. Sizing is ATR-risk-sized
(``stop_atr_mult * ATR`` from the swing extreme), expressed as a 0..1 ``size``
fraction — the deployment style shared with the bear leg.

All cross-candle state (cooldowns, pending signals, open-position bookkeeping)
lives in ``ctx.shared`` via ``@strategy(stateful=True)``; pure helpers take
cursor-truncated arrays so there is no lookahead, and the signal/stator state
never bleeds across split/sweep windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from src.bt.size.pure import equity_of, risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

# Reuse the screens' exact pure detectors so screen and strategy never diverge.
from src.bt.screen.screens.macd_divergence import (
    _bearish_divergence as _macd_bear,
    _bullish_divergence as _macd_bull,
    _ta_histogram,
)
from src.bt.screen.screens.mfi_divergence import (
    _bearish_divergence as _mfi_bear,
    _bullish_divergence as _mfi_bull,
    _ta_mfi,
)

STRATEGY_TYPE = "macd_mfi_divergence_dsl"

#: ATR period for stop/target risk (matches ta default).
_ATR_PERIOD = 14
#: Side-specific oscillator over-extension just before entry (cap on momentum
#: so we are not chasing a vertical rip / cascade).
_LONG_MAX_MFI = 60.0
_SHORT_MIN_MFI = 40.0
#: Fraction of the open lot shed at the ATR take-profit target.
_PARTIAL_QTY = 0.5

EntryMode = Literal["either", "both"]
Direction = Literal["long", "short", "both"]


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- which screens + how they combine --
    entry_mode: EntryMode = "either"
    direction: Direction = "both"
    # -- MACD screen knobs (mirror macd_divergence.Params) --
    fast: int = 12
    slow: int = 26
    signal: int = 9
    macd_bull_hist_floor: float = 0.0
    macd_bear_hist_ceiling: float = 0.0
    # -- MFI screen knobs (mirror mfi_divergence.Params) --
    mfi_period: int = 14
    mfi_floor: float = 35.0
    mfi_ceiling: float = 65.0
    # -- shared divergence pivots / warmup --
    pivot_lookback: int = 5
    min_lookback: int = 100
    # -- confirmation --
    fast_ma: int = 20
    slow_ma: int = 63
    confirm_vol_window: int = 5
    max_confirm_latency: int = 5
    long_trend_gate: bool = False
    short_trend_gate: bool = True
    # -- SPY regime gate (signed Kaufman efficiency ratio on the market) --
    # A divergence entry is only allowed when the market regime matches its
    # direction. Read from ``regime_symbol`` (default SPY) so every symbol is
    # gated by the same breadth/equity regime, not its own series. ``regime_er``
    # is the min |signed ER| to permit an entry; ``<= 0`` disables the gate.
    regime_symbol: str = "SPY"
    regime_lookback: int = 20
    regime_er: float = 0.0
    # -- risk / sizing (ATR-risk, 0..1 size) --
    stop_atr_mult: float = 2.0
    risk_pct: float = 0.02
    take_profit_atr: float = 3.0
    cooldown_bars: int = 10


# ---------------------------------------------------------------------------
# pure helpers (typed; operate on cursor-truncated arrays only)
# ---------------------------------------------------------------------------


def _last(v: object) -> float:
    """Last visible value of a SeriesView, or a bare float."""
    if isinstance(v, SeriesView):
        return float(v[-1])
    return float(cast(float, v))


def trend_fast_above_slow(fast: float, slow: float) -> bool:
    """Fast EMA above slow EMA (both finite)."""
    return bool(np.isfinite(fast) and np.isfinite(slow) and fast > slow)


def _efficiency_ratio(close: SeriesView, lookback: int) -> float:
    """Signed Kaufman efficiency ratio over the trailing ``lookback + 1`` closes.

    ``sign(net) * (|net| / sum_of_abs_step_changes)`` in [-1, 1]: positive =
    upward trend efficiency, negative = downward, ~0 = chop. NaN when
    insufficient data. Negative-index reads count back from the cursor tail so
    the value stays correct under split/sweep truncation (mirrors
    ``vp_breakout_dsl._efficiency_ratio``).
    """
    n = len(close)
    if n < lookback + 2:
        return float("nan")
    closes = [float(close[-1 - i]) for i in range(lookback + 1)]
    net = closes[0] - closes[-1]
    path = sum(abs(closes[i] - closes[i + 1]) for i in range(lookback))
    if path <= 1e-12:
        return 0.0
    return net / path


def _spy_regime_ok(ctx: StrategyContext, side: str, params: Params) -> bool:
    """SPY regime gate; disabled when ``regime_er <= 0``.

    Long entries require SPY's signed ER ``>= +regime_er``; short entries
    require ``<= -regime_er``. Supresses long-fade divergences in a confirmed
    bear market and short-fade divergences in a confirmed bull. Returns True
    when the gate is disabled (``regime_er <= 0``) or SPY data is unavailable
    (defensive: never silently block on a missing benchmark).
    """
    if params.regime_er <= 0.0:
        return True
    spy_close = ctx.ohlcv(params.regime_symbol).close
    er = _efficiency_ratio(spy_close, params.regime_lookback)
    if math.isnan(er):
        return False
    if side == "long":
        return er >= params.regime_er
    return er <= -params.regime_er


def volume_participation(vol: np.ndarray, window: int) -> bool:
    """Current bar volume participates vs. the trailing window (incl. current)."""
    n = len(vol)
    if n < 2:
        return False
    cur = float(vol[-1])
    win = vol[max(0, n - window) :]
    mean = float(win.mean())
    return bool(mean > 0 and cur >= mean and cur >= float(vol[-2]))


def stop_distance(entry: float, swing: float, atr: float, mult: float) -> float:
    """Risk distance from entry to the stop, floored at one ATR."""
    stop_price = swing - mult * atr
    if stop_price <= 0:
        return float("inf")
    return max(atr, abs(entry - stop_price))


def entry_size(qty: float, price: float, capital: float) -> float:
    """Back-solve a 0..1 size for a risk-sized absolute qty (may be 0)."""
    if price <= 0 or capital <= 0:
        return 0.0
    return min(1.0, max(0.0, qty * price / capital))


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    shared = ctx.shared
    shared.setdefault("cooldowns", {})
    shared.setdefault("positions", {})
    shared.setdefault("pending", {})
    params: Params = ctx.params

    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        if len(o.close) < params.min_lookback:
            continue

        if ctx.position(sym) is not None:
            shared["pending"].pop(sym, None)
            _manage_open(ctx, shared, params, sym, o)
            continue

        cd = shared["cooldowns"].get(sym, 0)
        if cd > 0:
            shared["pending"].pop(sym, None)
            shared["cooldowns"][sym] = cd - 1
            continue

        if sym in shared["pending"]:
            if _try_confirm(ctx, shared, params, sym, o):
                shared["pending"].pop(sym, None)
            else:
                _age_pending(shared, params, sym)
            continue

        _try_store_signal(ctx, shared, params, sym, o)


# ── setup detection (the two screens) ────────────────────────────────────


def _divergence_setups(
    ctx: StrategyContext, params: Params, sym: str, o
) -> tuple[float, float, list[str], list[str]]:
    """Run both screens' exact detectors on the cursor-truncated bars.

    Returns ``(bull_pivot, bear_pivot, bullish_sources, bearish_sources)``.
    A side's ``*_pivot`` is the *price* swing extreme (low for bullish,
    high
    for bearish) at the most recent fractal pivot where that divergence fired
    — ``nan`` when the side did not fire. ``*_sources`` are the subset of
    ``{"macd", "mfi"}`` that printed that divergence at the latest bar. Price
    series are pulled from OHLCV arrays (same data the screens read from
    ``state.frame``), so results match ``screen_over_history`` bar-for-bar and
    the stored stop pivot equals the screen's flagged swing extreme.
    """
    lows = o.low.to_array()
    highs = o.high.to_array()
    closes = o.close.to_array()

    bull_pivot = float("nan")
    bear_pivot = float("nan")
    bull: list[str] = []
    bear: list[str] = []

    hist = _ta_histogram(_as_series(closes), params.fast, params.slow, params.signal)
    macd_bull = _macd_bull(
        lows, hist, params.pivot_lookback, params.macd_bull_hist_floor
    )
    macd_bear = _macd_bear(
        highs, hist, params.pivot_lookback, params.macd_bear_hist_ceiling
    )
    if macd_bull is not None:
        bull.append("macd")
        bull_pivot = float(lows[macd_bull[0]])
    if macd_bear is not None:
        bear.append("macd")
        bear_pivot = float(highs[macd_bear[0]])

    mfi_arr = _ta_mfi(_frame(o), params.mfi_period)
    if len(mfi_arr) == len(closes):
        mfi_bull = _mfi_bull(lows, mfi_arr, params.pivot_lookback, params.mfi_floor)
        mfi_bear = _mfi_bear(highs, mfi_arr, params.pivot_lookback, params.mfi_ceiling)
        if mfi_bull is not None:
            bull.append("mfi")
            bull_pivot = min(bull_pivot, float(lows[mfi_bull[0]]))
        if mfi_bear is not None:
            bear.append("mfi")
            bear_pivot = max(bear_pivot, float(highs[mfi_bear[0]]))

    return bull_pivot, bear_pivot, bull, bear


def _as_series(arr: np.ndarray):
    """A zero-indexed pandas Series for the ta helpers (pivot index == array index)."""
    import pandas as pd

    return pd.Series(arr, index=range(len(arr)))


def _frame(o):
    """An OHLCV frame with the cursor-truncated arrays (for the MFI helper)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "open": o.open.to_array(),
            "high": o.high.to_array(),
            "low": o.low.to_array(),
            "close": o.close.to_array(),
            "volume": o.volume.to_array(),
        }
    )


def _try_store_signal(ctx, shared, params, sym, o) -> None:
    bull_pivot, bear_pivot, bull, bear = _divergence_setups(ctx, params, sym, o)

    if params.entry_mode == "both":
        # Require both screens on the same side before a setup is credible:
        # MFI (volume-backed money flow) confirms the MACD histogram break.
        has_bull = set(bull) >= {"macd", "mfi"}
        has_bear = set(bear) >= {"macd", "mfi"}
    else:
        has_bull = bool(bull)
        has_bear = bool(bear)

    direction: Direction = params.direction
    if has_bull and direction in ("long", "both"):
        _store(ctx, shared, params, sym, "long", bull, bull_pivot)
    elif has_bear and direction in ("short", "both"):
        _store(ctx, shared, params, sym, "short", bear, bear_pivot)


def _store(
    ctx, shared, params, sym, side: str, sources: list[str], pivot: float
) -> None:
    shared["pending"][sym] = {
        "side": side,
        "sources": tuple(sources),
        "pivot": pivot,
        "age": 0,
    }


def _age_pending(shared, params, sym) -> None:
    p = shared["pending"][sym]
    p["age"] += 1
    if p["age"] > params.max_confirm_latency:
        del shared["pending"][sym]


# ── confirmation / entry ──────────────────────────────────────────────────


def _try_confirm(ctx, shared, params, sym, o) -> bool:
    pending = shared["pending"][sym]
    side = pending["side"]
    close = float(o.close[-1])
    atr = _last(ctx.ta.atr(sym, _ATR_PERIOD))
    if not np.isfinite(atr) or atr <= 0:
        return False

    fast = _last(ctx.ta.ema(sym, params.fast_ma))
    slow = _last(ctx.ta.ema(sym, params.slow_ma))
    if not np.isfinite(fast):
        return False

    # (a) close back on the correct side of the fast EMA (structural for a
    # divergence: the swing low/high sits across the EMA by definition).
    if side == "long" and not (close > fast):
        return False
    if side == "short" and not (close < fast):
        return False

    # (b) optional slow-EMA trend gate (default: short requires bearish stack).
    if (
        side == "long"
        and params.long_trend_gate
        and not trend_fast_above_slow(fast, slow)
    ):
        return False
    if (
        side == "short"
        and params.short_trend_gate
        and trend_fast_above_slow(fast, slow)
    ):
        return False

    # (b2) SPY regime gate (signed efficiency ratio on the market benchmark).
    # A divergence reversal needs the *market* regime to agree with its side —
    # not just the symbol's own stack. Disabled when ``regime_er <= 0``.
    if not _spy_regime_ok(ctx, side, params):
        return False

    # (c) oscillator / volume participation on the confirmation bar.
    mfi_arr = _ta_mfi(_frame(o), params.mfi_period)
    mfi_now = float(mfi_arr[-1]) if len(mfi_arr) == len(o.close) else float("nan")
    if side == "long":
        if np.isfinite(mfi_now) and mfi_now > _LONG_MAX_MFI:
            return False
    else:
        if np.isfinite(mfi_now) and mfi_now < _SHORT_MIN_MFI:
            return False
    if not volume_participation(o.volume.to_array(), params.confirm_vol_window):
        return False

    return _emit_entry(
        ctx, shared, params, sym, close, atr, side, float(pending["pivot"])
    )


def _emit_entry(ctx, shared, params, sym, close, atr, side, pivot) -> bool:
    stop_dist = stop_distance(close, pivot, atr, params.stop_atr_mult)
    if not np.isfinite(stop_dist) or stop_dist <= 0:
        return False
    qty = risk_sized_qty(
        equity=equity_of(ctx.state.portfolio),
        price=close,
        stop_dist=stop_dist,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return False
    size = entry_size(qty, close, ctx.state.portfolio.initial_capital)
    if size <= 0:
        return False

    sl_fraction = stop_dist / close
    tp_fraction = (params.take_profit_atr * atr) / close
    if side == "long":
        ctx.long(
            sym,
            size=size,
            sl=sl_fraction,
            tp=tp_fraction,
            tag="macro-mfi",
            reason="[macd-mfi] long divergence confirmed",
        )
    else:
        ctx.short(
            sym,
            size=size,
            sl=sl_fraction,
            tp=tp_fraction,
            tag="macro-mfi",
            reason="[macd-mfi] short divergence confirmed",
        )
    shared["positions"][sym] = {"entry": close, "pivot": pivot, "side": side, "pid": ""}
    return True


# ── position management ──────────────────────────────────────────────────


def _manage_open(ctx, shared, params, sym, o) -> None:
    rec = shared["positions"].get(sym)
    if rec is not None and not rec.get("pid"):
        p = ctx.position(sym)
        if p is not None:
            rec["pid"] = p.position_id

    close = float(o.close[-1])
    side = rec.get("side") if rec else None
    if rec is not None and side is not None:
        if side == "long" and close <= rec["pivot"]:
            _exit(ctx, shared, sym, "swing low invalidation")
            return
        if side == "short" and close >= rec["pivot"]:
            _exit(ctx, shared, sym, "swing high invalidation")
            return

    atr = _last(ctx.ta.atr(sym, _ATR_PERIOD))
    if rec is not None and np.isfinite(atr):
        target = rec["entry"] + (
            params.take_profit_atr * atr
            if side == "long"
            else -params.take_profit_atr * atr
        )
        if (side == "long" and close >= target) or (
            side == "short" and close <= target
        ):
            ctx.partial_close(
                sym,
                qty=_PARTIAL_QTY,
                lot=rec.get("pid") or "",
                reason="[macd-mfi] partial @ target",
            )
            return


def _exit(ctx, shared, sym, reason) -> None:
    shared["positions"].pop(sym, None)
    shared["cooldowns"][sym] = ctx.params.cooldown_bars
    ctx.close(sym, reason=f"[macd-mfi] {reason}")
