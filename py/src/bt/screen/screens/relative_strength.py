"""Relative-strength screen — rank a universe vs a benchmark (SPY or QQQ).

Scores every symbol by its trailing ``lookback``-bar return **relative to a
benchmark** (default SPY). This is the classic O'Neil/IBD "Relative Strength"
idea, done cross-sectionally so the output is a *ranking*, not an absolute
signal: a symbol gets a high score when it has outperformed its peers against
the same market reference.

For each symbol we compute:

  ret(symbol)       = close[-1] / close[-(lookback+1)] - 1   (trailing return)
  excess_ret        = ret(symbol) - ret(benchmark)           (outperformance)
  rs_ratio          = close(symbol) / close(benchmark) * 100 (RS line level)

``score`` is the **percentile rank** of ``excess_ret`` (or the symbol's own
``return`` in ``absolute`` mode) across the universe on the latest bar
(``rank / (n-1)``), so it is naturally in 0..1: the strongest outperformers
score near 1.0, the worst laggards near 0.0. ``action`` is ``long`` when the
symbol sits in the top ``top_pct`` of the ranking, ``short`` when it sits in
the bottom ``top_pct`` (a weak/relative-laggard inverse play), else ``flat``.

Scoring knobs (``Params``): ``benchmark``, ``lookback``, ``mode``, ``top_pct``,
``warmup_bars``. Direction is decided purely by relative rank; absolute market
direction is surfaced as diagnostics only (an RS screen ranks relative
outperformers — it deliberately does not time the market).

The benchmark symbol must be present in ``ScreenState.frames`` (pass SPY/QQQ in
the universe or ``--symbols``); we raise a clear error otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, cast

import pandas as pd

from src.bt.screen.metrics import with_common_metrics
from src.bt.screen.types import Action, ScreenParams, ScreenResult, ScreenState

SCREEN_TYPE = "rs"

#: ``excess`` ranks symbols by how much they beat the benchmark; ``absolute``
#: ranks by the symbol's own trailing return (benchmark held out for context).
RankMode = Literal["excess", "absolute"]


@dataclass(frozen=True)
class Params(ScreenParams):
    #: Benchmark symbol (SPY or QQQ). Must carry a frame in ``ScreenState.frames``.
    benchmark: str = "SPY"
    #: Trailing bars over which return/RS is measured (63 ~ one quarter of dailies).
    lookback: int = 63
    #: Which quantity the cross-sectional percentile ranking uses.
    mode: RankMode = "excess"
    #: Nominal top/bottom fraction of the ranking that gets long/short actions.
    top_pct: float = 0.10
    #: Warmup (bars of history required, benchmark included).
    warmup_bars: int = 63


# ---------------------------------------------------------------------------
# pure helpers (vectorized; no module state)
# ---------------------------------------------------------------------------


def _lookback_return(closes: pd.Series, lookback: int) -> float | None:
    """Trailing ``lookback``-bar simple return as a fraction. None on shortage."""
    if closes is None or len(closes) < lookback + 1:
        return None
    past = closes.iloc[-(lookback + 1)]
    current = closes.iloc[-1]
    if past == 0 or not math.isfinite(past) or not math.isfinite(current):
        return None
    return float((current - past) / past)


def _rs_ratio(
    sym_closes: pd.Series | None, bench_closes: pd.Series | None
) -> float | None:
    """RS line level: symbol close / benchmark close * 100 (None if missing)."""
    if sym_closes is None or bench_closes is None:
        return None
    if sym_closes.empty or bench_closes.empty:
        return None
    sym = sym_closes.iloc[-1]
    bench = bench_closes.iloc[-1]
    if bench == 0 or not math.isfinite(sym) or not math.isfinite(bench):
        return None
    return float(sym / bench * 100.0)


def _percentile(rank: int, n: int) -> float:
    """Map a 0-based rank in an ``n``-element ranking to a 0..1 percentile.

    The best performer (rank ``n-1``) gets 1.0, the worst (rank 0) gets 0.0.
    A single ranked symbol is always 1.0 if positive, else 0.0.
    """
    if n <= 1:
        return 1.0 if rank > 0 else 0.0
    return float(min(1.0, max(0.0, rank / (n - 1))))


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------


def on_state(state: ScreenState, params: Params) -> tuple[ScreenResult, ...]:
    """Score every symbol's latest bar by relative strength vs the benchmark."""
    # Benchmark frame must be present for an RS ranking to make sense.
    bench_df = state.frame(params.benchmark.upper())
    bench_closes = (
        cast(pd.Series, bench_df["close"])
        if bench_df is not None and not bench_df.empty and "close" in bench_df.columns
        else None
    )
    if bench_closes is None:
        available = ", ".join(sorted(s for s, _ in state.frames)) or "(none)"
        raise ValueError(
            f"relative_strength benchmark {params.benchmark!r} not in frames. "
            f"Pass it in --symbols/--universe. Available: {available}"
        )

    # First pass: per-symbol trailing return, excess vs benchmark, RS line.
    metrics: dict[str, dict[str, float]] = {}
    for symbol, df in state.frames:
        if symbol == params.benchmark.upper():
            # The benchmark itself is the reference, never a directional play.
            continue
        closes = cast(pd.Series, df["close"]) if "close" in df.columns else None
        if closes is None or len(closes) < params.warmup_bars:
            metrics[symbol] = {
                "return": 0.0,
                "excess_ret": 0.0,
                "rs_ratio": _rs_ratio(closes, bench_closes) or 0.0,
                "rs_pct": -1.0,  # sentinel: not ranked (insufficient data)
            }
            continue

        sym_ret = _lookback_return(closes, params.lookback)
        bench_ret = _lookback_return(bench_closes, params.lookback)
        if sym_ret is None or bench_ret is None:
            metrics[symbol] = {
                "return": sym_ret or 0.0,
                "excess_ret": (sym_ret or 0.0) - (bench_ret or 0.0),
                "rs_ratio": _rs_ratio(closes, bench_closes) or 0.0,
                "rs_pct": -1.0,
            }
            continue

        metrics[symbol] = {
            "return": sym_ret,
            "excess_ret": sym_ret - bench_ret,
            "rs_ratio": _rs_ratio(closes, bench_closes) or 0.0,
            "rs_pct": 0.0,  # filled below once the rank is known
        }

    # Second pass: cross-sectional percentile ranking of the chosen quantity.
    # Only symbols that survived warmup (non-sentinel) enter the ranking;
    # short-history symbols keep their -1.0 sentinel and resolve to flat.
    ranked = sorted(
        (s for s, m in metrics.items() if m["rs_pct"] >= 0.0),
        key=lambda s: (
            metrics[s]["excess_ret"]
            if params.mode == "excess"
            else metrics[s]["return"]
        ),
    )
    n = len(ranked)
    for i, symbol in enumerate(ranked):
        metrics[symbol]["rs_pct"] = _percentile(i, n)

    # Emit one result per symbol (the benchmark itself is excluded).
    results: list[ScreenResult] = []
    for symbol, df in state.frames:
        if symbol == params.benchmark.upper():
            continue
        m = metrics[symbol]
        pct = m["rs_pct"]
        score = pct if pct >= 0.0 else 0.0
        action: Action = "flat"
        signals: list[str]

        if pct >= 0.0:
            if pct >= 1.0 - params.top_pct:
                action = "long"
            elif pct < params.top_pct:
                action = "short"
            base = m["excess_ret"] if params.mode == "excess" else m["return"]
            label = f"RS-{pct:.0%}"
            direction = "↑" if base >= 0 else "↓"
            signals = [
                f"bench {params.benchmark}",
                label,
                f"{direction} {abs(base):.1%}",
            ]
            if action != "flat":
                signals.append(f"top{int(params.top_pct * 100)}%")
        else:
            signals = ["WARMUP"]

        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=score,
                action=action,
                signals=tuple(signals),
                model_features=with_common_metrics(
                    {
                        "return": m["return"],
                        "excess_ret": m["excess_ret"],
                        "rs_ratio": m["rs_ratio"],
                        "rs_pct": pct if pct >= 0.0 else 0.0,
                    },
                    df,
                ),
            )
        )
    return tuple(results)
