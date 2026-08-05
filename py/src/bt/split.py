"""Walk-forward / single-anchor IS-OOS split validation.

Evaluates a strategy's FIXED parameter set across in-sample (IS) and
out-of-sample (OOS) windows. It does NOT re-tune params per fold — the
engine has no optimizer. Answers the honest-validation question:
"given these locked params, how does performance hold up out-of-sample?"

Pure fold math lives here (test-friendly); engine wiring is `run_split`.
Mirrors the repo's `pure.py` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import pandas as pd

from src.bt.strategies import init_strat
from src.bt.types import StrategyConfig, PortfolioResult
from src.utils import parse_timestamp

DAY = pd.offsets.BDay(1)


@dataclass(frozen=True, slots=True)
class TestFold:
    """One IS/OOS evaluation window pair.

    IS = [is_start, is_end]; OOS = [oos_start, oos_end]. OOS begins on the
    next trading day after IS ends, so the windows are disjoint and adjacent.
    """

    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def is_trading_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.is_start, self.is_end)

    def oos_trading_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.oos_start, self.oos_end)


@dataclass(frozen=True)
class FoldMetrics:
    fold: TestFold
    in_sample: PortfolioResult
    out_of_sample: PortfolioResult


@dataclass(frozen=True)
class SplitReport:
    config_name: str
    params: Mapping[str, object]
    folds: tuple[FoldMetrics, ...]

    def oos_sharpe_series(self) -> pd.Series:
        return pd.Series(
            [f.out_of_sample.sharpe_ratio for f in self.folds],
            index=[f.fold.index for f in self.folds],
            name="oos_sharpe",
        )

    def mean_oos_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return float(self.oos_sharpe_series().mean())

    def min_oos_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return float(self.oos_sharpe_series().min())

    def oos_vs_is_degradation(self) -> float:
        """mean(oos Sharpe) / mean(is Sharpe). 1.0 = no degradation."""
        if not self.folds:
            return 0.0
        is_avg = sum(f.in_sample.sharpe_ratio for f in self.folds) / len(self.folds)
        if abs(is_avg) < 1e-12:
            return 0.0
        return self.mean_oos_sharpe() / is_avg


def anchor_split(
    cfg: StrategyConfig,
    is_end: pd.Timestamp,
    *,
    train_start: pd.Timestamp | None = None,
) -> list[TestFold]:
    """Single split: IS=[trading_start, is_end], OOS=[is_end+1d, trading_end].

    Raises ValueError when is_end is not strictly before trading_end (the OOS
    window would be empty), or when is_end precedes trading_start.
    """
    start = parse_timestamp(cfg.trading_start)
    end = parse_timestamp(cfg.trading_end)
    is_end = parse_timestamp(is_end)

    if is_end <= start:
        raise ValueError(
            f"--is-end {is_end.date()} must be after trading_start {start.date()}"
        )
    if is_end >= end:
        raise ValueError(
            f"--is-end {is_end.date()} must be before trading_end {end.date()} "
            "(OOS window would be empty)"
        )

    is_start = train_start if train_start is not None else start
    return [
        TestFold(
            index=0,
            is_start=is_start,
            is_end=is_end,
            oos_start=is_end + DAY,
            oos_end=end,
        )
    ]


def walk_forward_folds(
    cfg: StrategyConfig,
    n_folds: int,
    *,
    min_is_years: float = 5.0,
    oos_length: str | pd.DateOffset = "auto",
    train_start: pd.Timestamp | None = None,
) -> list[TestFold]:
    """Expansion-window walk-forward: IS always starts at trading_start, grows.

    Produces exactly `n_folds` non-empty folds. With `oos_length="auto"` the
    OOS chunk is `span/(n_folds+1)`, so the leading IS plus `n_folds` OOS
    chunks tile `[trading_start, trading_end]` (a leading IS anchor of one
    chunk + n OOS chunks = n+1 equal slices). Fold i's IS = [is_start, is_end_i]
    (is_end_i = the boundary before OOS chunk i+1, so IS grows monotonically)
    and its OOS is the next chunk reaching the following boundary (the last
    one reaches trading_end).

    With an explicit `oos_length` DateOffset the boundaries step by that
    offset, clamped to `trading_end`; a trailing chunk too short to be
    non-empty is skipped, and fewer-than-requested folds warn (plan edge
    case 2).

    `is_start` respects the train_start warmup if given.

    Warns once when the first fold's IS is shorter than min_is_years.
    """
    start = parse_timestamp(cfg.trading_start)
    end = parse_timestamp(cfg.trading_end)
    span = end - start

    if n_folds < 1:
        raise ValueError("--folds must be >= 1")

    if oos_length == "auto":
        oos = span / (n_folds + 1)
        boundaries: list[pd.Timestamp] = [
            (start + oos * (i + 1)).normalize() for i in range(n_folds)
        ]
    else:
        assert isinstance(oos_length, pd.DateOffset), (
            "oos_length must be 'auto' or a pd.DateOffset"
        )
        cursor = start
        boundaries: list[pd.Timestamp] = []
        for _ in range(n_folds):
            cursor = (cursor + oos_length).normalize()
            if cursor >= end:
                cursor = end  # clamp — never walk past trading_end
            boundaries.append(cursor)
            if cursor == end:
                break

    is_start = train_start if train_start is not None else start
    if is_start > start:
        raise ValueError(
            f"--train-start {is_start.date()} cannot be after trading_start "
            f"{start.date()}"
        )

    first_is_len = boundaries[0] - is_start
    if first_is_len.days / 365.25 < min_is_years:
        import warnings

        warnings.warn(
            f"First fold IS is only {first_is_len.days / 365.25:.1f}y "
            f"(< min_is_years {min_is_years:g}); later folds have more history. "
            "Runs may be thin early on.",
            stacklevel=2,
        )

    folds: list[TestFold] = []
    for i, is_end in enumerate(boundaries):
        oos_start_val = is_end + DAY
        oos_end_val = boundaries[i + 1] if i + 1 < len(boundaries) else end
        if oos_end_val <= oos_start_val:
            continue  # empty OOS chunk — skip (plan edge case 2)
        folds.append(
            TestFold(
                index=len(folds),  # contiguous, even after empty-chunk skips
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start_val,
                oos_end=oos_end_val,
            )
        )
    if len(folds) < n_folds:
        import warnings

        warnings.warn(
            f"Only {len(folds)} non-empty fold(s) possible within "
            f"[{start.date()}, {end.date()}] — requested {n_folds}. "
            "Reduce --folds or shrink the OOS offset.",
            stacklevel=2,
        )
    return folds


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def _reset_strategy_state(strat_mod) -> None:
    """Reset a strategy's cross-run mutable state via its reset_global() hook.

    Convention: every strategy with runtime state holds it in one module-level
    `GLOBAL: dict` and exposes `reset_global()` which rebinds `GLOBAL` to a
    fresh dict with correct defaults. The engine never resets these, so without
    an explicit reset (re-importing won't restore the original empty dicts)
    state bleeds silently across folds — a real bug in prior sweeps. Stateless
    strategies do not need the hook; this is a no-op for them.
    """
    reset = getattr(strat_mod, "reset_global", None)
    if reset is not None:
        reset()


def _run_window(
    cfg: StrategyConfig,
    strat_mod,
    data: pd.DataFrame,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
) -> PortfolioResult:
    """Run one IS or OOS window by overriding the config's trading window.

    Data is loaded once per split, so this only slices the `can_trade` gate —
    no per-window data reload.
    """
    from src.bt.engine.backtest import Backtest, run

    window_cfg = replace(
        cfg,
        trading_start=trading_start.isoformat(),
        trading_end=trading_end.isoformat(),
    )
    bt = Backtest(window_cfg)
    _reset_strategy_state(strat_mod)
    results = run(bt, data, strat_mod=strat_mod)
    return results.pf


def run_split(
    cfg: StrategyConfig,
    folds: list[TestFold],
) -> SplitReport:
    """Run one backtest per IS and OOS window of every fold.

    - strategy_params are NEVER mutated across folds (locked params).
    - Loads candles once over [train_start, trading_end], window-sliced per
      fold via trading-window overrides (no per-fold data reload).
    - Resets module-level strategy state before EVERY window run.
    """
    from src.bt.data_feed import load_candles

    if not folds:
        raise ValueError("No folds to run — check the split windows")

    strat_mod = init_strat(cfg.strategy_type)
    load_start = min(
        parse_timestamp(cfg.training_start),
        min(f.is_start for f in folds),
    )
    data = load_candles(
        cfg.symbols,
        load_start,
        parse_timestamp(cfg.trading_end),
        cfg.bars[0],
    )

    fold_metrics: list[FoldMetrics] = []
    for fold in folds:
        is_end, is_start = fold.is_end, fold.is_start
        oos_start, oos_end = fold.oos_start, fold.oos_end
        is_result = _run_window(cfg, strat_mod, data, is_start, is_end)
        oos_result = _run_window(cfg, strat_mod, data, oos_start, oos_end)
        fold_metrics.append(
            FoldMetrics(fold=fold, in_sample=is_result, out_of_sample=oos_result)
        )

    return SplitReport(
        config_name=cfg.name,
        params=dict(cfg.strategy_params),
        folds=tuple(fold_metrics),
    )


# ---------------------------------------------------------------------------
# rendering / serialization
# ---------------------------------------------------------------------------


def _win_rate(result: PortfolioResult) -> float:
    closed = [t for t in result.trades if t.status.value == "closed"]
    if not closed:
        return 0.0
    return sum(1.0 for t in closed if t.pnl > 0) / len(closed)


def _metrics_row(
    row: tuple[PortfolioResult, PortfolioResult],
) -> tuple[str, ...]:
    cells: list[str] = []
    for r in row:
        cells.append(f"{getattr(r, 'annual_return'):.2%}")
        cells.append(f"{getattr(r, 'sharpe_ratio'):.2f}")
        cells.append(f"{getattr(r, 'max_drawdown'):.2%}")
        cells.append(f"{getattr(r, 'calmar_ratio'):.2f}")
        win = _win_rate(r)
        cells.append(f"{win:.1%}")
    return tuple(cells)


def render_split_report(report: SplitReport) -> str:
    """Render IS vs OOS table per fold + summary row."""
    from src.bt.table import Col, Table, render

    cols: tuple[Col, ...] = (Col("Fold", ">"),)
    for side in ("IS", "OOS"):
        cols += (
            Col(f"{side} Ann", ">"),
            Col(f"{side} Sharpe", ">"),
            Col(f"{side} MaxDD", ">"),
            Col(f"{side} Calmar", ">"),
            Col(f"{side} WinRate", ">"),
        )

    rows: list[tuple[str, ...]] = []
    for fm in report.folds:
        label = (
            f"{fm.fold.is_end.date()}→{fm.fold.oos_end.date()}"
            if fm.fold.index == 0
            else f"f{fm.fold.index}"
        )
        rows.append((label,) + _metrics_row((fm.in_sample, fm.out_of_sample)))

    if not report.folds:
        rows = [("(no folds)", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—")]

    table = Table(columns=cols, rows=tuple(rows))
    lines: list[str] = [f"\nSplit: {report.config_name}"]

    summary = (
        f"Mean OOS Sharpe {report.mean_oos_sharpe():.2f} · "
        f"Min OOS Sharpe {report.min_oos_sharpe():.2f} · "
        f"OOS/IS degradation {report.oos_vs_is_degradation():.2f}"
    )
    lines.append(summary)
    lines.extend(render(table))
    return "\n".join(lines)


def split_report_to_dict(report: SplitReport) -> dict:
    """Serialize a SplitReport into a plain JSON-ready dict."""
    float_fields = (
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "sortino_ratio",
    )

    def _result_dict(r: PortfolioResult) -> dict:
        d = {f: float(getattr(r, f)) for f in float_fields}
        d["win_rate"] = _win_rate(r)
        d["trade_count"] = len([t for t in r.trades if t.status.value == "closed"])
        return d

    return {
        "config": report.config_name,
        "params": {str(k): v for k, v in report.params.items()},
        "folds": [
            {
                "index": fm.fold.index,
                "is_window": fm.fold.is_trading_window()[0].date().isoformat(),
                "is_end": fm.fold.is_end.date().isoformat(),
                "oos_window": fm.fold.oos_trading_window()[0].date().isoformat(),
                "oos_end": fm.fold.oos_end.date().isoformat(),
                "is": _result_dict(fm.in_sample),
                "oos": _result_dict(fm.out_of_sample),
            }
            for fm in report.folds
        ],
        "agg": {
            "mean_oos_sharpe": report.mean_oos_sharpe(),
            "min_oos_sharpe": report.min_oos_sharpe(),
            "oos_vs_is_degradation": report.oos_vs_is_degradation(),
        },
    }


__all__ = [
    "TestFold",
    "FoldMetrics",
    "SplitReport",
    "anchor_split",
    "walk_forward_folds",
    "run_split",
    "render_split_report",
    "split_report_to_dict",
]
