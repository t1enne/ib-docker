"""Hyperparameter sweep for a strategy's params.

Runs the full cartesian product of a param grid, each combination as its own
backtest, and ranks results by a chosen metric. Params that collide with
top-level ``StrategyConfig`` fields (e.g. ``stop_loss``, ``take_profit``,
``position_size``) override the config directly; everything else overrides
``strategy_params`` (the strategy's own typed params).

Pure grid/ranking logic lives here (test-friendly); engine wiring is
``run_sweep``. Mirrors the repo's ``pure.py`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from typing import Any, Callable

from src.bt.types import StrategyConfig, PortfolioResult

# Top-level StrategyConfig fields a sweep may override directly; all other
# keys override the strategy's own params via strategy_params.
_CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in fields(StrategyConfig))


@dataclass(frozen=True)
class SweepResult:
    """One grid combination and its backtest result."""

    overrides: dict[str, Any]
    params: dict[str, Any]  # merged strategy_params (for reporting)
    pf: PortfolioResult

    @property
    def param_values(self) -> dict[str, Any]:
        """Flat view of all swept params actually applied (overrides only)."""
        return dict(self.overrides)


def product_grid(param_lists: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of param lists -> list of param-combination dicts.

    Returns ``[{}]`` for an empty grid (single baseline run).
    """
    if not param_lists:
        return [{}]
    keys = list(param_lists)
    combos = product(*(param_lists[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def _build_config(cfg: StrategyConfig, overrides: dict[str, Any]) -> StrategyConfig:
    """Return a copy of cfg with overrides applied (config fields or strategy params)."""
    cfg_kwargs: dict[str, Any] = {}
    strat_kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _CONFIG_FIELDS:
            cfg_kwargs[key] = value
        else:
            strat_kwargs[key] = value
    return replace(
        cfg,
        **cfg_kwargs,
        strategy_params={**cfg.strategy_params, **strat_kwargs},
    )


def run_sweep(
    cfg: StrategyConfig,
    param_lists: dict[str, list],
    sort_metric: str = "annual_return",
) -> list[SweepResult]:
    """Run every grid combination and return results ranked by sort_metric.

    Candles are loaded once over the config's full window and reused across
    combinations (only params vary). Strategy module state is reset before
    every run so combinations don't bleed into each other.
    """
    metric_names = {f.name for f in fields(PortfolioResult)}
    if sort_metric not in metric_names:
        raise ValueError(
            f"Unknown sort metric {sort_metric!r}; available: "
            f"{', '.join(sorted(metric_names))}"
        )

    from src.bt.engine.backtest import Backtest, run
    from src.bt.data_feed import load_candles
    from src.bt.strategies import init_strat

    strat_mod = init_strat(cfg.strategy_type)
    reset = getattr(strat_mod, "reset_global", None)

    grid = product_grid(param_lists)
    data = None
    results: list[SweepResult] = []

    for overrides in grid:
        conf = _build_config(cfg, overrides)
        if data is None:
            bt_probe = Backtest(conf)
            data = load_candles(
                conf.symbols,
                bt_probe.window.train_start,
                bt_probe.window.test_end,
                conf.bars[0],
            )
        if reset is not None:
            reset()
        bt = Backtest(conf)
        res = run(bt, data, strat_mod=strat_mod)
        results.append(
            SweepResult(
                overrides=overrides,
                params=dict(conf.strategy_params),
                pf=res.pf,
            )
        )

    # Higher is better for every PortfolioResult metric we order by
    # (max_drawdown is negative, so ascending = descending here too).
    results.sort(key=lambda r: getattr(r.pf, sort_metric), reverse=True)
    return results


def render_sweep_report(
    results: list[SweepResult],
    sort_metric: str,
    limit: int | None = None,
) -> str:
    """Render ranked sweep results as an aligned table."""
    from src.bt.table import Col, Table, render

    shown = results if limit is None else results[:limit]

    metric_cols: tuple[tuple[str, Callable[[SweepResult], str]], ...] = (
        ("TotalRet", lambda r: f"{r.pf.total_return:.2%}"),
        ("Annual", lambda r: f"{r.pf.annual_return:.2%}"),
        ("Sharpe", lambda r: f"{r.pf.sharpe_ratio:.2f}"),
        ("MaxDD", lambda r: f"{r.pf.max_drawdown:.2%}"),
        ("Calmar", lambda r: f"{r.pf.calmar_ratio:.2f}"),
        ("Trades", lambda r: str(len(r.pf.trades))),
    )

    # Param columns: union of all swept keys across results, stable order.
    all_keys: list[str] = []
    for r in shown:
        for k in r.overrides:
            if k not in all_keys:
                all_keys.append(k)

    headers: tuple[Col, ...] = tuple([Col("params", "<")])
    headers = headers + tuple(Col(name, ">") for name, _ in metric_cols)

    rows: tuple[tuple[str, ...], ...] = tuple(
        tuple(
            [
                " ".join(f"{k}={r.overrides[k]}" for k in all_keys),
                *(fmt(r) for _, fmt in metric_cols),
            ]
        )
        for r in shown
    )

    lines = [
        f"Sweep: {len(results)} combos · sorted by {sort_metric}",
        "(swept params in the `params` column; metrics as usual)",
    ]
    lines.extend(render(Table(columns=headers, rows=rows)))
    return "\n".join(lines).rstrip()


def sweep_report_to_json(results: list[SweepResult]) -> dict:
    """Serialize ranked sweep results into a JSON-ready dict."""
    float_fields = (
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "sortino_ratio",
    )

    def _result_dict(pf: PortfolioResult) -> dict:
        return {f: float(getattr(pf, f)) for f in float_fields}

    return {
        "results": [
            {
                "params": r.overrides,
                "metrics": _result_dict(r.pf),
            }
            for r in results
        ]
    }


__all__ = [
    "SweepResult",
    "product_grid",
    "run_sweep",
    "render_sweep_report",
    "sweep_report_to_json",
]
