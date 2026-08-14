"""Hyperparameter sweep for a strategy.

The sweep input is a partial config JSON that deep-merges into the strategy
config: top-level keys override ``StrategyConfig`` fields and a nested
``strategy_params`` object merges into the strategy's own params. Any value
that is a list is swept (cartesian product over all list-valued leaves).

Example (mirrors the config shape, no extra routing keys):

    {"strategy_params": {"position_size": [0.7, 0.85, 1.0], "rebalance_frequency": [2, 5]}}

Pure grid/expand logic lives here (test-friendly); engine wiring is
``run_sweep``. Mirrors the repo's ``pure.py`` convention.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from itertools import product
from typing import Any, Callable

import pandas as pd

from src.bt.types import StrategyConfig, PortfolioResult


@dataclass(frozen=True)
class SweepResult:
    """One grid combination and its backtest result."""

    overrides: dict[str, Any]
    params: dict[str, Any]  # merged strategy_params (for reporting)
    pf: PortfolioResult


# ---------------------------------------------------------------------------
# deep merge + grid expansion
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into ``base`` (nested dicts merged in place)."""
    out = dict(base)
    for key, value in update.items():
        cur = out.get(key)
        if isinstance(value, dict) and isinstance(cur, dict):
            out[key] = _deep_merge(cur, value)
        else:
            out[key] = value
    return out


def _expand_leaves(node: dict) -> list[tuple[tuple[str, ...], Any]]:
    """Collect every (path, candidate values) leaf that is a list."""
    leaves: list[tuple[tuple[str, ...], Any]] = []

    def walk(sub: dict, path: tuple[str, ...]) -> None:
        for key, value in sub.items():
            child = path + (key,)
            if isinstance(value, list):
                leaves.append((child, value))
            elif isinstance(value, dict):
                walk(value, child)

    walk(node, ())
    return leaves


def grid_combos(merge: dict) -> list[dict[str, Any]]:
    """Expand list-valued leaves in a deep-merge override into combos.

    Scalar values are fixed; every list-valued leaf is swept. Returns one
    merged override dict per cartesian combination (empty ``merge`` -> ``[{}]``).
    """
    leaves = _expand_leaves(merge)
    if not leaves:
        return [merge]
    keys, value_lists = zip(*leaves)
    combos: list[dict[str, Any]] = []
    for values in product(*value_lists):
        out = deepcopy(merge)
        for path, val in zip(keys, values):
            cursor = out
            for segment in path[:-1]:
                cursor = cursor[segment]
            cursor[path[-1]] = val
        combos.append(out)
    return combos


def build_config(cfg: StrategyConfig, merge: dict) -> StrategyConfig:
    """Deep-merge ``merge`` into ``cfg`` and rebuild the config.

    Top-level keys override ``StrategyConfig`` fields; ``strategy_params``
    merges into the config's strategy params dict.
    """
    config_dict = {f.name: getattr(cfg, f.name) for f in fields(StrategyConfig)}
    merged = _deep_merge(config_dict, merge)
    return StrategyConfig(**merged)


def _combo_data_span(conf: StrategyConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (train_start, test_end) load span a combo needs."""
    from src.bt.engine.backtest import Backtest

    b = Backtest(conf)
    return b.window.train_start, b.window.test_end


def run_sweep(
    cfg: StrategyConfig,
    merge: dict,
    sort_metric: str = "annual_return",
    on_result: Callable[[int, int, dict[str, Any], PortfolioResult], None]
    | None = None,
) -> list[SweepResult]:
    """Sweep ``merge`` (partial config JSON) over ``cfg``, ranked by sort_metric.

    List-valued leaves in ``merge`` are swept (cartesian). Scalar leaves
    override once. Strategy module state resets between runs.

    Data loading is grouped by symbol set: for each distinct ``symbols`` list
    across combos, candles load once over the *union* of all that set's spans
    (train_start..test_end), then each combo's window is sliced from that feed
    via ``window_df`` — no per-combo reload, and sweeping top-level
    ``symbols`` / ``training_start`` / ``trading_start`` / ``trading_end``
    now works correctly.

    ``on_result`` (optional) is called with (index, total, flat_overrides, pf)
    as each combo finishes, letting callers stream results live instead of
    waiting for the full ranking.
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
    from src.bt.window import window_df

    strat_mod = init_strat(cfg.strategy_type)
    reset = getattr(strat_mod, "reset_global", None)

    combos = grid_combos(merge)
    results: list[SweepResult] = []

    # Bucket combos by (symbol set, bar) so each distinct set loads exactly once.
    confs = [build_config(cfg, patch) for patch in combos]
    by_key: dict[tuple[tuple[str, ...], str], list[StrategyConfig]] = {}
    for conf in confs:
        by_key.setdefault((tuple(conf.symbols), conf.bars[0]), []).append(conf)

    # Load once per key over the union of that set's spans (train_start..test_end).
    feed_cache: dict[tuple[tuple[str, ...], str], pd.DataFrame] = {}
    for (symbols, bar), conf_list in by_key.items():
        spans = [_combo_data_span(c) for c in conf_list]
        load_start = min(s for s, _ in spans)
        load_end = max(e for _, e in spans)
        feed_cache[(symbols, bar)] = load_candles(
            list(symbols), load_start, load_end, bar
        )

    for patch, conf in zip(combos, confs):
        bt = Backtest(conf)
        data = window_df(
            feed_cache[(tuple(conf.symbols), conf.bars[0])], bt.window.test_end
        )
        if reset is not None:
            reset()
        res = run(bt, data, strat_mod=strat_mod)

        # Flatten swept leaf values for reporting (dot-joined path = value).
        overrides: dict[str, Any] = {}
        for path, _ in _expand_leaves(merge):
            node: Any = patch
            for segment in path:
                node = node[segment]
            overrides[".".join(path)] = node

        if on_result is not None:
            on_result(len(results), len(combos), overrides, res.pf)

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
    "grid_combos",
    "build_config",
    "run_sweep",
    "render_sweep_report",
    "sweep_report_to_json",
]
