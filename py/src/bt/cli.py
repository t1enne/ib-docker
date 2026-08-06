"""Backtest CLI — run walk-forward backtests from strategy configs.

Usage:
    py bt run breakout.json
    py data query AAPL --from 2024-01-01 | py ind ema --span 20 | py bt run strategy.json
"""

from __future__ import annotations

import json

import click
import pandas as pd


@click.group(name="bt")
def bt_group():
    """Backtesting engine for trading strategies."""


@bt_group.command(name="run")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["jsonl", "json", "text"]),
    default="text",
    help="Output format: jsonl (equity curve lines), json (full result), "
    "or text (summary)",
)
def bt_run(strategy_file: str, fmt: str):
    """Run a backtest from a strategy JSON config file.

    STRATEGY_FILE: JSON config with symbols, dates, strategy params.

    Output:
      text  — human-readable summary table.
      json  — full structured result as one JSON document (metrics, trades,
              equity curve, benchmarks).
      jsonl — equity curve points as JSON lines, then a final metrics+trades
              record.
    """
    from src.bt import load_strategy, run_backtest_results
    from src.bt import get_backtest_results_analysis
    from src.bt.output import render_result_json, render_result_jsonl

    config = load_strategy(strategy_file)
    results = run_backtest_results(config)

    if fmt == "json":
        click.echo(
            json.dumps(render_result_json(results), indent=2, default=_json_default)
        )
    elif fmt == "jsonl":
        for line in render_result_jsonl(results):
            click.echo(json.dumps(line, default=_json_default))
    else:
        click.echo(
            get_backtest_results_analysis(
                results.pf, benchmark_curves=results.benchmark_curves
            )
        )


@bt_group.command(name="split")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--is-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Anchor IS/OOS split date (single-split mode).",
)
@click.option(
    "--folds",
    type=int,
    default=None,
    help="Walk-forward fold count (overrides --is-end).",
)
@click.option(
    "--min-is-years",
    type=float,
    default=5.0,
    help="Min IS window length for walk-forward.",
)
@click.option(
    "--train-start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Override warmup start before IS.",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Report format.",
)
def bt_split(
    strategy_file: str,
    is_end,
    folds,
    min_is_years,
    train_start,
    fmt: str,
):
    """Walk-forward / single-split in-sample vs out-of-sample validation.

    Evaluates the strategy's FIXED params across IS/OOS windows. Reports
    out-of-sample Sharpe and IS to OOS degradation. Does NOT re-tune params.
    """
    from src.bt import load_strategy
    from src.bt.split import (
        anchor_split,
        render_split_report,
        run_split,
        split_report_to_dict,
        walk_forward_folds,
    )

    cfg = load_strategy(strategy_file)
    train_start_ts: pd.Timestamp | None = (
        _cli_ts(train_start) if train_start is not None else None
    )

    try:
        if folds is not None:
            folds_list = walk_forward_folds(
                cfg,
                folds,
                min_is_years=min_is_years,
                train_start=train_start_ts,
            )
        elif is_end is not None:
            folds_list = anchor_split(cfg, _cli_ts(is_end), train_start=train_start_ts)
        else:
            raise click.UsageError(
                "Provide one of --is-end (single split) or --folds (walk-forward)."
            )

        report = run_split(cfg, folds_list)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if fmt == "json":
        click.echo(json.dumps(split_report_to_dict(report), indent=2))
    else:
        click.echo(render_split_report(report))


@bt_group.command(name="sweep")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.argument("param_grid", type=str)
@click.option(
    "--sort-by",
    default="annual_return",
    help="PortfolioResult metric to rank combos by (e.g. sharpe_ratio).",
)
@click.option(
    "--limit",
    "top_n",
    type=int,
    default=None,
    help="Show only the top N combos (default: all).",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Report format.",
)
def bt_sweep(
    strategy_file: str, param_grid: str, sort_by: str, top_n: int | None, fmt: str
):
    """Sweep a strategy's params over a grid and rank backtest results.

    PARAM_GRID: a partial config JSON that deep-merges into the strategy
    config. Top-level keys override StrategyConfig fields; a nested
    "strategy_params" object merges into the strategy's own params. Any value
    that is a list is swept (cartesian product over all sweepable leaves);
    scalars override once.

    Example:
      ibkr bt sweep mystrat.json '{"position_size":[0.8,0.95], "strategy_params":{"drift_tolerance":[0.01,0.05]}}'
    """
    from src.bt import load_strategy
    from src.bt.sweep import (
        render_sweep_report,
        run_sweep,
        sweep_report_to_json,
    )

    try:
        grid: dict = _parse_param_grid(param_grid)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    cfg = load_strategy(strategy_file)
    results = run_sweep(cfg, grid, sort_metric=sort_by)

    if fmt == "json":
        click.echo(json.dumps(sweep_report_to_json(results), indent=2))
    else:
        click.echo(render_sweep_report(results, sort_metric=sort_by, limit=top_n))


@bt_group.command(name="analyze")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="json",
    help="Output format: json (full metrics/trades, default) or text (summary).",
)
def bt_analyze(strategy_file: str, fmt: str):
    """Analyze a backtest result from a strategy config.

    Runs the backtest once, then reports the actual PortfolioResult metrics -
    never re-parsed from rendered text.

    Output: JSON with metrics, trades, and equity curve (default), or the
    human-readable summary (text).
    """
    from src.bt import (
        load_strategy,
        run_backtest_results,
        get_backtest_results_analysis,
    )
    from src.bt.output import render_result_json

    config = load_strategy(strategy_file)
    results = run_backtest_results(config)

    if fmt == "text":
        click.echo(
            get_backtest_results_analysis(
                results.pf, benchmark_curves=results.benchmark_curves
            )
        )
    else:
        click.echo(
            json.dumps(render_result_json(results), indent=2, default=_json_default)
        )


def _json_default(o):
    """JSON fallback encoder for non-native objects at the output edge.

    Converts pandas Timestamps (equity/z-score) and Enum members to plain
    values. Used only when serializing CLI output — never inside engine logic.
    """
    if isinstance(o, pd.Timestamp):
        return str(o)
    value = getattr(o, "value", None)
    if value is not None:
        return value
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _parse_param_grid(raw: str) -> dict:
    """Parse a JSON-like param grid, tolerating unquoted bare keys.

    Accepts strict JSON and shorthand like ``{ma_slow: [9, 14, 21]}``.
    Column values (keys) with no quotes are wrapping in double quotes before
    parsing. Single-quoted text stays literal.
    """
    try:
        return json.loads(raw, parse_int=int, parse_float=float)
    except json.JSONDecodeError:
        pass
    import re

    # Tolerate unquoted bare keys anywhere in the JSON: wrap each bare key
    # (identifier not already preceded by a quote) in double quotes.
    quoted = re.sub(
        r"(?P<notsquote>^|[^\"'])([A-Za-z_][A-Za-z0-9_]*)\s*:",
        r'\g<notsquote>"\2" :',
        raw,
    )
    try:
        parsed = json.loads(quoted, parse_int=int, parse_float=float)
    except json.JSONDecodeError as exc:
        raise ValueError(f"param_grid is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError("param_grid must be a JSON object.")
    return parsed


def _cli_ts(dt) -> pd.Timestamp:
    """Convert a click DateTime value into a non-NaT Timestamp."""
    ts = pd.Timestamp(dt)
    if pd.isna(ts):
        raise click.UsageError(f"Invalid datetime: {dt}")
    assert isinstance(ts, pd.Timestamp)
    return ts
