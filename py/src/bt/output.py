"""Rendering of backtest results into JSON-shaped output.

This is the output boundary. It consumes ``BacktestResults`` / ``PortfolioResult``
/ ``Trade`` objects directly and returns JSON-serializable structures — it does
NOT define a backtest_results→dict intermediate that flows through the engine.
Engine logic deals in ``BacktestResults``; text and JSON are both produced from
the same objects when output is actually emitted.

The helpers return plain dicts/points that already carry JSON-native scalars
(floats, strings); pandas Timestamps and Enums are normalized here, so callers
pass ``default=_json_default`` only as a safety net.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def _ts_str(v: Any) -> str | None:
    """Format a timestamp-like value as an ISO string, else None."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return str(v)
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v)


def _metric_pairs(pf: Any) -> Iterable[tuple[str, float]]:
    """Yield (name, float(value)) for every scalar PortfolioResult metric."""
    from dataclasses import fields

    skip = {"trades", "equity_curve"}
    for f in fields(pf):
        if f.name in skip:
            continue
        val = getattr(pf, f.name)
        if isinstance(val, bool):  # never a PortfolioResult metric, but be safe
            continue
        if isinstance(val, (int, float)):
            yield f.name, float(val)
        else:
            yield f.name, str(val)


def trade_json(trade: Any) -> dict[str, Any]:
    """One Trade -> JSON-ready dict (scalars already flattened)."""
    return {
        "position_id": getattr(trade, "position_id", ""),
        "symbol": trade.symbol,
        "position": getattr(trade.position, "value", trade.position),
        "qty": float(trade.qty),
        "entry_time": _ts_str(trade.entry_time),
        "entry_price": float(trade.entry_price),
        "exit_time": _ts_str(trade.exit_time),
        "exit_price": float(trade.exit_price) if trade.exit_price is not None else None,
        "last_price": float(trade.last_price),
        "stop_loss": float(trade.stop_loss) if trade.stop_loss is not None else None,
        "take_profit": float(trade.take_profit)
        if trade.take_profit is not None
        else None,
        "pnl": float(trade.pnl),
        "commission": float(trade.commission),
        "slippage": float(trade.slippage),
        "status": getattr(trade.status, "value", trade.status),
        "close_reason": (
            getattr(trade.close_reason, "value", trade.close_reason)
            if trade.close_reason is not None
            else None
        ),
        "reason": str(trade.reason) if trade.reason is not None else None,
    }


def equity_points(equity_curve: pd.Series) -> list[dict[str, Any]]:
    """Equity curve -> [{"ts": iso, "equity": float}, ...]."""
    out: list[dict[str, Any]] = []
    for ts, val in equity_curve.items():
        out.append({"ts": _ts_str(ts) or str(ts), "equity": float(val)})
    return out


def benchmark_json(benchmark_curves: dict[str, pd.Series]) -> dict[str, Any]:
    """Benchmark curves -> {symbol: [{"ts": iso, "equity": float}, ...]}."""
    return {sym: equity_points(curve) for sym, curve in benchmark_curves.items()}


def render_result_json(results: Any) -> dict[str, Any]:
    """BacktestResults -> one JSON-ready dict (metrics + trades + equity + benchmarks)."""
    return {
        "metrics": dict(_metric_pairs(results.pf)),
        "trades": [trade_json(t) for t in results.pf.trades],
        "equity_curve": equity_points(results.pf.equity_curve),
        "benchmark_curves": benchmark_json(results.benchmark_curves),
    }


def render_result_jsonl(results: Any) -> list[dict[str, Any]]:
    """BacktestResults -> JSONL-shaped list.

    One dict per equity-curve point, then a final ``{"metrics": ..., "trades":
    ...}`` record for consumers that read to EOF.
    """
    points: list[dict[str, Any]] = [
        {"ts": _ts_str(ts) or str(ts), "equity": float(val)}
        for ts, val in results.pf.equity_curve.items()
    ]
    points.append(
        {
            "metrics": dict(_metric_pairs(results.pf)),
            "trades": [trade_json(t) for t in results.pf.trades],
        }
    )
    return points


__all__ = [
    "trade_json",
    "equity_points",
    "benchmark_json",
    "render_result_json",
    "render_result_jsonl",
]
