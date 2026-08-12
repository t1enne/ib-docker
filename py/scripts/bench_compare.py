"""Run a set of strategy configs, timing each, and report annual_return + trade count.

Usage:
    uv run python scripts/bench_compare.py [config1.json ...]
"""

from __future__ import annotations

import json
import sys
from time import perf_counter

from src.bt import load_strategy, run_backtest_results


def bench(path: str) -> dict:
    config = load_strategy(path)
    strat_type = config.strategy_type
    t0 = perf_counter()
    results = run_backtest_results(config)
    elapsed = perf_counter() - t0

    annual_return = getattr(results.pf, "annual_return", None)
    trades = getattr(results.pf, "trades", None) or []
    n_trades = len(trades)

    return {
        "config": path,
        "strategy_type": strat_type,
        "name": getattr(config, "name", ""),
        "elapsed_s": round(elapsed, 3),
        "annual_return": _num(annual_return),
        "n_trades": n_trades,
    }


def _num(v):
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except TypeError, ValueError:
        return v


def main(paths: list[str]) -> None:
    rows = []
    for p in paths:
        try:
            row = bench(p)
        except Exception as e:  # noqa: BLE001
            row = {"config": p, "error": str(e)}
        rows.append(row)
        print(json.dumps(row, indent=2))
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
