"""Strategy auto-discovery.

Scans src/bt/strategies/ for modules with on_candle + STRATEGY_TYPE.
No manual registration needed — drop a .py file with STRATEGY_TYPE
and on_candle() and it's available.
"""

from __future__ import annotations

import glob
import importlib
import os
from typing import Protocol, cast


class _StrategyModule(Protocol):
    """Shape of a discovered strategy module."""

    STRATEGY_TYPE: str
    __name__: str

    def on_candle(
        self, state: object, candle: object, params: dict
    ) -> list[object]: ...


_strategy_registry: dict[str, _StrategyModule] | None = None


def _discover() -> dict[str, _StrategyModule]:
    """Scan strategies/ directory for strategy modules with STRATEGY_TYPE."""
    pkg_dir = os.path.dirname(__file__)
    pkg_name = "src.bt.strategies"

    registry: dict[str, _StrategyModule] = {}

    for path in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_") or stem == "utils":
            continue

        mod = importlib.import_module(f".{stem}", package=pkg_name)

        if not (hasattr(mod, "on_candle") and hasattr(mod, "STRATEGY_TYPE")):
            continue

        strategy_type: str = getattr(mod, "STRATEGY_TYPE")
        if strategy_type in registry:
            existing = registry[strategy_type]
            raise ImportError(
                f"Duplicate STRATEGY_TYPE '{strategy_type}' in "
                f"{existing.__name__} and {mod.__name__}"
            )

        registry[strategy_type] = cast(_StrategyModule, mod)

    return registry


def init_strat(strat_name: str) -> _StrategyModule:
    """Look up a strategy module by its STRATEGY_TYPE.

    Modules are discovered on first call by scanning src/bt/strategies/*.py
    for modules with both on_candle() and STRATEGY_TYPE defined.
    """
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = _discover()

    try:
        return _strategy_registry[strat_name]
    except KeyError:
        available = sorted(_strategy_registry.keys())
        raise KeyError(
            f"Unknown strategy type '{strat_name}'. Available: {available}"
        ) from None


__all__ = ["init_strat"]
