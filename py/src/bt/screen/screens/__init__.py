"""Screen auto-discovery.

Scans ``src/bt/screen/screens/`` for modules with ``SCREEN_TYPE`` +
``on_state()``. No manual registration — drop a .py file with those and it's
available. Mirrors the strategies convention.
"""

from __future__ import annotations

import glob
import importlib
import os
from typing import Protocol, cast

from src.bt.screen.types import ScreenParams, ScreenResult


class _ScreenModule(Protocol):
    """Shape of a discovered screen module."""

    SCREEN_TYPE: str
    __name__: str
    Params: type[ScreenParams] | None

    def on_state(self, state: object, params: object) -> tuple[ScreenResult, ...]: ...


_registry: dict[str, _ScreenModule] | None = None


def _discover() -> dict[str, _ScreenModule]:
    """Scan screens/ directory for screen modules with SCREEN_TYPE + on_state."""
    pkg_dir = os.path.dirname(__file__)
    pkg_name = "src.bt.screen.screens"

    registry: dict[str, _ScreenModule] = {}

    for path in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_"):
            continue

        mod = importlib.import_module(f".{stem}", package=pkg_name)

        if not (hasattr(mod, "on_state") and hasattr(mod, "SCREEN_TYPE")):
            continue

        screen_type: str = getattr(mod, "SCREEN_TYPE")
        if screen_type in registry:
            existing = registry[screen_type]
            raise ImportError(
                f"Duplicate SCREEN_TYPE '{screen_type}' in "
                f"{existing.__name__} and {mod.__name__}"
            )

        registry[screen_type] = cast(_ScreenModule, mod)

    return registry


def init_screen(screen_name: str) -> _ScreenModule:
    """Look up a screen module by its SCREEN_TYPE."""
    global _registry
    if _registry is None:
        _registry = _discover()

    try:
        return _registry[screen_name]
    except KeyError:
        available = sorted(_registry.keys())
        raise KeyError(
            f"Unknown screen '{screen_name}'. Available: {available}"
        ) from None


def resolve_screen_params(
    screen_name: str,
    raw_params: dict,
) -> ScreenParams | dict:
    """Instantiate typed Params if the screen defines one, else return raw dict."""
    mod = init_screen(screen_name)
    params_cls = getattr(mod, "Params", None)
    if params_cls is None:
        return raw_params
    return params_cls.from_dict(raw_params)


__all__ = ["init_screen", "resolve_screen_params"]
