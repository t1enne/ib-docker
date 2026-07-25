"""Screen module — Finviz-like stock screener using strategy logic.

A screen module lives under a project-level `screens/` directory
(e.g., `screens/breakout_screen.py`), exports a `make()` factory and
conforms to the `ScreenFn` Protocol:

    from src.screen.types import ScreenFn, ScreenResult

    def make(symbols: list[str], params: dict[str, Any]) -> ScreenFn: ...

The factory receives the full symbol list and params at construction time
so implementors can precompute universe-level statistics.

Usage (CLI):
    python main.py screen breakout_screen universe.yml --param fast=50

Usage (Python):
    from src.screen import load_screen, run_screen
    from src.syncm import load_universe_config

    universe = load_universe_config("universe.yml")
    output = await run_screen("breakout_screen", universe.symbols)
"""

from datetime import date

import asyncio
import importlib.util
import logging
import os
import sys
from typing import Any

from src.screen.style import Fmt, Style
from src.screen.types import (
    ScreenFn,
    ScreenResult,
    ScreenOutput,
)

__all__ = [
    # Types
    "ScreenFn",
    "ScreenResult",
    "ScreenOutput",
    # Core API
    "import_screen",
    "make_screen",
    "run_screen",
    "cli_screen",
    # Terminal styling (reusable)
    "Style",
    "Fmt",
]

logger = logging.getLogger(__name__)

# Where screen modules are discovered
_SCREENS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "screens")
)


def discover_screens() -> list[str]:
    """List all available screen module names (without .py suffix)."""
    if not os.path.isdir(_SCREENS_DIR):
        return []

    names: list[str] = []
    for f in sorted(os.listdir(_SCREENS_DIR)):
        if f.endswith(".py") and f != "__init__.py":
            names.append(f.removesuffix(".py"))
    return names


def import_screen(name: str) -> Any:
    """Dynamically import a screen module and return the module object.

    Validates that the module exports a ``make()`` factory.
    Does NOT call the factory — use ``run_screen`` to construct an instance
    with the desired parameters after merging with DEFAULTS.

    Args:
        name: Screen module name (e.g. "breakout_screen")

    Returns:
        The imported module object

    Raises:
        ModuleNotFoundError: If no screen module with that name exists
        AttributeError: If the module is missing ``make()``
    """
    module_path = os.path.join(_SCREENS_DIR, f"{name}.py")
    if not os.path.isfile(module_path):
        available = discover_screens()
        raise ModuleNotFoundError(
            f"Screen module '{name}' not found at {module_path}.\n"
            f"Available screens: {available if available else '(none)'}"
        )

    spec = importlib.util.spec_from_file_location(f"screens.{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load screen module from {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"screens.{name}"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "make"):
        raise AttributeError(
            f"Screen module '{name}' is missing required factory 'make()'.\n"
            f"Expected: def make("
            f"symbols: list[str], params: dict[str, Any]) -> ScreenFn: ..."
        )

    return mod


def make_screen(
    name: str,
    symbols: list[str],
    params: dict[str, Any] | None = None,
) -> ScreenFn:
    """Import a screen module and construct a ScreenFn instance.

    Merges ``params`` with the module-level ``DEFAULTS``, then calls
    the module's ``make()`` factory. Validates the result conforms to
    the ``ScreenFn`` Protocol at runtime.

    Args:
        name: Screen module name (e.g. "breakout_screen")
        symbols: List of ticker symbols (for universe-level precomputation)
        params: Merged parameters (DEFAULTS + user overrides preferred,
                but raw user params will be merged with DEFAULTS here)

    Returns:
        ScreenFn instance
    """
    mod = import_screen(name)

    mod_defaults: dict[str, Any] = getattr(mod, "DEFAULTS", {})
    merged_params = {**mod_defaults, **(params or {})}

    return mod.make(symbols=symbols, params=merged_params)


async def run_screen(
    name: str,
    symbols: list[str],
    from_date: str | None = None,
    to_date: str | None = None,
    bar: str = "1h",
    params: dict[str, Any] | None = None,
) -> ScreenOutput:
    """Run a screen across a universe of symbols.

    Loads the screen module, fetches candle data for each symbol,
    computes results, and ranks them.

    Args:
        name: Screen module name (e.g. "breakout_screen")
        symbols: List of ticker symbols to screen
        from_date: Optional data start date (YYYY-MM-DD)
        to_date: Optional data end date (YYYY-MM-DD)
        bar: Bar size (e.g. "1h", "1d")
        params: Screen-specific parameters (merged with DEFAULTS)

    Returns:
        ScreenOutput with ranked results.
    """
    from src.utils import get_local_candles, parse_timestamp

    instance = make_screen(name, symbols=symbols, params=params)

    start_ts = parse_timestamp(from_date) if from_date else None
    end_ts = parse_timestamp(to_date) if to_date else None

    results: list[ScreenResult] = []
    for symbol in symbols:
        candles = get_local_candles(symbol, start_ts, end_ts)
        if candles.empty:
            logger.warning("No candle data for %s, skipping", symbol)
            continue

        try:
            result = instance.compute(symbol, candles)
            results.append(result)
        except Exception as e:
            logger.error("Error screening %s: %s", symbol, e)
            continue

    ranked = instance.rank(results)
    params = getattr(instance, "params", params or {})
    return ScreenOutput(
        screen_name=name,
        results=tuple(ranked),
        params=dict(params),  # type: ignore[arg-type]
    )


# ── Table rendering ────────────────────────────────────────────


def print_screen_output(output: ScreenOutput) -> str:
    """Format screen output as a table string (Finviz-style) with ANSI colors.

    Renders each row using the ``Fmt`` helpers from ``src.screen.style``.
    Metadata values are formatted via ``Fmt.float_val``, which handles
    percentage keys (``PCT_DISPLAY_KEYS``), adaptive precision, and
    sign-based coloring.

    Args:
        output: ScreenOutput with ranked results

    Returns:
        Formatted table string with ANSI escape codes.
    """
    from io import StringIO

    buf = StringIO()

    # ── Title ──────────────────────────────────────────────
    title = f"╔═══ Screen: {output.screen_name}"
    if output.params:
        param_str = " | ".join(f"{k}={v}" for k, v in output.params.items())
        title += f" ({param_str})"
    title += " ═══╗"
    buf.write(f"\n{Fmt.title(title)}\n\n")

    if not output.results:
        buf.write(f"{Fmt.warning('No results.')}\n")
        return buf.getvalue()

    # ── Column descriptors ─────────────────────────────────
    #   Fixed columns: Rank, Symbol, Signal, Score, Price
    #   Dynamic columns: sorted metadata keys (omit "reason")
    meta_keys = sorted(k for k in output.results[0].metadata if k not in ("reason",))

    # ── Header row ─────────────────────────────────────────
    col_labels = ["Rank", "Symbol", "Signal", "Score", "Price"]
    col_widths = [5, 8, 8, 8, 10]
    for k in meta_keys:
        col_labels.append(k)
        col_widths.append(14)

    header_parts: list[str] = []
    for label, w in zip(col_labels, col_widths, strict=False):
        header_parts.append(Fmt.header_label(f"{label:<{w}}"))
    buf.write(" ".join(header_parts) + "\n")

    # ── Separator ──────────────────────────────────────────
    total_w = sum(col_widths) + len(col_widths) - 1  # spaces between cols
    buf.write(f"{Fmt.dim('─' * total_w)}\n")

    # ── Data rows ──────────────────────────────────────────
    for i, r in enumerate(output.results, 1):
        parts: list[str] = [
            Fmt.rank(i),
            Fmt.symbol(r.symbol),
            Fmt.signal(r.signal),
            Fmt.score(r.score),
            Fmt.price(r.price, r.signal),
        ]
        for k in meta_keys:
            val = r.metadata.get(k, "")
            if isinstance(val, float):
                parts.append(Fmt.float_val(val, k))
            else:
                parts.append(Fmt.plain_val(val))
        buf.write(" ".join(parts) + "\n")

    # ── Footer ─────────────────────────────────────────────
    buf.write(f"\n{Fmt.dim(f'╚═══ {len(output.results)} symbols screened ═══╝')}\n")
    return buf.getvalue()


# ── CLI integration ────────────────────────────────────────────


def _parse_screen_params(
    params: tuple[str, ...],
) -> dict[str, str | float | int | bool]:
    """Parse key=value param strings into typed dict."""
    parsed: dict[str, str | float | int | bool] = {}
    for p in params:
        if "=" not in p:
            raise ValueError(f"Invalid parameter format: '{p}'. Use key=value.")
        key, val = p.split("=", 1)
        try:
            if "." in val:
                parsed[key] = float(val)
            else:
                parsed[key] = int(val)
        except ValueError:
            if val.lower() in ("true", "false"):
                parsed[key] = val.lower() == "true"
            else:
                parsed[key] = val
    return parsed


def cli_screen(
    screen_name: str | None = None,
    universe: str = "universe.yml",
    params: tuple[str, ...] = (),
) -> str:
    """CLI entry point for the screen command.

    Handles param parsing, universe loading, and running the screen.
    Returns the formatted output string for the CLI to echo.

    Args:
        screen_name: Name of the screen module to run
        universe: Path to universe YAML file
        params: Key=value parameter strings

    Returns:
        Formatted output string.
    """
    from src.syncm import load_universe_config

    if not screen_name:
        raise ValueError("SCREEN_NAME is required")

    # Parse params
    parsed_params = _parse_screen_params(params)

    # Load universe
    universe_data = load_universe_config(universe)

    # Resolve date range
    assert universe_data.from_date is not None, "Missing from date in universe config"

    from_date = universe_data.from_date
    to_date = universe_data.to_date or date.today()

    # Run screen
    import click

    click.echo(
        f"Screening {len(universe_data.symbols)} symbols with '{screen_name}'..."
    )
    output = asyncio.run(
        run_screen(
            name=screen_name,
            symbols=universe_data.symbols,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            bar=universe_data.bar,
            params=parsed_params,
        )
    )

    return print_screen_output(output)
