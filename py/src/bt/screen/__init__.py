"""Screen module — engine-agnostic scoring/signaling for manual trading.

High-level entry points:

  - ``screen()``  — load OHLCV from the feed, build state, run a screen, rank.
  - ``build_state`` / ``run_screen`` / ``rank`` — the pure building blocks.

A screen never emits a ``TradeSignal``; it returns ``ScreenResult`` with a
0..1 score and human-readable reasons. A high score means "condition fired",
not "profitable trade" — scoring is pre-cost by design.
"""

from __future__ import annotations

import pandas as pd

from src.bt.screen.runner import (
    build_state,
    run_screen,
    rank,
    screen_over_history,
    rank_divergence,
    DivergenceParams,
)
from src.bt.screen.screens import init_screen, resolve_screen_params
from src.bt.screen.types import (
    ScreenParams,
    ScreenResult,
    ScreenFn,
    ScreenState,
)


def screen(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
    screen_name: str,
    params: dict | None = None,
) -> tuple[ScreenResult, ...]:
    """One-shot convenience: feed OHLCV -> score -> rank, newest bar only.

    Scores only the latest bar (the current decision), which is the manual-
    trading use case. For a walk across history, build states per timestamp and
    call ``run_screen`` repeatedly over the frames.

    Lazy-imports the feed adapter so the pure screen core stays lighter than a
    data_feed dependency (and avoids a package-init cycle with ``src.bt``).
    """
    from src.bt.screen.adapter import state_from_feed

    state = state_from_feed(symbols, start, end, bar)
    return rank(run_screen(state, screen_name, params))


__all__ = [
    "ScreenParams",
    "ScreenResult",
    "ScreenFn",
    "ScreenState",
    "build_state",
    "run_screen",
    "rank",
    "screen",
    "screen_over_history",
    "rank_divergence",
    "DivergenceParams",
    "init_screen",
    "resolve_screen_params",
]
