"""Screen core types.

A screen is an engine-agnostic scoring/signaling layer for manual trading.
Unlike backtest strategies (which emit ``TradeSignal`` fill instructions from a
``BacktestState``), a screen takes a thin ``ScreenState`` view (frames + already
computed model fields) and returns a ranked ``ScreenResult`` — a 0..1 score and
human-readable reasons, never a fill instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import pandas as pd

from src.bt.strategies.types import StrategyParams

TrendRegime = Literal["BULL", "BEAR", "RANGE"]
VolRegime = Literal["LOW_VOL", "MED_VOL", "HIGH_VOL"]

Action = Literal["long", "short", "flat"]

T = TypeVar("T", bound="StrategyParams")


@dataclass(frozen=True)
class ScreenParams(StrategyParams):
    """Base for typed per-screen scoring knobs.

    Reuses ``StrategyParams.from_dict`` (extracts declared fields, ignores
    extras). Subclass per screen with its scoring knobs.
    """


@dataclass(frozen=True)
class ScreenState:
    """Thin, purpose-built input view — NOT ``BacktestState``.

    A screen has no portfolio, no loop cursor, and no model updater. It reads
    OHLCV frames directly and, where a screen needs a regime/vol label, those
    are precomputed by the runner/adapter (never by the engine).
    """

    ts: pd.Timestamp
    # (symbol, OHLCV frame) — each frame has columns open/high/low/close/volume.
    frames: tuple[tuple[str, pd.DataFrame], ...]
    trend: dict[str, TrendRegime | None]  # per-symbol "BULL"/"BEAR"/"RANGE"/None
    vol: dict[str, VolRegime | None]  # per-symbol "LOW_VOL"/"MED_VOL"/"HIGH_VOL"/None

    def frame(self, symbol: str) -> pd.DataFrame | None:
        """Return the OHLCV frame for ``symbol``, or None if absent."""
        for sym, df in self.frames:
            if sym == symbol:
                return df
        return None


@dataclass(frozen=True)
class ScreenResult:
    """One scored decision for a symbol at a timestamp.

    ``score`` is a 0..1 signal strength — NOT a quantity, NOT a profit
    expectation. A high score means "condition fired", not "trade this".
    """

    symbol: str
    timestamp: pd.Timestamp
    score: float  # 0..1
    action: Action
    signals: tuple[str, ...]  # human-readable reasons, e.g. "mom cross up"
    model_features: dict[str, float]  # diagnostic: momentum, atr, spread, ...


class ScreenFn(Protocol):
    """Signature of a screen: state + params -> scored results."""

    def __call__(
        self, state: ScreenState, params: ScreenParams
    ) -> tuple[ScreenResult, ...]: ...
