"""Typed strategy parameter dataclasses.

Each strategy defines a frozen dataclass with its parameters and defaults.
The engine instantiates these once from StrategyConfig.strategy_params dict,
and strategies receive a typed instance instead of a raw dict.

Usage in a strategy module:
    from dataclasses import dataclass
    from src.bt.strategies.types import StrategyParams

    @dataclass(frozen=True)
    class Params(StrategyParams):
        fast: int = 9
        slow: int = 14
        vol_window: int = 20

    def on_candle(state, candle, params: Params) -> list[TradeSignal]:
        ema_fast = ta.ema(closes, params.fast)  # typed, no .get()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TypeVar

T = TypeVar("T", bound="StrategyParams")


@dataclass(frozen=True)
class StrategyParams:
    """Base class for typed strategy parameters.

    Provides from_dict() that extracts only declared fields,
    ignoring extras. Subclass with your strategy's parameters.

    Example:
        @dataclass(frozen=True)
        class Params(StrategyParams):
            fast: int = 9
            slow: int = 14
            vol_window: int = 20
            vol_multiplier: float = 1.5
    """

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Extract declared fields from a raw dict, filling defaults for missing keys.

        Extra keys in the dict (e.g., engine-injected 'symbols', 'rolling_window_size')
        are ignored — they're available via StrategyConfig directly.
        """
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)
