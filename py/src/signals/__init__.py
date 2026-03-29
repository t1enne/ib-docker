"""Signal generation module for live trading.

Subscribes to IBKR websocket bar data and runs existing strategy
on_tick() logic to generate and print trading signals.

Usage:
    python main.py signal strats/breakout_ema.yaml
"""

from src.signals.generator import SignalGenerator, generate_signals
from src.signals.feed import LiveBarFeed
from src.signals.types import SignalEvent

__all__ = [
    "SignalGenerator",
    "generate_signals",
    "LiveBarFeed",
    "SignalEvent",
]
