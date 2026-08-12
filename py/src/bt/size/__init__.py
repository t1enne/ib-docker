"""Position-sizing layer: pure, config-driven share-quantity computation."""

from src.bt.size.pure import (
    SizingParams,
    compute_qty,
    risk_sized_qty,
    equity_of,
    sized_signal,
    SizingMode,
)

__all__ = [
    "SizingParams",
    "compute_qty",
    "risk_sized_qty",
    "equity_of",
    "sized_signal",
    "SizingMode",
]
