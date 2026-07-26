from typing import Dict
from dataclasses import dataclass, field


@dataclass
class RegimeStats:
    n_regimes: int
    mean_return: Dict[int, float] = field(default_factory=dict)
    volatility: Dict[int, float] = field(default_factory=dict)
    frequency: Dict[int, float] = field(default_factory=dict)
