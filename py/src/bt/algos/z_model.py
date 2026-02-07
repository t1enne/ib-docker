from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

from src.utils import get_ols_fit_model


@dataclass
class TrainedZModel:
    beta: float
    mean: float
    std: float
    n_observations: int


class ZModel:
    def __init__(self, symbols: List[str], rolling_window_size: int):
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size

    def train(self, data: dict[str, pd.DataFrame]) -> TrainedZModel:
        sym1, sym2 = self.symbols
        s1_vals = [float(x) for x in data[sym1]["Close"].dropna().values]
        s2_vals = [float(x) for x in data[sym2]["Close"].dropna().values]

        min_len = min(len(s1_vals), len(s2_vals))
        s1_vals = s1_vals[-min_len:]
        s2_vals = s2_vals[-min_len:]

        if len(s1_vals) < 2:
            beta = 1.0
        else:
            model = get_ols_fit_model(pd.Series(s1_vals), pd.Series(s2_vals))
            _, beta = model.params

        spreads = [s1_vals[i] - beta * s2_vals[i] for i in range(len(s1_vals))]

        mean = float(np.mean(spreads))
        std = float(np.std(spreads, ddof=1))

        return TrainedZModel(
            beta=beta,
            mean=mean,
            std=std,
            n_observations=len(spreads),
        )

    def calculate_z(self, prices: dict[str, float], model: TrainedZModel) -> float:
        sym1, sym2 = self.symbols
        price1 = float(prices[sym1])
        price2 = float(prices[sym2])

        spread = price1 - model.beta * price2

        if model.std == 0:
            return 0.0

        z = (spread - model.mean) / model.std
        return round(z, 2)
