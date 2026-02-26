import pandas as pd
from src.utils import get_ols_fit_model, get_local_candles


def pnd(symbols: list[str]):
    df1 = get_local_candles(symbols[0].upper())
    df2 = get_local_candles(symbols[1].upper())
    s1 = df1["Close"]
    s2 = df2["Close"]
    model = get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    scaled_s2 = alpha + beta * s2
    spread_series = s1 - scaled_s2
    z_score = (spread_series - spread_series.mean()) / spread_series.std()
    pd.DataFrame(
        {
            symbols[0]: s1,
            f"{symbols[1]}_scaled": scaled_s2,
            "spread": spread_series,
            "z_score": z_score,
        }
    )
