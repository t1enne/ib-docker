from typing import Dict
import pandas as pd
import numpy as np

from src.utils import get_returns, read_candles, symmetric_cointegration_p


def get_metrics(df1: pd.DataFrame, df2: pd.DataFrame):
    # Correlation of log returns
    corr = np.corrcoef(df1["Returns"], df2["Returns"])[0, 1]
    # Cointegration p-value
    p_value = symmetric_cointegration_p(df1["Close"], df2["Close"])
    return round(corr, 2), round(p_value, 3)


def get_candles_info(data: Dict[str, pd.DataFrame]):
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    min_date = min(all_dates)
    max_date = max(all_dates)
    total_days = len(all_dates)
    return min_date, max_date, total_days


def matrix(symbols: list[str]):
    symbols = list(map(lambda s: s.upper(), symbols))
    data = {sym: get_returns(read_candles(sym.upper())) for sym in symbols}

    # Collect all dates across symbols

    min, max, total = get_candles_info(data)
    corr_matrix = pd.DataFrame(index=pd.Index(symbols), columns=pd.Index(symbols))
    cointegration_matrix = pd.DataFrame(
        index=pd.Index(symbols), columns=pd.Index(symbols)
    )

    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            if i == j:
                corr_matrix.loc[sym1, sym2] = 1.0
                cointegration_matrix.loc[sym1, sym2] = 0.0
            else:
                corr, p_val = get_metrics(data[sym1], data[sym2])
                corr_matrix.loc[sym1, sym2] = corr
                cointegration_matrix.loc[sym1, sym2] = p_val

    print(f"Date range: {min.date()} to {max.date()}, Total days: {total}")
    print("Correlation Matrix:")
    print(corr_matrix)
    print("\nCointegration Matrix (p-values):")
    print(cointegration_matrix)


__all__ = ["matrix"]
