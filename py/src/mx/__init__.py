import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from src.utils import read_candles


def calculate_cointegration(price1, price2) -> float:
    # Perform OLS regression: price1 = a + b * price2
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    residuals = model.resid
    # ADF test on residuals
    adf_result = adfuller(residuals)
    p_value = float(adf_result[1])
    return p_value


def get_metrics(df1: pd.DataFrame, df2: pd.DataFrame):
    # Merge on Date
    merged = pd.merge(
        df1[["Date", "Close"]],
        df2[["Date", "Close"]],
        on="Date",
        suffixes=("_1", "_2"),
    )
    # Set Date as index
    merged.set_index("Date", inplace=True)
    # Calculate daily log returns
    merged.loc[:, "Return_1"] = np.log(
        merged["Close_1"] / merged["Close_1"].shift(1)
    ).round(4)
    merged.loc[:, "Return_2"] = np.log(
        merged["Close_2"] / merged["Close_2"].shift(1)
    ).round(4)
    # Drop NaN
    merged = merged.dropna()
    # Correlation of log returns
    corr = np.corrcoef(merged["Return_1"], merged["Return_2"])[0, 1]
    # Cointegration p-value
    p_value = calculate_cointegration(merged["Close_1"], merged["Close_2"])
    return round(corr, 2), round(p_value, 2)


def matrix(symbols: list[str]):
    symbols = list(map(lambda s: s.upper(), symbols))
    data = {sym: read_candles(sym.upper()) for sym in symbols}

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

    print("Correlation Matrix:")
    print(corr_matrix)
    print("\nCointegration Matrix (p-values):")
    print(cointegration_matrix)


__all__ = ["matrix"]
