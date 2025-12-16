from typing import Any, Dict, List
import pandas as pd
import sqlite3
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime


def read_candles(
    symbol: str, start_date: str | None = None, end_date: str | None = None
):
    # return pd.read_csv(f"mdata/{symbol}.csv")
    con = sqlite3.connect("../data/db.sqlite")
    cur = con.cursor()
    _sd = start_date and datetime.strptime(start_date, "%Y-%m-%d").timestamp() or 0.0
    _ed = end_date and datetime.strptime(end_date, "%Y-%m-%d").timestamp() or 0.0
    from_filter = f"and o.timestamp >= {int(_sd) * 1000}" if start_date else ""
    to_filter = f"and o.timestamp <= {int(_ed) * 1000}" if end_date else ""
    q = f"""select s.ticker as symbol,                
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            from ohlcv_1d o left join symbol s
            on o.symbol_id = s.id
            where s.ticker = UPPER('{symbol}')
            {from_filter}
            {to_filter}
            """
    res = cur.execute(q)
    data = res.fetchall()
    con.close()
    columns = pd.Index(
        [
            "Symbol",
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )
    df = pd.DataFrame(data, columns=columns)
    df = df.assign(Date=pd.to_datetime(df["Timestamp"], unit="ms"))
    df = df.set_index("Date").drop(columns=["Timestamp"])
    return df


def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "Returns"] = np.log(df["Close"] / df["Close"].shift(1)).round(4)
    return df.dropna()


def get_ols_fit_model(y, x):
    if y.empty or x.empty or len(y) != len(x):
        # Return a dummy model or raise
        raise ValueError("Empty or mismatched data for OLS")
    y = np.log(y).astype(float)
    x = np.log(x).astype(float)
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()


def hedge_ratio_residuals(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Returns residuals from OLS(y ~ x) using log prices."""
    model = get_ols_fit_model(y, x)
    alpha, beta = model.params
    return y - (alpha + beta * x)


def calculate_zscore_spread(s1, s2):
    """Calculate z-score normalized spread"""
    if s1.empty or s2.empty or len(s1) != len(s2):
        return pd.Series(dtype=float)
    model = get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    scaled_s2 = alpha + beta * s2
    spread_series = s1 - scaled_s2
    if spread_series.empty:
        return pd.Series(dtype=float)
    return (spread_series - spread_series.mean()) / spread_series.std()


def eg_pvalue(price1, price2):
    """Engle-Granger p-value in one direction: price1 ~ price2"""
    resid = hedge_ratio_residuals(price1, price2)
    return adfuller(resid)[1]  # p-value


def symmetric_cointegration_p(price1, price2):
    """
    Proper symmetric cointegration:
    - compute p-values in both directions
    - take the minimum — the statistically valid symmetric measure
    """
    p1 = float(eg_pvalue(price1, price2))
    p2 = float(eg_pvalue(price2, price1))
    # return min(p1, p2)  # recommended
    return float((p1 + p2) / 2)


def pick(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k not in keys}


def validate_schema(data, schema):
    """
    Validate complex nested data structures
    """

    def check_type(value, expected_type):
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)

        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                return False
            return all(
                key in value and check_type(value[key], type_check)
                for key, type_check in expected_type.items()
            )

        return False

    return check_type(data, schema)


def get_ts(ds: str) -> pd.Timestamp:
    _ts = pd.Timestamp(ds)
    if isinstance(_ts, pd.Timestamp):
        return _ts

    raise ValueError(f"Failed when creating timestamp for {ds}")
