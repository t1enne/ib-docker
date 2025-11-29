import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import sqlite3


def read_candles(symbol: str):
    # return pd.read_csv(f"mdata/{symbol}.csv")
    con = sqlite3.connect("../data/db.sqlite")
    cur = con.cursor()
    res = cur.execute(
        f"""select s.ticker as symbol,
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            from ohlcv_1d o left join symbol s
            on o.symbol_id = s.id
            where s.ticker = '{symbol}'"""
    )
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
    df = df.set_index("Timestamp")
    return df


def candle(symbol: str, interval="1d", start="2023-01-01", end="2025-01-01"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, start=start, end=end)
    df.to_csv(f"{symbol}.csv")


def plot_corr(merged: pd.DataFrame, symbol: str):
    merged["Spread"] = merged["Return_stock"] - merged["Return_bench"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["Return_stock"],
            mode="lines",
            name=f"{symbol} Return",
        )
    )
    # Plot raw returns for IWM
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["Return_bench"],
            mode="lines",
            name="IWM Return",
            yaxis="y2",
        )
    )

    # Plot spread
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["Spread"],
            mode="lines",
            name="Return Spread",
            yaxis="y3",
        )
    )

    # Update layout for multiple y-axes
    fig.update_layout(
        title=f"{symbol} Returns, Benchmark IWM Returns, and Spread",
        yaxis=dict(title="Returns", side="left"),
        yaxis2=dict(title="IWM Returns", overlaying="y", side="right"),
        yaxis3=dict(
            title="Spread", anchor="x", overlaying="y", side="right", position=0.85
        ),
        legend=dict(x=0.01, y=0.99),
    )

    fig.show()


def calculate_cointegration(price1: pd.Series, price2: pd.Series) -> float:
    # Perform OLS regression: price1 = a + b * price2
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    residuals = model.resid
    # ADF test on residuals
    adf_result = adfuller(residuals)
    p_value = float(adf_result[1])
    return p_value


def get_metrics(df: pd.DataFrame, benchmark: pd.DataFrame, symbol: str):
    # Merge on Date
    merged = pd.merge(
        df[["Date", "Close"]],
        benchmark[["Date", "Close"]],
        on="Date",
        suffixes=("_stock", "_bench"),
    )
    # Set Date as index
    merged.set_index("Date", inplace=True)
    # Calculate daily log returns
    merged.loc[:, "Return_stock"] = np.log(
        merged["Close_stock"] / merged["Close_stock"].shift(1)
    ).round(4)
    merged.loc[:, "Return_bench"] = np.log(
        merged["Close_bench"] / merged["Close_bench"].shift(1)
    ).round(4)
    # Drop NaN
    merged = merged.dropna()
    # Correlation of log returns
    corr = np.corrcoef(merged["Return_stock"], merged["Return_bench"])[0, 1]
    # Cointegration p-value
    p_value = calculate_cointegration(merged["Close_stock"], merged["Close_bench"])
    return corr, p_value


def populate_df(df: pd.DataFrame, rf: float):
    # Calculate daily returns
    df.loc[:, "Daily_Return"] = df["Close"].pct_change()
    # Calculate excess returns
    df.loc[:, "Excess_Return"] = df["Daily_Return"] - rf
    # Calculate Sharpe ratio
    mean_excess = df["Excess_Return"].mean()
    std_excess = df["Excess_Return"].std()
    sharpe_ratio = (mean_excess / std_excess) * np.sqrt(252)
    return sharpe_ratio


def main():
    start = "2023-01-01"
    end = "2025-01-01"
    benchmark = "MSFT"
    syms = ["AMZN"]
    correlations = []
    cointegrations = []
    benchmark_df = read_candles(benchmark)
    for ticker in syms:
        corr, p_value = get_metrics(read_candles(ticker), benchmark_df, ticker)
        correlations.append(corr)
        cointegrations.append(p_value)

    results = pd.DataFrame(
        {"Stocks": syms, "Correlation": correlations, "Cointegration": cointegrations}
    )
    print(results)


if __name__ == "__main__":
    main()
