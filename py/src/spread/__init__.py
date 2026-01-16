from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import read_candles, get_ols_fit_model


def spread(
    symbols: tuple[str, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rolling: Optional[int] = None,
):
    sym1, sym2 = symbols
    ma_windows = [9, 14, 50]
    df1 = read_candles(sym1.upper(), start_date, end_date)
    df2 = read_candles(sym2.upper(), start_date, end_date)
    ratio = pd.DataFrame(
        {
            "ratio": df1["Close"] / df2["Close"],
            f"{sym1}": df1["Close"],
            f"{sym2}": df2["Close"],
        },
        index=df1.index,
    )

    # Estimate beta from the last rolling points (or all if no rolling)
    window_size = rolling if rolling else len(df1)
    tail1 = df1["Close"].tail(window_size)
    tail2 = df2["Close"].tail(window_size)
    model = get_ols_fit_model(tail1, tail2)
    _, beta = model.params

    # Calculate spreads
    spreads = df1["Close"] - beta * df2["Close"]

    # Calculate rolling z-score on spreads
    if rolling:
        # Use fixed beta and rolling z-score on spreads
        rolling_mean = spreads.rolling(window=rolling, min_periods=rolling).mean()
        rolling_std = spreads.rolling(window=rolling, min_periods=rolling).std()
        z_score = (spreads - rolling_mean) / rolling_std
    else:
        # For consistency, even without rolling, use fixed beta approach
        z_score = (spreads - spreads.mean()) / spreads.std()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["z", "ratio"],
        vertical_spacing=0.05,  # Reduce vertical spacing
        horizontal_spacing=0.05,  # Reduce horizontal spacing
    )
    fig.add_trace(
        go.Scatter(
            x=df1.index,
            y=z_score,
            mode="lines",
            name=f"z-score {sym1}-{sym2}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio["ratio"],
            mode="lines",
            name="ratio",
            hovertemplate=f"{sym1}: %{{customdata[0]:.2f}}<br>{sym2}: %{{customdata[1]:.2f}}<br>Ratio: %{{y:.4f}}<extra></extra>",
            customdata=ratio[[f"{sym1}", f"{sym2}"]].values,
        ),
        row=2,
        col=1,
    )

    for window in ma_windows:
        z_score_ema = z_score.ewm(span=window).mean()
        ratio_ema = ratio["ratio"].ewm(span=window).mean()
        fig.add_trace(
            go.Scatter(
                x=df1.index,
                y=z_score_ema,
                mode="lines",
                name=f"z-score EMA {window}",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio_ema,
                mode="lines",
                name=f"ratio EMA {window}",
                line=dict(width=1),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(type="date")
    fig.show()
