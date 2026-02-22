from typing import Optional
import pandas as pd
from pandas import Timestamp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import read_candles, parse_timestamp
from src.bt.zscore import calculate_rolling_z


def spread(
    symbols: tuple[str, str],
    start_date: Optional[Timestamp] = None,
    end_date: Optional[Timestamp] = None,
    rolling: Optional[int] = None,
):
    sym1, sym2 = symbols
    ma_windows = [14]
    _start = parse_timestamp(start_date) if start_date else None
    _end = parse_timestamp(end_date) if end_date else None
    df1 = read_candles(sym1.upper(), _start, _end)
    df2 = read_candles(sym2.upper(), _start, _end)

    prices1 = df1["Close"].tolist()
    prices2 = df2["Close"].tolist()
    dates = df1.index

    ratio = pd.DataFrame(
        {
            "ratio": df1["Close"] / df2["Close"],
            f"{sym1}": df1["Close"],
            f"{sym2}": df2["Close"],
        },
        index=dates,
    )

    if rolling:
        window = rolling

        z_scores: list[float] = []
        for i in range(len(prices1)):
            s1 = prices1[: i + 1]
            s2 = prices2[: i + 1]
            z, _, _ = calculate_rolling_z(s1, s2, window)
            z_scores.append(z)

        z_score = pd.Series(z_scores, index=dates)
    else:
        z, _, _ = calculate_rolling_z(prices1, prices2, len(prices1))
        z_score = pd.Series([z], index=[dates[-1]])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["z", "ratio"],
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
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
                x=dates,
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

    output_file = f"plots/spread_{sym1}_{sym2}.html"
    fig.write_html(output_file)
    print(f"Saved to {output_file}")
