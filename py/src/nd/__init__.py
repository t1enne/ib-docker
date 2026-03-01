import plotly.express as px
from src.utils import get_log_returns, get_local_candles


def nd(symbol: str, ma: int):
    df = get_log_returns(get_local_candles(symbol.upper()))
    col = "close"
    series = df[col]
    rolling_series = series.rolling(window=ma).mean()
    deviation = series - rolling_series
    z_score = (deviation - deviation.rolling(window=ma).mean()) / deviation.rolling(
        window=ma
    ).std()

    fig = px.line(
        x=df.index,
        y=z_score,
        title=f"z-score of {symbol}-{ma}MA",
    )
    fig.show()
