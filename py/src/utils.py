import pandas as pd
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
