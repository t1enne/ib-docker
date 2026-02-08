import datetime
import re
from typing import Optional
from src.consts import BAR_INTERVAL
from src.db.models import get_ohlcv_model
from src.db import db

from .shared import client, get_contract_info


def validate_candles_args(conid, period, bar, startTime):
    if not isinstance(conid, int):
        raise ValueError("conid must be int")
    if bar not in BAR_INTERVAL:
        raise ValueError(f"bar must be one of {BAR_INTERVAL}")
    if period is not None:
        if not isinstance(period, str) or not re.match(r"\d+d$", period):
            raise ValueError(r"period must be string matching \d+d$")
    if startTime is not None and not isinstance(startTime, str):
        raise ValueError("startTime must be str or None")


async def candles(
    conid: int,
    period: Optional[str] = None,
    bar: str = "1d",
    startTime: Optional[str] = None,
):
    # Validate
    validate_candles_args(conid, period, bar, startTime)

    symbol_info = await get_contract_info(conid)
    print(f"Getting candles for {symbol_info.ticker}")

    model = get_ohlcv_model(bar)

    params = {
        "conid": conid,
        "bar": bar,
        "period": period,
    }
    if startTime:
        dt = datetime.datetime.fromisoformat(startTime.replace("Z", "+00:00"))
        params["startTime"] = dt.strftime("%Y%m%d-%H:%M:%S")

    print(f"Getting candles with params: {params}")

    try:
        r = await client.get("iserver/marketdata/history", params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"Failed getting marketdata: {e}")
        return

    if not data:
        print("No candles found")
        return

    print(f"Inserting {len(data)} candles")

    insert_data = [
        {
            "symbol_id": symbol_info.id,
            "timestamp": item["t"],
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"],
            "volume": item["v"],
        }
        for item in data
    ]

    with db.atomic():
        model.insert_many(insert_data).on_conflict(
            conflict_target=(model.timestamp, model.symbol_id),
            update={
                model.open: model.open,
                model.high: model.high,
                model.low: model.low,
                model.close: model.close,
                model.volume: model.volume,
            },
        ).execute()
