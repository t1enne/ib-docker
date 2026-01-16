from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime
from ..consts.interval import BAR_INTERVAL
from .shared import client, get_contract_info
from ..db.models import get_ohlcv_model
from ..db import db

class ArgsSchema(BaseModel):
    conid: int
    bar: str = Field(..., pattern='|'.join(BAR_INTERVAL))
    period: Optional[str] = Field(None, pattern=r'\d+d$')
    startTime: Optional[str] = None

async def candles(
    conid: int,
    period: Optional[str] = None,
    bar: str = "1d",
    startTime: Optional[str] = None,
):
    # Validate
    ArgsSchema(conid=conid, period=period, bar=bar, startTime=startTime)
    
    symbol_info = await get_contract_info(conid)
    print(f"Getting candles for {symbol_info.ticker}")
    
    table_name = f"ohlcv_{bar}"
    model = get_ohlcv_model(bar)
    
    params = {
        "conid": conid,
        "bar": bar,
        "period": period,
    }
    if startTime:
        dt = datetime.datetime.fromisoformat(startTime.replace('Z', '+00:00'))
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
            }
        ).execute()