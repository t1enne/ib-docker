import httpx
from typing import Dict, Any
from src.db.models import Symbol

client = httpx.AsyncClient(
    base_url="https://localhost:5000/v1/api/", timeout=10.0, verify=False
)


async def fetch_contract_info(conid: int) -> Dict[str, Any]:
    ep = f"iserver/contract/{conid}/info"
    try:
        r = await client.get(ep)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise ValueError(f"Failed call to {ep}: {e}")


async def search_contract(ticker: str) -> int:
    try:
        r = await client.get("iserver/secdef/search", params={"symbol": ticker})
        r.raise_for_status()
        data = r.json()
        if data:
            return int(data[0]["conid"])
    except Exception as e:
        raise ValueError(f"Failed to search contract for {ticker}: {e}")
    raise ValueError(f"No contract found for {ticker}")


async def get_contract_info(conid: int) -> Symbol:
    # Query DB first
    try:
        symbol = Symbol.get(Symbol.id == conid)
        return symbol
    except Symbol.DoesNotExist:
        pass

    # Fetch from API
    cinfo = await fetch_contract_info(conid)

    # Insert into DB
    symbol = Symbol.create(
        id=conid,
        ticker=cinfo["symbol"],
        name=cinfo.get("company_name"),
        market=cinfo["exchange"],
        currency=cinfo["currency"],
    )
    return symbol
