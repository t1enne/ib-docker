from ib_rest_api_client.models import (
    GetIserverSecdefSearchSecType,
    ErrorOnlyResponse,
    SecdefSearchResponseItem,
)
from ib_rest_api_client.api.trading_contracts import get_iserver_secdef_search
from src.syncm.ibkr_layer.shared import client, auth_client
from pprint import pprint


def _is_usd_stock(entry: SecdefSearchResponseItem):
    return entry.description in ["ARCA", "NASDAQ"]


async def lookup(ticker: str) -> SecdefSearchResponseItem:
    try:
        # r = await client.get("iserver/secdef/search", params={"symbol": ticker})
        r = await get_iserver_secdef_search.asyncio(
            client=auth_client,
            symbol=ticker,
        )
        if isinstance(r, ErrorOnlyResponse):
            raise ValueError(f"Failed to search contract for {ticker}")
        if not isinstance(r, list):
            raise ValueError(f"Failed to search contract for {ticker}")

        print(r)
        data = [item for item in r if _is_usd_stock(item)]
        first = data[0]
        if not data or not first:
            raise ValueError(f"No contract found for {ticker}")

        return first

    except Exception as e:
        raise ValueError(f"Failed to search contract for {ticker}: {e}")
    raise ValueError(f"No contract found for {ticker}")


__all__ = ["lookup"]
