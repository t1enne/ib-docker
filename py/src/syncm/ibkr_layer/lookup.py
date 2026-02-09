from src.syncm.ibkr_layer.shared import client


def _is_nasdaq_stock(entry: dict):
    if not isinstance(entry, dict):
        return False

    company_header = entry.get("companyHeader") or "".upper()
    description = entry.get("description") or "".upper()
    sections = entry.get("sections", [])

    # Check if it's a stock (has STK section)
    has_stock = any(s.get("secType") or "" == "STK" for s in sections)
    if not has_stock:
        return False

    # Check for NASDAQ indicator in header or description
    is_nasdaq = "NASDAQ" in company_header or "NASDAQ" in description
    return is_nasdaq


async def lookup(ticker: str) -> int:
    try:
        r = await client.get("iserver/secdef/search", params={"symbol": ticker})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            print(data)
            raise ValueError(f"Unexpected result")

        # Filter for NASDAQ USD stocks
        nasdaq_usd_stocks = list(filter(_is_nasdaq_stock, data))
        if len(nasdaq_usd_stocks) == 0:
            print(f"Raw data for debugging: {data}")  # Optional: log for debugging
            raise ValueError(f"No NASDAQ USD stock found for {ticker}")

        if len(nasdaq_usd_stocks) > 1:
            print(
                f"Multiple NASDAQ USD stocks found for {ticker}: {[e.get('conid') for e in nasdaq_usd_stocks]}"
            )
            print("Raw data:", data)
            raise ValueError(f"Multiple NASDAQ USD stocks found for {ticker}")

        return int(nasdaq_usd_stocks[0]["conid"])

    except Exception as e:
        raise ValueError(f"Failed to search contract for {ticker}: {e}")
    raise ValueError(f"No contract found for {ticker}")


__all__ = ["lookup"]
