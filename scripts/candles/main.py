#!/usr/bin/env python3
"""
Interactive Brokers Client Portal API - Candlestick Data Fetcher
This script fetches OHLCV candlestick data for specified symbols and time ranges.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class Security:
    conid: int
    symbol: str
    sections: List[Dict[str, str]]
    secType: str
    description: str
    company_name: str


class CandlesFetcher:
    def __init__(self, base_url="https://localhost:5000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1/api"

    def search_securities(self, symbol) -> List[Security]:
        """Search for securities and return available options."""
        endpoint = "iserver/secdef/search"
        url = f"{self.api_base}/{endpoint}"

        params = {"symbol": symbol}

        try:
            response = requests.get(url, params=params, verify=False, timeout=10)
            response.raise_for_status()

            data = response.json()
            print(data)

            error = "error" in data and data.get("error")
            if error:
                print(f"failed to fetch symbols: {error}")
                return []

            if len(data) == 0:
                return []

            securities = []
            for item in data:
                print(item)
                # Extract secType from either root or sections[0]
                if "secType" in item:
                    sec_type = item["secType"]
                elif item.get("sections") and "secType" in item["sections"][0]:
                    sec_type = item["sections"][0]["secType"]
                else:
                    raise BaseException("unrecognized securitytype")

                security = Security(
                    conid=item.get("conid"),
                    symbol=item.get("symbol"),
                    sections=item.get("sections", []),
                    secType=sec_type,
                    description=item.get("description", ""),
                    company_name=item.get("companyHeader", ""),
                )
                securities.append(security)

            return securities

        except requests.exceptions.RequestException as e:
            print(f"Error searching for {symbol}: {e}")
            return []

    def get_contract_id(
        self, securities: List[Security], sec_type: str
    ) -> Security | None:
        """Get contract ID for a symbol with optional security type filter."""

        if not securities:
            print("No securities found for")
            return None

        if sec_type:
            filtered = [s for s in securities if s.secType == sec_type]
            if filtered:
                return filtered[0]

        if len(securities) == 1:
            return securities[0]

        print(f"\nFound {len(securities)} securities:")
        print("-" * 50)
        for i, sec in enumerate(securities, 1):
            conid = sec.conid
            symbol = sec.symbol
            sec_type = sec.secType
            exchange = sec.description or ""
            company_name = sec.company_name
            print(f"{i}. *{symbol}* {company_name} _{sec_type}_ ({exchange}) [{conid}]")

        while True:
            try:
                choice = input(f"\nSelect security (1-{len(securities)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(securities):
                    return securities[idx]
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nOperation cancelled.")
                return None

    def fetch_candles(
        self,
        conid,
        interval="1d",
        period="30d",
        start_date="20231018-16:00:00",
        end_date=None,
    ):
        """Fetch candlestick data for a symbol."""

        endpoint = "iserver/marketdata/history"
        url = f"{self.api_base}/{endpoint}"

        params = {"conid": conid, "period": period, "bar": interval}

        if start_date:
            start_ts = int(
                datetime.strptime(start_date, "%Y%m%d-%H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )
            params["startTime"] = start_ts

        if end_date:
            end_ts = int(
                datetime.strptime(start_date, "%Y%m%d-%H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )
            params["endTime"] = end_ts
        try:
            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "data" not in data:
                print(f"No data returned for {conid}")
                return None

            candles_data = data["data"]
            df = pd.DataFrame(
                [
                    {
                        "Date": datetime.fromtimestamp(
                            candle["t"] / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "Open": candle.get("o", 0),
                        "High": candle.get("h", 0),
                        "Low": candle.get("l", 0),
                        "Close": candle.get("c", 0),
                        "Volume": candle.get("v", 0),
                    }
                    for candle in candles_data
                ]
            )

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching candles for {conid}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV candlestick data from IBKR"
    )
    parser.add_argument("symbol", help="Stock symbol to fetch data for")
    parser.add_argument(
        "-i", "--interval", default="1d", help="Time interval (default: 1d)"
    )
    parser.add_argument("-s", "--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("-t", "--sec-type", help="Security type (STK, OPT, BOND, etc.)")
    parser.add_argument("-o", "--output", help="Output CSV dir")
    parser.add_argument("-p", "--period", help="Time away from startTime. period=6d")

    args = parser.parse_args()
    fetcher = CandlesFetcher()

    print(f"Searching for {args.symbol}...")

    securities = fetcher.search_securities(args.symbol)
    contract = fetcher.get_contract_id(securities, args.sec_type)
    if not contract:
        print(f"Unable to get contract for symbol: {args.symbol}")
        return None

    df = fetcher.fetch_candles(
        conid=contract.conid,
        interval=args.interval,
        period=args.period,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if df is not None and not df.empty:
        print(f"\nFetched {len(df)} candles:")
        print(df.to_string(index=False))
        filename = f"{args.symbol}_{contract.secType}_{args.interval}"
        filepath = f"{args.output}/{filename}.csv"

        if args.output:
            df.to_csv(filepath, index=False)
            print(f"\nData saved to: {filepath}")
    else:
        print("No data retrieved")
        sys.exit(1)


if __name__ == "__main__":
    main()
