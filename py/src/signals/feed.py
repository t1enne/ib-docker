"""Live bar feed via IBKR websocket (smh subscription)."""

from src.syncm.ibkr_layer.shared import client

import asyncio
import json
import logging
import ssl
from typing import AsyncGenerator, Optional

import pandas as pd
import websockets
import websockets.asyncio.client

from src.bt.state import Candle
from typing import cast

logger = logging.getLogger(__name__)

WS_URL = "wss://localhost:5000/v1/api/ws"
KEEPALIVE_INTERVAL_S = 30
RECONNECT_BASE_DELAY_S = 1
RECONNECT_MAX_DELAY_S = 30


def _parse_bar_message(
    msg: dict, conid_to_ticker: dict[int, str], bar: str
) -> Optional[Candle]:
    """Parse an smh websocket message into a Candle (OHLCV bar).

    IBKR smh messages have the shape:
    {
        "topic": "smh+265598",
        "conid": 265598,
        "t": 1711720800000,   # timestamp in ms
        "o": 512.10,          # open
        "h": 513.50,          # high
        "l": 511.80,          # low
        "c": 512.90,          # close
        "v": 1234567          # volume
    }

    They may also arrive as arrays in a "data" field.
    """
    topic = msg.get("topic", "")
    if not topic.startswith("smh+"):
        return None

    conid = msg.get("conid")
    if conid is None:
        # Extract conid from topic: "smh+265598"
        try:
            conid = int(topic.split("+")[1])
        except IndexError, ValueError:
            return None

    ticker = conid_to_ticker.get(conid)
    if ticker is None:
        return None

    ts = msg.get("t")
    if ts is None:
        return None

    try:
        pdt = cast(pd.Timestamp, pd.Timestamp(ts, unit="ms"))
        assert not pd.isna(pdt)
        return Candle(
            timestamp=pdt,
            symbol=ticker,
            open=float(msg.get("o", 0)),
            high=float(msg.get("h", 0)),
            low=float(msg.get("l", 0)),
            close=float(msg.get("c", 0)),
            volume=float(msg.get("v", 0)),
            interval=bar,
        )
    except (TypeError, ValueError) as e:
        logger.warning("Failed to parse bar message: %s (%s)", msg, e)
        return None


def _parse_bar_array_message(
    msg: dict, conid_to_ticker: dict[int, str], bar: str
) -> list[Candle]:
    """Parse an smh message that contains an array of bars."""
    topic = msg.get("topic", "")
    if not topic.startswith("smh+"):
        return []

    conid = msg.get("conid")
    if conid is None:
        try:
            conid = int(topic.split("+")[1])
        except IndexError, ValueError:
            return []

    ticker = conid_to_ticker.get(conid)
    if ticker is None:
        return []

    data = msg.get("data", [])
    bars: list[Candle] = []
    for bar_data in data:
        ts = bar_data.get("t")
        if ts is None:
            continue
        try:
            pdt = cast(pd.Timestamp, pd.Timestamp(ts, unit="ms"))
            assert not pd.isna(pdt)
            bars.append(
                Candle(
                    timestamp=pdt,
                    symbol=ticker,
                    open=float(bar_data.get("o", 0)),
                    high=float(bar_data.get("h", 0)),
                    low=float(bar_data.get("l", 0)),
                    close=float(bar_data.get("c", 0)),
                    volume=float(bar_data.get("v", 0)),
                    interval=bar,
                )
            )
        except (TypeError, ValueError) as e:
            logger.warning("Failed to parse bar in array: %s (%s)", bar_data, e)
    return bars


class LiveBarFeed:
    """Websocket-based live bar feed using IBKR smh subscriptions.

    Connects to the IBKR Client Portal websocket and subscribes to
    streaming historical bar data for the given symbols.
    """

    def __init__(self, symbols: dict[str, int], bar: str):
        """
        Args:
            symbols: {ticker: conid} mapping
            bar: bar size, e.g. "1h", "1d"
        """
        self.symbols = symbols
        self.bar = bar
        self.conid_to_ticker = {conid: ticker for ticker, conid in symbols.items()}
        self._ws: Optional[websockets.asyncio.client.ClientConnection] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._running = True

    async def auth_ws(self, session_id: str):
        if not self._ws:
            raise ValueError("WS not initialized")

        logger.info("Authenticating WS")
        await self._ws.send(f'{{ "session": {session_id} }}')
        msg = await self._ws.recv()
        print(msg)

    async def connect(self) -> None:
        """Open websocket connection and subscribe to bar data."""
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        logger.info("Connecting to %s", WS_URL)
        self._ws = await websockets.asyncio.client.connect(
            WS_URL,
            ssl=ssl_ctx,
            additional_headers={"Origin": "https://localhost:5000"},
        )

        tickle_result = await client.get("/tickle")
        session_id = tickle_result.json().get("session")
        await self.auth_ws(session_id)

        self._running = True

        # Start keepalive
        self._keepalive_task = asyncio.create_task(self._keepalive())

        # Subscribe to smh for each symbol
        for ticker, conid in self.symbols.items():
            sub_msg = f'smd+{conid}+{{"fields":["31","83"]}}'
            logger.info("Subscribing: %s (%s)", ticker, sub_msg)
            await self._ws.send(sub_msg)

        # try:
        #     async for message in self._ws:
        #         data = json.loads(message)
        #         print("Received:", data)
        # except websockets.exceptions.ConnectionClosed:
        #     print("Disconnected")

    async def _keepalive(self) -> None:
        """Send periodic tic keepalive messages."""
        while self._running:
            try:
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                if self._ws:
                    await self._ws.send("tic")
            except Exception:
                break

    async def bars(self) -> AsyncGenerator[Candle, None]:
        """Yield Candle objects as new bars arrive from the websocket.

        Handles reconnection on disconnect with exponential backoff.
        """
        delay = RECONNECT_BASE_DELAY_S

        while self._running:
            try:
                if self._ws is None:
                    await self.connect()
                assert self._ws is not None
                async for raw in self._ws:
                    delay = RECONNECT_BASE_DELAY_S  # reset on successful message

                    try:
                        msg = json.loads(raw)
                        print(msg)
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON message: %s", raw[:100])
                        continue

                    topic = msg.get("topic", "")

                    # Skip non-smh messages (system, status, etc.)
                    if not topic.startswith("smh+"):
                        if topic:
                            logger.debug("Skipping topic: %s", topic)
                        continue

                    # Try single bar format first
                    bar = _parse_bar_message(msg, self.conid_to_ticker, self.bar)
                    if bar is not None:
                        yield bar
                        continue

                    # Try array format
                    bars = _parse_bar_array_message(
                        msg, self.conid_to_ticker, self.bar
                    )
                    for bar in bars:
                        yield bar

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("WebSocket disconnected: %s", e)
                self._ws = None
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                    self._keepalive_task = None
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                self._ws = None
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                    self._keepalive_task = None

            if not self._running:
                break

            logger.info("Reconnecting in %ds...", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY_S)

    async def close(self) -> None:
        """Close the websocket connection."""
        self._running = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._ws:
            # Unsubscribe from smh feeds
            for conid in self.symbols.values():
                try:
                    await self._ws.send(f"umh+{conid}")
                except Exception:
                    pass
            await self._ws.close()
            self._ws = None
