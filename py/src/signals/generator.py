"""Signal generator - runs strategy logic against live bar data."""

import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Any, List, Optional, cast

import click
import pandas as pd

from src.bt.algos import init_strat
from src.bt.engine.utils import merge_bt_state
from src.bt.state import (
    BacktestState,
    Tick,
    TradeSignal,
    create_initial_backtest_state,
)
from src.bt.types import StrategyConfig
from src.signals.feed import LiveBarFeed
from src.signals.types import SignalEvent
from src.syncm.ibkr_layer import lookup
from src.utils import get_local_candles

logger = logging.getLogger(__name__)

# Number of historical bars to preload for indicator warmup
HISTORY_BUFFER_BARS = 500


async def _resolve_symbols(tickers: list[str]) -> dict[str, int]:
    """Resolve ticker symbols to IBKR conids.

    Uses the existing syncm lookup layer which checks the local DB first,
    then falls back to the IBKR API.
    """
    from src.db.models import SymbolSchema

    result: dict[str, int] = {}
    for ticker in tickers:
        # Check local DB first
        s = SymbolSchema.get_or_none(SymbolSchema.ticker == ticker.upper())
        if s:
            result[ticker.upper()] = s.conid
            continue

        # Fall back to API lookup
        contract = await lookup(ticker)
        conid = int(contract.conid or "-")
        result[ticker.upper()] = conid

    return result


def _load_historical_candles(symbols: list[str], bar: str) -> dict[str, pd.DataFrame]:
    """Load historical candles from local DB for indicator warmup.

    Returns a dict mapping symbol to per-symbol DataFrame with DatetimeIndex.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = get_local_candles(symbol, bar=bar)
        if df.empty:
            logger.warning("No local candle data for %s", symbol)
            continue

        # Take last HISTORY_BUFFER_BARS rows
        df = df.tail(HISTORY_BUFFER_BARS)

        # Drop the 'symbol' column if present
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        result[symbol.upper()] = df[["open", "high", "low", "close", "volume"]]

    return result


def _append_candle(state: BacktestState, tick: Tick) -> BacktestState:
    """Append a candle to the per-symbol candles dict."""
    new_row = pd.DataFrame(
        {
            "open": [tick.open],
            "high": [tick.high],
            "low": [tick.low],
            "close": [tick.close],
            "volume": [tick.volume],
        },
        index=[tick.timestamp],
    )

    candles = dict(state.candles)
    current = candles.get(tick.symbol)
    if current is None or current.empty:
        candles[tick.symbol] = new_row
    else:
        candles[tick.symbol] = pd.concat([current, new_row])

    return merge_bt_state(state, dict(candles=candles))


def _signal_to_event(signal: TradeSignal, strategy_name: str) -> SignalEvent:
    """Convert a TradeSignal from strategy to a printable SignalEvent."""
    return SignalEvent(
        timestamp=signal.timestamp.to_pydatetime(),
        symbol=signal.symbol,
        action=signal.action.value,
        price=signal.price,
        reason=str(signal.reason) if signal.reason else "",
        strategy_name=strategy_name,
        z_score=signal.z_score,
    )


class SignalGenerator:
    """Generates trading signals from live IBKR bar data using existing strategies.

    Bootstraps with historical candle data for indicator warmup, then
    subscribes to live bars via websocket and runs the strategy on_tick()
    for each new bar.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strat_mod = init_strat(config.strategy_type)
        if self.strat_mod is None:
            raise ValueError(
                f"Unknown strategy type: {config.strategy_type}. "
                f"Check strategy_type in your YAML config."
            )
        self.state: Optional[BacktestState] = None
        self.feed: Optional[LiveBarFeed] = None
        self._seen_timestamps: dict[str, pd.Timestamp] = {}

    async def _bootstrap(self) -> dict[str, int]:
        """Resolve symbols and load historical data. Returns {ticker: conid}."""
        symbols = [s.upper() for s in self.config.symbols]

        # Resolve to conids
        click.echo("Resolving symbols...")
        symbol_conids = await _resolve_symbols(symbols)
        for ticker, conid in symbol_conids.items():
            click.echo(f"  {ticker} -> conid {conid}")

        # Try to load historical candles for warmup
        click.echo(f"Loading historical candles (last {HISTORY_BUFFER_BARS} bars)...")
        candles = _load_historical_candles(symbols, self.config.bar)

        for symbol in symbols:
            sym_df = candles.get(symbol.upper())
            if sym_df is not None and not sym_df.empty:
                click.echo(f"  {symbol}: {len(sym_df)} bars loaded")
            else:
                click.echo(
                    f"  {symbol}: no data (indicators may be inaccurate until buffer fills)"
                )

        # Build initial state with pre-loaded candles
        now = pd.Timestamp.now()
        self.state = create_initial_backtest_state(
            symbols=symbols,
            initial_capital=self.config.initial_capital,
            start_timestamp=now,
            rolling_window_size=self.config.rolling_window_size,
        )

        # Replace empty candles with historical data
        if candles:
            self.state = merge_bt_state(self.state, dict(candles=candles))

        return symbol_conids

    def _process_tick(self, tick: Tick) -> list[SignalEvent]:
        """Process a single tick through the strategy and return any signals."""
        assert self.state is not None

        # Skip duplicate timestamps (ws may resend the current bar)
        last_ts = self._seen_timestamps.get(tick.symbol)
        if last_ts is not None and tick.timestamp <= last_ts:
            return []
        self._seen_timestamps[tick.symbol] = tick.timestamp

        # Append candle to state
        self.state = _append_candle(self.state, tick)
        self.state = merge_bt_state(self.state, dict(timestamp=tick.timestamp))

        # Run strategy
        strategy_params = dict(self.config.strategy_params or {})
        if self.config.rolling_window_size is not None:
            strategy_params.setdefault(
                "rolling_window_size", self.config.rolling_window_size
            )
        if self.config.symbols:
            strategy_params.setdefault("symbols", list(self.config.symbols))

        try:
            trade_signals: List[TradeSignal] = self.strat_mod.on_tick(
                self.state, tick, strategy_params
            )
        except Exception as e:
            logger.error("Strategy error on tick %s: %s", tick.timestamp, e)
            return []

        return [_signal_to_event(s, self.config.name) for s in trade_signals]

    async def start(self) -> None:
        """Main daemon loop. Connects to ws, receives bars, generates signals."""
        symbol_conids = await self._bootstrap()

        # Print startup banner
        click.echo("")
        click.echo("Signal generator started")
        click.echo(f"  Strategy: {self.config.name} ({self.config.strategy_type})")
        for ticker, conid in symbol_conids.items():
            click.echo(f"  Symbol:   {ticker} (conid: {conid})")
        click.echo(f"  Bar:      {self.config.bar}")
        click.echo("")
        click.echo("Waiting for bars...")
        click.echo("")

        # Set up live feed
        self.feed = LiveBarFeed(symbol_conids, self.config.bar)

        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def _shutdown():
            click.echo("\nShutting down...")
            shutdown_event.set()
            exit(0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

        try:
            async for tick in self.feed.ticks():
                if shutdown_event.is_set():
                    break

                events = self._process_tick(tick)
                for event in events:
                    click.echo(str(event))

        finally:
            if self.feed:
                await self.feed.close()
            click.echo("Signal generator stopped.")


async def generate_signals(config: StrategyConfig) -> None:
    """Entry point for signal generation. Called from CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    gen = SignalGenerator(config)
    await gen.start()
