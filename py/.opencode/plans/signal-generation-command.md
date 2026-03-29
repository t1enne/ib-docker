# Signal Generation Command - Implementation Plan

## Goal

Add a `signal` CLI command that runs as a daemon, subscribes to live IBKR market data via websocket (smh bar streaming), feeds candles through existing strategy on_tick() logic, and prints trading signals to console.

## Architecture

```
main.py signal strats/breakout_ema.yaml
  -> load_strategy() -> StrategyConfig
  -> SignalGenerator(config)
       -> resolve symbols to conids (syncm.ibkr_layer.lookup)
       -> load historical candle buffer from local DB (or sync if missing)
       -> init strategy module via init_strat()
       -> build initial BacktestState (candles preloaded, dummy portfolio)
  -> LiveBarFeed (websocket)
       -> open ws to wss://localhost:5000/v1/api/ws
       -> subscribe smh+{conid}+{"period":"2d","bar":"1h"} per symbol
       -> receive streaming OHLCV bar updates -> parse into Tick
       -> keepalive tic every 30s
  -> Strategy Loop
       -> for each bar from ws:
            append candle to BacktestState
            call strat_mod.on_tick(state, tick, strategy_params)
            print any TradeSignal to console
            update state
       -> auto-reconnect on ws disconnect
       -> graceful shutdown on SIGINT/SIGTERM
```

## Files to Create

### 1. src/signals/__init__.py
Module entry. Exports generate_signals.

### 2. src/signals/feed.py - LiveBarFeed
WS connection + bar subscription.
- Connect wss://localhost:5000/v1/api/ws (websockets library)
- Subscribe smh+{conid}+{"period":"2d","bar":"1h","source":"t"}
- Parse incoming JSON into Tick objects
- Send tic keepalive every 30s
- Reconnect with exponential backoff on disconnect
- Async generator yielding Tick objects

### 3. src/signals/generator.py - SignalGenerator + generate_signals()
- Resolve symbols to conids via syncm.ibkr_layer.lookup
- Load historical candles from DB (get_local_candles) for indicator warmup
- Build BacktestState with preloaded candles + dummy portfolio
- Init strategy via init_strat(config.strategy_type)
- Main loop: receive tick from feed -> append candle -> on_tick -> print signals

### 4. src/signals/types.py
SignalEvent dataclass for formatted output.

## Files to Modify

### 5. main.py
Add signal command calling asyncio.run(generate_signals(config))

### 6. pyproject.toml
Add websockets>=14.0 dependency

## Console Output

Startup:
```
Signal generator started
  Strategy: breakout_ema
  Symbols:  SPY (conid: 265598)
  Bar:      1h
  Waiting for bars...
```

Signals:
```
[14:00:01] breakout_ema | SPY LONG  @ 512.34 | [breakout] enter long
[15:00:01] breakout_ema | SPY CLOSE @ 510.20 | [breakout] ema cross below
```

## State Management

- Bootstrap: load 500 historical candles from DB per symbol (enough for EMA-200 + margin)
- If insufficient local data, sync from IBKR via syncm.sync_data first
- Convert to (symbol, timestamp) MultiIndex for BacktestState.candles
- Dummy PortfolioState (empty positions) - stateless signal generation
- Strategies only emit entry signals (long/short) since positions always empty

## Error Handling

- WS disconnect: auto-reconnect, exponential backoff 1s->30s
- IBKR keepalive: tic every 30s
- Graceful shutdown: SIGINT/SIGTERM handler
- Startup validation: check strategy_type + symbol resolution
