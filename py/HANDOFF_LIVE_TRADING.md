# Handoff: Live Trading Engine

**Architect:** pi (architect mode)
**Implementer:** you
**Status:** Ready to implement — **4 open decisions pending (see §Open Decisions). Do not pick silently; wire them from the config or ask.**

This doc is the single source of truth for building `src/live/`. It reuses the existing backtest/strategy/data layers heavily — the live engine only adds the **tick→bar→strategy→execute loop** and a **broker-feed seam** on top of what already works.

---

## 0. TL;DR

- **New package `src/live/`** — 8 new modules + 2 edits (`main.py`, nothing in `src/bt`).
- **Reuses (do NOT reimplement):** `Candle`, `TradeSignal`, `BacktestState`, `PortfolioState`, `ActionType`, `FillEvent`, `CandleStore`, `init_strat`, `resolve_params`, `create_initial_backtest_state`, `src.bt.portfolio.pure.apply_fill`, `src.bt.strategies.ta_context.init_ta`, `src.data.db` query, `src.data.resample`.
- **First broker adapter = IBKR REST snapshot poller** (not the raw `/ws` socket — see A1). The `BrokerFeed` Protocol is the adapter seam so a socket adapter drops in later.
- **Simulated fills for v1** (A4): strategies emit `TradeSignal`s; the engine applies them to `PortfolioState` via `apply_fill`, logs `OrderIntent`s. **No real IBKR order placement.**
- **In-memory state, no persistence** (A5): fresh `BacktestState` per launch, warmed from local DB history so strategy lookback has context.
- **Clarify the shape of each copied interface in §2/§3** — the copied signatures below are verbatim from the current codebase and must remain compatible.

---

## 1. Assumptions & Open Decisions

### Assumptions already locked
- **A2 — time-based bar aggregation**, aligned to the strategy's base interval; `Candle` objects flow into the SAME strategy hook used by backtest.
- **A3 — reuse the strategy function only, not the backtest `run()` loop.** Live is tick-driven, not a feature-driven generator. `run_backtest` is NOT called.
- Asynchronous feed via `asyncio` (matching `src/data`), synchronous pure core.

### ❗ OPEN DECISIONS (confirm with user before finalizing; default value noted)
1. **A1 — adapter source.** Plan defaults to **REST snapshot poller** (`get_iserver_marketdata_snapshot`, default poll 1s). Building the raw IBKR `/ws` socket is a separate multi-day deliverable (proprietary framing; the client-lib `get_ws` is only an HTTP-upgrade stub). *Default: snapshot poller.*
2. **A4 — real orders vs simulated fills.** Plan defaults to **simulated** (log `OrderIntent`, apply via `apply_fill`). Real routing = extra `OrderBroker` that posts to IBKR. *Default: simulated.*
3. **A5 — persistence.** Plan defaults to **none** — fresh state each launch, warmed from DB. *Default: none.*
4. **Trading-hours gate.** Plan defaults to **aggregate whatever streams** (no market-calendar filtering). Optionally gate on `src.data.xcal.is_non_trading_day`. *Default: no gate.*

> If the user picks differently on any of these, only the affected modules change — the envelope (`BrokerFeed`, `LiveEngine`, `baragg`) is unchanged. The key invariant: **the strategy plane sees identical `Candle`s regardless of the feed source or fill mode.**

---

## 2. Types & Module Ownership

### New modules

| Module | Owns |
|---|---|
| `src/live/types.py` | All shared domain types below |
| `src/live/result.py` | `Ok`/`Err` Result building block (15 lines, no dep) |
| `src/live/baragg.py` | Pure time-based bar aggregation `Tick`→closed `Candle`s |
| `src/live/feed.py` | `BrokerFeed` Protocol + `TickSink`/`FeedError` |
| `src/live/feed_ibkr.py` | `IBKRSnapshotFeed` (concrete adapter) |
| `src/live/engine.py` | Pure live pipeline: tick→aggregate→strategy→execute→state |
| `src/live/runner.py` | Wiring: load history, warm state, boot feed, drive engine |
| `src/live/cli.py` | `ibkr live run <config.json>` (click command) |

### Modified
| Module | Change |
|---|---|
| `src/bt/strategy` — **none** | not modified; reuse as-is |
| `main.py` | register `live_group` |

### Priority / dependency order
1. `result.py` → 2. `types.py` → 3. `baragg.py` → 4. `feed.py` → 5. `feed_ibkr.py` → 6. `engine.py` → 7. `runner.py` → 8. `cli.py` / `main.py`.

---

### All types (verbatim contract)

```python
# src/live/result.py
from typing import Generic, TypeVar
T = TypeVar("T"); E = TypeVar("E")

@dataclass(frozen=True)
class Ok(Generic[T, E]):
    value: T
@dataclass(frozen=True)
class Err(Generic[T, E]):
    error: E
Result = Ok[T, E] | Err[T, E]
```

```python
# src/live/types.py
from typing import Literal, Protocol, Awaitable, Callable, Tuple, Dict, Mapping, TypeAlias
import pandas as pd
from src.bt.state import Candle, TradeSignal, BacktestState, PortfolioState, ActionType

@dataclass(frozen=True)
class Tick:
    symbol: str
    timestamp: pd.Timestamp      # localized/naive-UTC bar timestamp
    price: float                 # last trade price
    bid: float | None
    ask: float | None
    volume: float                # >= 0
    source_ts: float              # raw ms epoch from broker (ordering/dedup)

@dataclass(frozen=True)
class BarInterval:
    unit: Literal["sec", "min", "hour", "day"]
    size: int                     # > 0
    def seconds(self) -> int: ... # size * unit_base_sec

@TypeAlias
BarConfig = tuple[BarInterval, ...]   # bars[0] = base (signal); bars[1:] = htf

@dataclass(frozen=True)
class PartialBar:
    interval: BarInterval
    symbol: str
    open: float; high: float; low: float; close: float; volume: float
    boundary_start: pd.Timestamp   # inclusive
    boundary_end: pd.Timestamp     # exclusive

@dataclass(frozen=True)
class FeedError:
    kind: Literal["auth", "rate_limit", "sub", "transport", "symbol_unknown"]
    message: str
    symbol: str | None = None

class TickSink(Protocol):
    def __call__(self, tick: Tick) -> Awaitable[None]: ...

class LiveConfig:
    strategy_type: str
    symbols: Tuple[str, ...]
    initial_capital: float
    commission: float
    strategy_params: dict
    bars: BarConfig
    warmup_back_hours: int        # hours of DB history to warm the store
    base_interval_str: str        # e.g. "1h" or "60s" label
    poll_interval_s: float        # adapter cadence (default 1.0)
```

```python
# src/live/feed.py
class BrokerFeed(Protocol):
    async def subscribe(self, symbol: str, sink: TickSink) -> Result[None, FeedError]: ...
    async def unsubscribe(self, symbol: str) -> Result[None, FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...
```

```python
# src/live/feed_ibkr.py  (concrete)
@dataclass(frozen=True)
class SnapshotQuote:
    conid: int; symbol: str
    last: float | None; bid: float | None; ask: float | None
    volume: float; ts: float        # ms epoch
```

```python
# src/live/engine.py
@dataclass(frozen=True)
class OrderIntent:
    symbol: str; action: ActionType; qty: float
    ref_price: float; timestamp: pd.Timestamp; tag: str

@dataclass(frozen=True)
class ExecutedOrder:
    intent: OrderIntent; exec_price: float; commission: float; timestamp: pd.Timestamp

@dataclass(frozen=True)
class LiveState:
    bs: BacktestState
    partials: Mapping[tuple[str, str], PartialBar]     # (symbol, interval_str) -> partial
    open_ints: Tuple[OrderIntent, ...]
```

---

## 3. Functions (signatures + one-line contract)

### Aggregator — `src/live/baragg.py` (pure, deterministic, no I/O, no mutation)

```python
def open_boundary(timestamp: pd.Timestamp, iv: BarInterval) -> pd.Timestamp
    """Floor `timestamp` to the interval boundary (e.g. sec-60 -> :00, min-1 -> :00)."""
def to_interval_str(iv: BarInterval) -> str
    """Return '60s' | '5min' | '1h' | '1d' — the string used for Candle.interval + store keys."""
def start_partial(tick: Tick, iv: BarInterval) -> PartialBar
    """Open a new bar: boundary_start = open_boundary(tick.timestamp, iv), end = start + iv."""
def absorb(bar: PartialBar, tick: Tick) -> PartialBar
    """Fold one tick: high = max(high, price), low = min(low, price), close=price, volume+=tick.volume. New PartialBar."""
def is_closed(bar: PartialBar, now: pd.Timestamp) -> bool
    """True when now >= bar.boundary_end (exclusive close)."""
def finalize_close(bar: PartialBar, last_price: float) -> Candle
    """Freeze a closed bar into a backtest Candle (interval = to_interval_str(bar.interval))."""
def aggregate_ticks(
    partials: Mapping[tuple[str, str], PartialBar],
    ticks: Sequence[Tick],
    now: pd.Timestamp,
    intervals: Sequence[BarInterval],
) -> tuple[Mapping[tuple[str, str], PartialBar], tuple[Candle, ...]]
    """Pure: fold ticks into partials; for every closed base/htf bar return a closed Candle AND a fresh open partial for it. Deterministic order: per symbol base then htf."""
```

**Aggregation rule (make this exact):**
- One `PartialBar` keyed `(symbol, interval_str)` for each configured interval.
- On absorbing a `Tick`, only update the bar whose boundary contains `tick.timestamp`.
- A bar is `closed` (and a `Candle` emitted) when `now >= boundary_end`. Emit exactly once, then start a fresh partial from the last tick.
- Base bar (smallest interval) closes first per symbol, then HTF bars that closed on the same `now`.

### Adapter — `src/live/feed.py` + `src/live/feed_ibkr.py`

```python
# src/live/feed_ibkr.py
class IBKRSnapshotFeed:
    """REST snapshot-poller conforming to BrokerFeed. Async."""
    def __init__(self, client: Any, conid: Mapping[str, int], poll_interval_s: float) -> None: ...
    async def subscribe(self, symbol: str, sink: TickSink) -> Result[None, FeedError]: ...
    async def unsubscribe(self, symbol: str) -> Result[None, FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...
    # private
    async def _poll_loop(self) -> None: ...
def parse_snapshot(raw: Sequence[dict], conid: Mapping[str, int]) -> tuple[SnapshotQuote, ...]:
    """Pure: parse get_iserver_marketdata_snapshot rows; drop rows for unknown conids."""
def to_tick(q: SnapshotQuote) -> Tick:
    """Pure: price = last ?? midpoint(bid, ask); volume as reported; source_ts = q.ts."""
```

**Snapshot field mapping (from client-lib `MdFields`):** use `MdFields.VALUE_*` codes — field **31** = last price, **74** = open, **73** = close, **70** = bid, **71** = ask, **55/58** = volumes. Map the parsed row's `last` (field 31), `bid` (70), `ask` (71), `volume` (55), `ts`.

### Engine — `src/live/engine.py` (pure transforms; side effects only in `runner`)

```python
def warm_state(config: LiveConfig, history: pd.DataFrame) -> LiveState:
    """Seed a BacktestState's CandleStore from historical candles (warm lookback)."""
def on_tick(s: LiveState, tick: Tick, now: pd.Timestamp) -> tuple[LiveState, tuple[Candle, ...], tuple[OrderIntent, ...]]:
    """Fold one tick: aggregate; any closed base Candle -> run strategy -> execute intents.""
def _on_strategy(
    s: LiveState, candle: Candle, resolved_params: object, strategy_fn: Any,
) -> tuple[LiveState, tuple[TradeSignal, ...]]:
    """Run strategy hook with cursor at candle.timestamp; bucket signals by symbol."""
def execute_intents(s: LiveState, intents: tuple[OrderIntent, ...]) -> LiveState:
    """Apply each intent via src.bt.portfolio.pure.apply_fill (simulated fill at ref_price)."""
```

### Runner — `src/live/runner.py`

```python
def load_history(symbols: tuple[str, ...], back_hours: int, interval_str: str) -> pd.DataFrame:
    """Read warmup candles from local DB (src.data.db query) as a MultiIndex (symbol, timestamp) OHLCV frame."""
def discover_strategy(strategy_type: str) -> Any:
    """return init_strat(strategy_type)   (reuse src.bt.strategies.init_strat)"""
def is_dsl_strategy(strategy_fn: Any) -> bool:
    """return getattr(strategy_fn, 'ctx_fn', None) is not None  (marker set by @strategy)"""
def build_live_state(config: LiveConfig, history: pd.DataFrame, strat_mod: Any) -> LiveState:
    """Create BacktestState + CandleStore warmed from history; for DSL strategies attach a TaContext (init_ta) + mint strategy_state={}."""
async def run_live(config: LiveConfig) -> None:
    """Top-level async driver: build state, resolve_params, boot feed, subscribe, poll, shutdown."""
```

### CLI — `src/live/cli.py`

```python
@click.command("run")
@click.argument("config_path")
def live_run(config_path: str) -> None:
    """Load LiveConfig from JSON, call asyncio.run(run_live(config))."""
def load_live_config(path: str) -> LiveConfig:
    """Parse + validate JSON -> LiveConfig (validate bars/timestamps/symbols)."""
```

`main.py` edit:

```python
from src.live.cli import live_group
...
main.add_command(live_group)
```

---

## 4. Call Graph (must be implemented as specified)

### Production

```ts
main (click)
  → live_run(config_path: str): None                                    --> src/live/cli.py
    → load_live_config(path: str): LiveConfig                            --> src/live/cli.py
    → asyncio.run(run_live(config: LiveConfig): None)                     --> src/live/runner.py
      → load_history(symbols, back_hours, interval_str): pd.DataFrame     --> src/data/db
      → discover_strategy(strategy_type): Any (init_strat)               -> src/bt/strategies
      → resolve_params(strategy_type, params): object                    -> src/bt/strategies
      → build_live_state(config, history, strat_mod): LiveState          --> src/live/engine.py
        → create_initial_backtest_state(symbols, capital, start)          -> src/bt/state/factories
        → CandleStore(rows)                                                -> src/bt/engine/candle_store
        → init_ta(history, symbols, base) -> attach_ta (if DSL)          -> src/bt/strategies/ta_context
        → strategy_state = {}; store.attach_strategy_state (if stateful DSL)
      → IBKRSnapshotFeed(client, conid_map, poll_interval_s)              --> src/live/feed_ibkr.py
      → for symbol: feed.subscribe(symbol, sink) -> Result[None, FeedError] --> src/live/feed_ibkr.py
          → _poll_loop(): None   (while active)
            → get_iserver_marketdata_snapshot.asyncio_detailed(conids, fields)   -- IBKR REST
            → parse_snapshot(raw, conid_map): tuple[SnapshotQuote, ...]          (pure)
            → to_tick(q: SnapshotQuote): Tick                                   (pure)
            → sink(tick)  --> engine.on_tick callback
      → on_tick(s, tick, now): (LiveState, candles, intents)             --> src/live/engine.py
        → aggregate_ticks(partials, [tick], now, intervals): (partials, closed) --> src/live/baragg.py
          → absorb(bar, tick): PartialBar                                       (pure)
          → is_closed(bar, now): bool                                           (pure)
          → finalize_close(bar, last): Candle                                   (pure)
        → _on_strategy(s, candle, resolved_params, strategy_fn): (LiveState, signals) --> src/live/engine.py
          → strategy_fn(state: BacktestState, candle: Candle, params): list[TradeSignal]  (REUSED hook)
          → bucket_signals(signals): dict[str, tuple[TradeSignal, ...]]
        → execute_intents(s, intents): LiveState                                  --> src/live/engine.py
          → apply_fill(portfolio, fill): PortfolioState                         -> src/bt/portfolio/pure
      → feed.close(): Result[None, FeedError>
```

### Tests

```ts
live-engine tests
  → FakeFeed (implements BrokerFeed): enqueue(Tick)              --> tests/test_live_engine.py
  → on_tick(s, tick, now): (LiveState, candles, intents)
    → aggregate_ticks(...) -> absorb / is_closed / finalize_close  (assert OHLC)
baragg tests
  → absorb(PartialBar, Tick): PartialBar            (high/low/close/vol updates)
  → open_boundary(ts, iv): pd.Timestamp             (boundary alignment, inclusive/exclusive)
  → aggregate_ticks(...): (partials, closed)         (HTF cascade: N base bars -> one HTF bar)
feed tests
  → parse_snapshot(rows, conid): tuple[SnapshotQuote,...]   (field mapping + unknown-drop)
  → to_tick(SnapshotQuote): Tick                          (last vs midpoint fallback)
runner tests
  → build_live_state(config, history): LiveState    (store populated; cursor/future safe)
```

---

## 5. Verification (runnable assertions, no framework)

Drop these as `src/live/tests/test_live.py` using plain `assert`. Run with `uv run pytest src/live/tests/`.

1. **Boundary alignment**
   ```python
   def test_open_boundary_minute():
       ts = pd.Timestamp("2026-08-15 10:05:37")
       assert open_boundary(ts, BarInterval("min", 1)) == pd.Timestamp("2026-08-15 10:05:00")
   ```
2. **OHLC aggregation**
   ```python
   def test_absorb_ohlc():
       iv = BarInterval("sec", 60)
       p = start_partial(Tick("AAPL", ts0, price=100.0, bid=None, ask=None, volume=1.0, source_ts=0), iv)
       p = absorb(p, Tick("AAPL", ts1, price=102.5, bid=None, ask=None, volume=1.0, source_ts=1))
       p = absorb(p, Tick("AAPL", ts2, price=101.0, bid=None, ask=None, volume=1.0, source_ts=2))
       assert p.open == 100.0 and p.high == 102.5 and p.low == 100.0 and p.close == 101.0
       assert p.volume == 3.0
   ```
3. **Bar closes at boundary exclusive**
   ```python
   def test_is_closed_boundary():
       ts0 = pd.Timestamp("2026-08-15 10:05:00"); iv = BarInterval("sec", 60)
       p = start_partial(Tick("AAPL", ts0, price=100.0, ...), iv)
       assert is_closed(p, pd.Timestamp("2026-08-15 10:05:59")) is False
       assert is_closed(p, pd.Timestamp("2026-08-15 10:06:00")) is True
   ```
4. **HTF cascade**
   ```python
   def test_htf_aggregates_after_n_base():
       # feed 5 base bars' ticks; assert exactly ONE 5min Candle emitted, OHLC correct
   ```
5. **Snapshot parse + fallback**
   ```python
   def test_parse_and_tick():
       rows = [{"conid": 1, "31": 101.5, "70": 101.4, "71": 101.6, "55": 1000, ...}]
       (q,) = parse_snapshot(rows, {1: "AAPL"})
       assert to_tick(q).price == 101.5
       # missing last -> midpoint fallback
   ```
6. **Live/backtest parity (the critical check)**
   ```python
   def test_live_parity():
       # Feed the SAME candle sequence into backtest.candle_generator and into the
       # live aggregator; assert the strategy sees identical Candle OHLCV + interval
       # in both. This proves the bar aggregator is a drop-in feed.
   ```
7. **Warm state cursor-safety** — `build_live_state` store has no future candles: `store.count(sym, iv) == len(history)`.

---

## 6. State / Cursor / DSL integration notes (read before implementing)

- The strategy hook is invoked with a **cursor-advanced** `BacktestState` (`state.candles.advance(candle.timestamp)` before `strategy_fn(state, candle, params)`), exactly as `_generate_signals` does in `src/bt/engine/backtest.py`. Copy that cursor contract; do not fork the whole loop.
- **DSL strategies** need `state.candles.ta` to be a bound `TaContext`. Mirror `run()`: if `getattr(strategy_fn, "ctx_fn", None) is not None`, build `ta = init_ta(history, symbols, base)` and `store.attach_ta(ta); ta.bind(store)`. If the strategy is stateful (`getattr(strategy_fn, "stateful", False)`), mint `strategy_state = {}` and `store.attach_strategy_state(strategy_state)`.
- **CandleStore is warmed by copying prebuilt numpy rows** (like `_append_candle` builds) OR by building the store from the history frame directly. Either is fine as long as `store[(sym, iv)]` returns a DataFrame with the full history and cursor is at the latest warm bar. Prefer building `CandleRows` directly from the history DataFrame (no per-row `_append_candle` loop) for startup cost.
- **Interval string** on `Candle.interval` must match what strategies/config use (base `bars[0]`). Use `to_interval_str(BarInterval)` for both warmup and live candles so lookback keys align.
- **Position/fill math** reuses the pure functions untouched. The engine must never mutate — always `replace`/reconstruct `PortfolioState`/`BacktestState`.

---

## 7. AGENTS.md compliance checklist

- [ ] Full type annotations on every function/dataclass. No `Any` without a `# comment:` reason.
- [ ] `@dataclass(frozen=True)` for all state; `Protocol` for injection (BrokerFeed).
- [ ] No mutation — pure transforms; side effects (network, DB) only in `runner.py`/`feed_ibkr.py`.
- [ ] Functions ≤ 50 LOC, classes ≤ 150 LOC.
- [ ] `Result[Ok, Err]`/`Result` used at I/O boundaries instead of exceptions for expected feed errors.
- [ ] New logic has assertion-based tests (§5). `make check` must pass; runs < 10s.
- [ ] No new dependencies (asyncio + httpx + stdlib only).
- [ ] Top-level imports only; `TYPE_CHECKING` guard if needed for circulars.
- [ ] Reuse `src.data.db` for DB reads; never re-implement query logic.

---

## 8. YAGNI (explicitly NOT built)

Real IBKR order placement; raw `/ws` socket adapter; multi-broker adapters beyond IBKR poller; state persistence/recovery; order-books/L2 depth; real-time PnL dashboards; latency metrics; new dependencies. Each is a separate follow-on.

---

## 9. Open decisions to confirm with the user before finalizing

1. **A1** — snapshot poller OK as first adapter (default), vs raw-socket now?
2. **A4** — simulated fills (default) vs real IBKR order placement in v1?
3. **A5** — no persistence (default) vs restart recovery?
4. **Trading-hours gate** — aggregate whatever streams (default) vs filter on market calendar?

Confirm these four and the plan is fully implementable as written.
