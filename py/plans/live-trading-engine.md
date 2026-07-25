# Live Trading Engine — Plan

> Build production-grade live trading on top of the existing `SignalGenerator`.
> Integrate backtest primitives (risk, portfolio, strategies, models) and add the two
> missing pillars: **BarAggregator** (tick → bar) and **LiveOrderBuilder** (IBKR order execution).

---

## 1. Architecture Overview

```
IBKR WS (tick / smh) ──→ BarAggregator ──→ StrategyFn.on_tick() ──→ TradeSignal
                                   ↑                                  │
                             1s ticks                          LiveOrderBuilder
                                                                     │
                                                              IBKR REST / WS
                                                                     │
                                                              Fill Confirmation
                                                                     │
                                                              ┌──────┴──────┐
                                                              │ ibkr_pf     │
                                                              │ bt_pf       │
                                                              └──────┬──────┘
                                                                     │
                                                              LiveState (persisted)
```

**Two data paths coexist from day one:**

| Path                    | Source                                           | Consumer                         | Resolution               |
| ----------------------- | ------------------------------------------------ | -------------------------------- | ------------------------ |
| **Bar path** (today)    | IBKR `smh` WS                                    | Existing `StrategyFn` strategies | 1h / 1d bars             |
| **Tick path** (phase 2) | IBKR `smd` WS → `BarAggregator` → resampled bars | Same `StrategyFn` strategies     | 1s ticks → N-minute bars |

The `BarAggregator` is the abstraction that makes this transparent: strategies
always receive bars; whether those bars come from the wire or are aggregated
from ticks is invisible to them.

---

## 2. Module Layout

New files live under `src/live/` — a sibling to `src/bt/` and `src/screen/`.

```
src/live/
├── __init__.py              # Public API: LiveEngine, run_live
├── types.py                 # Live-specific types (LiveState, IbkrOrder, OrderStatus)
│
├── baragg/
│   ├── __init__.py          # create_bar_aggregator
│   ├── types.py             # AggregatedBar, BarAggregator Protocol
│   ├── registry.py          # Resample registry (time-based, tick-count-based)
│   └── time_based.py        # TimeBarAggregator — fixed-interval bars
│
├── order/
│   ├── __init__.py          # create_live_order_builder
│   ├── types.py             # LiveOrder, OrderSubmit, OrderFill, OrderStatus
│   ├── builder.py           # LiveOrderBuilder — maps TradeSignal → IBKR order JSON
│   ├── client.py            # IbkrOrderClient — REST + WS order lifecycle
│   └── fills.py             # FillConfirmer — reconcile order fills → FillEvent
│
├── pf/
│   ├── __init__.py          # create_portfolio_adapter
│   ├── ibkr_pf.py           # IbkrPortfolioAdapter — IBKR account → PortfolioState
│   └── bt_pf.py             # BtPortfolioAdapter — in-memory bt PortfolioState
│
├── state/
│   ├── __init__.py          # create_live_state_manager
│   ├── types.py             # LiveState (extends BacktestState with live fields)
│   └── persistence.py       # SqlitePersistence — save/restore LiveState
│
├── daemon.py                # LiveEngine — main loop orchestration
└── config.py                # LiveConfig dataclass, CLI parser
```

**No changes to existing `src/bt/` or `src/screen/` files** — the live engine
imports and wires the existing primitives; it doesn't modify them.

---

## 3. Phase 1: BarAggregator (tick → bar)

### 3.1 Problem

Today `LiveBarFeed` subscribes to IBKR `smh` (streaming market data history)
which sends pre-computed bars. This works for 1h/1d strategies but:

- No sub-minute resolution
- Tied to IBKR's bar boundaries
- Cannot (later) accept raw tick data from `smd` subscriptions

### 3.2 Solution

A `BarAggregator` protocol that accepts raw ticks and yields completed bars
on interval boundaries.

```python
# src/live/baragg/types.py

@dataclass(frozen=True)
class AggregatedBar:
    symbol: str
    timestamp: pd.Timestamp  # bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str             # e.g. "1h", "5min"
    tick_count: int = 0


class BarAggregator(Protocol):
    """Receives ticks, emits completed bars."""

    def update(self, tick: Tick) -> list[AggregatedBar]:
        """Process one tick. Returns 0+ completed bars since last call."""
        ...

    @property
    def current_bar(self) -> AggregatedBar | None:
        """The in-progress bar (may be None before first tick)."""
        ...

    def reset(self) -> None:
        """Clear all state (e.g., on market session reset)."""
        ...
```

### 3.3 TimeBarAggregator — First Implementation

```python
# src/live/baragg/time_based.py

class TimeBarAggregator:
    """Fixed-interval bar aggregation from 1-second ticks.

    Aggregates ticks into bars aligned to UTC clock boundaries.
    A bar is completed and emitted when a tick falls into the next interval.

    Configurable interval: "1min", "5min", "15min", "1h", "4h", "1d"
    """

    def __init__(self, interval: str, symbol: str):
        self.interval = interval
        self.symbol = symbol
        self._bar: _PartialBar | None = None
        self._freq = pd.tseries.frequencies.to_offset(interval)

    def update(self, tick: Tick) -> list[AggregatedBar]:
        if tick.symbol != self.symbol:
            return []

        tick_ts = tick.timestamp
        bucket = tick_ts.floor(self.interval)

        completed: list[AggregatedBar] = []

        if self._bar is None:
            self._bar = _PartialBar(
                bucket=bucket,
                open=tick.close,   # first tick determines open
                high=tick.close,
                low=tick.close,
                close=tick.close,
                volume=tick.volume,
                tick_count=1,
            )
        elif bucket > self._bar.bucket:
            # Emit completed bar
            completed.append(self._bar.to_aggregated(self.symbol, self.interval))
            # Start new bar
            self._bar = _PartialBar(
                bucket=bucket,
                open=tick.close,
                high=tick.close,
                low=tick.close,
                close=tick.close,
                volume=tick.volume,
                tick_count=1,
            )
        else:
            # Update in-progress bar
            self._bar.high = max(self._bar.high, tick.close)
            self._bar.low = min(self._bar.low, tick.close)
            self._bar.close = tick.close
            self._bar.volume += tick.volume
            self._bar.tick_count += 1

        return completed
```

### 3.4 Future: TickCountAggregator / VolumeAggregator

The `registry.py` will provide a factory that selects the aggregator by interval
and provides a `create_bar_aggregator(symbol, interval) -> BarAggregator` function.
The registry is extensible: add a new aggregator class for tick-count bars or
volume bars without changing the callers.

### 3.5 Integration with LiveBarFeed

The existing `LiveBarFeed` yields `Tick` objects from IBKR's `smh` (pre-computed
bars). The tick path (phase 2) will:

1. Subscribe to `smd` (streaming market data) instead of `smh`
2. Feed ticks into `BarAggregator`
3. Emit completed bars to `StrategyFn`

Both paths produce the same `AggregatedBar` type — strategies are agnostic.

---

## 4. Phase 2: LiveOrderBuilder

### 4.1 Problem

Today `TradeSignal` objects are converted to `SignalEvent` and printed to stdout.
No orders are placed. The backtest code has `execute_signal` in `bt/execution/pure.py`
but it creates model prices with slippage, not real IBKR orders.

### 4.2 Solution

A `LiveOrderBuilder` that maps `TradeSignal` → IBKR order payload and a client
that submits the order and tracks its lifecycle.

```python
# src/live/order/types.py

class IbkrOrderType(Enum):
    MKT = "MKT"
    LMT = "LMT"
    STP = "STP"
    STP_LMT = "STP_LMT"
    TRAIL = "TRAIL"
    REL = "REL"


@dataclass(frozen=True)
class LiveOrder:
    """An order submitted to IBKR."""
    order_id: str            # IBKR order ID
    conid: int
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: IbkrOrderType
    quantity: float
    limit_price: float | None
    stop_price: float | None
    tif: Literal["DAY", "GTC", "IOC", "FOK"] = "DAY"
    outside_rth: bool = False
    parent_id: str | None = None          # For bracket orders
    strategy_ref: str | None = None       # For reconciliation


@dataclass(frozen=True)
class OrderFill:
    order_id: str
    fill_price: float
    fill_qty: float
    fill_time: pd.Timestamp
    commission: float
    status: Literal["filled", "partially_filled"]


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    status: Literal["PendingSubmit", "PreSubmitted", "Submitted",
                     "Filled", "Cancelled", "Inactive"]
    filled_qty: float
    remaining_qty: float
    avg_fill_price: float
    last_update: pd.Timestamp
```

### 4.3 LiveOrderBuilder

```python
# src/live/order/builder.py

class LiveOrderBuilder:
    """Maps TradeSignal + PortfolioState → LiveOrder for IBKR submission."""

    def __init__(self, conids: dict[str, int], account_id: str):
        self.conids = conids
        self.account_id = account_id

    def build_entry_order(
        self,
        signal: TradeSignal,
        position_size_pct: float,
        capital: float,
    ) -> LiveOrder:
        """Build an entry order from a TradeSignal."""
        conid = self.conids[signal.symbol]
        is_long = signal.action == ActionType.long
        base_qty = capital * position_size_pct / signal.price
        qty = round(base_qty, 0)  # whole shares for IBKR

        if qty <= 0:
            raise ValueError(f"Computed zero quantity for {signal.symbol}")

        return LiveOrder(
            order_id="",                     # assigned by IBKR
            conid=conid,
            symbol=signal.symbol,
            side="BUY" if is_long else "SELL",
            order_type=IbkrOrderType.MKT,    # start simple
            quantity=max(1, int(qty)),
            limit_price=None,
            stop_price=None,
            tif="DAY",
            strategy_ref=str(signal.reason or ""),
        )

    def build_close_order(self, position: Position, conid: int) -> LiveOrder:
        """Build an order to close an existing position."""
        is_long = position.type == ActionType.long
        return LiveOrder(
            order_id="",
            conid=conid,
            symbol=position.symbol,
            side="SELL" if is_long else "BUY",
            order_type=IbkrOrderType.MKT,
            quantity=int(abs(position.qty)),
            limit_price=None,
            stop_price=None,
            tif="DAY",
        )

    def build_bracket_stop(
        self,
        entry_order_id: str,
        position: Position,
        conid: int,
    ) -> LiveOrder:
        """Build a STOP order attached to an entry (bracket)."""
        is_long = position.type == ActionType.long
        return LiveOrder(
            order_id="",
            conid=conid,
            symbol=position.symbol,
            side="SELL" if is_long else "BUY",
            order_type=IbkrOrderType.STP,
            quantity=int(abs(position.qty)),
            limit_price=None,
            stop_price=position.stop_loss,
            tif="GTC",
            parent_id=entry_order_id,
        )

    def build_bracket_tp(
        self,
        entry_order_id: str,
        position: Position,
        conid: int,
    ) -> LiveOrder:
        """Build a LIMIT order for take-profit (bracket)."""
        is_long = position.type == ActionType.long
        return LiveOrder(
            order_id="",
            conid=conid,
            symbol=position.symbol,
            side="SELL" if is_long else "BUY",
            order_type=IbkrOrderType.LMT,
            quantity=int(abs(position.qty)),
            limit_price=position.take_profit,
            stop_price=None,
            tif="GTC",
            parent_id=entry_order_id,
        )
```

### 4.4 Order Client (IBKR REST / WS)

```python
# src/live/order/client.py

class IbkrOrderClient:
    """Submit and track orders via IBKR REST API + WS callbacks.

    REST endpoints used:
        POST /iserver/account/{accountId}/orders   → place order
        GET  /iserver/account/orders                → list live orders
        GET  /iserver/account/order/status/{id}     → order detail
        DELETE /iserver/account/{accountId}/order/{id} → cancel
    """

    def __init__(self, account_id: str):
        self.account_id = account_id

    async def place_order(self, order: LiveOrder) -> str:
        """Submit order. Returns IBKR order ID."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        ...

    async def get_order_status(self, order_id: str) -> OrderStatus:
        ...

    async def get_live_orders(self) -> list[OrderStatus]:
        """Poll all open orders for reconciliation."""
        ...

    async def place_bracket(
        self,
        entry: LiveOrder,
        stop: LiveOrder,
        tp: LiveOrder,
    ) -> tuple[str, str, str]:
        """Submit entry + attached stop/limit bracket.

        IBKR's /iserver/account/{id}/orders accepts an 'orders' array
        with parent/child relationships. This submits all three in one call.
        """
        ...
```

### 4.5 Fill Confirmer

```python
# src/live/order/fills.py

class FillConfirmer:
    """Reconcile order statuses → FillEvents for PortfolioState.

    Polls order status on a timer (or receives WS push). Converts fills
    to the bt FillEvent type so the existing bt/portfolio/pure.py applies them.
    """

    def __init__(self, client: IbkrOrderClient):
        self.client = client
        self._known_orders: dict[str, OrderStatus] = {}

    async def poll(self) -> list[FillEvent]:
        """Fetch all live orders, detect new/changed fills, return FillEvents."""
        live_orders = await self.client.get_live_orders()
        new_fills: list[FillEvent] = []

        for status in live_orders:
            prev = self._known_orders.get(status.order_id)
            if prev is not None and status.status == "Filled" and prev.status != "Filled":
                # New fill
                new_fills.append(self._to_fill_event(status))
            self._known_orders[status.order_id] = status

        return new_fills

    def _to_fill_event(self, status: OrderStatus) -> FillEvent:
        """Map OrderStatus → bt FillEvent for portfolio update."""
        ...
```

---

## 5. Phase 3: Portfolio Reconciliation (ibkr_pf vs bt_pf)

### 5.1 Problem

The backtest `PortfolioState` tracks positions, cash, PnL in memory. Live
trading needs to reconcile this model with IBKR's actual account state.
On startup, the engine must discover existing positions from IBKR rather
than starting with an empty portfolio.

### 5.2 Solution

A `PortfolioAdapter` protocol with two implementations:

```python
# src/live/pf/types.py

class PortfolioAdapter(Protocol):
    """Synchronizes a bt PortfolioState with an external source of truth."""

    async def reconcile(self, bt_portfolio: PortfolioState) -> PortfolioState:
        """Reconcile bt state → adjusted bt state based on external truth.

        E.g., if IBKR has a position that bt doesn't know about (stop-loss
        that was filled while the engine was down), inject it here.
        """
        ...

    async def snapshot(self) -> PortfolioState:
        """Load a fresh PortfolioState from the external source (cold start)."""
        ...


# src/live/pf/ibkr_pf.py

class IbkrPortfolioAdapter:
    """Reads IBKR account data via REST API to build PortfolioState.

    Uses:
        GET /portfolio/{accountId}/positions/{page} → current positions
        GET /portfolio/{accountId}/summary          → cash, equity
        GET /iserver/account/trades                 → recent trade history
    """

    def __init__(self, account_id: str, conid_to_ticker: dict[int, str]):
        self.account_id = account_id
        self.conid_to_ticker = conid_to_ticker

    async def snapshot(self) -> PortfolioState:
        """Build a PortfolioState from IBKR's current positions."""
        ...

    async def reconcile(self, bt_portfolio: PortfolioState) -> PortfolioState:
        """Merge IBKR truth with bt model state."""
        ...


# src/live/pf/bt_pf.py

class BtPortfolioAdapter:
    """No-op adapter: the bt PortfolioState IS the source of truth.

    Used in paper-trading / simulation mode where the engine creates its
    own positions and doesn't talk to a broker.
    """

    async def snapshot(self) -> PortfolioState:
        return _EMPTY_PORTFOLIO

    async def reconcile(self, bt_portfolio: PortfolioState) -> PortfolioState:
        return bt_portfolio  # bt state is authoritative
```

The adapter is selected at startup based on `LiveConfig.pf_mode`:

- `pf_mode = "bt"` → `BtPortfolioAdapter` — pure simulation
- `pf_mode = "ibkr"` → `IbkrPortfolioAdapter` — live sync with IBKR

---

## 6. Phase 4: State Persistence

### 6.1 Problem

If the engine restarts (crash, deploy, market close), `PortfolioState` and open
position data are lost. The engine must reconstruct state from IBKR positions
(ibkr_pf) or from a local database (bt_pf simulation mode).

### 6.2 Solution

```python
# src/live/state/persistence.py

class SqlitePersistence:
    """Save and restore LiveState to local SQLite (same db as src/db).

    Tables:
        live_trades   — mirrors bt's Trade records with IBKR order IDs
        live_state    — periodic snapshots of PortfolioState + metadata
        live_orders   — order submission log for audit trail
    """

    async def save_snapshot(self, state: LiveState) -> None:
        """Write periodic state snapshot (every N bars / M minutes)."""
        ...

    async def restore(self) -> LiveState | None:
        """Load most recent snapshot. Returns None on first run."""
        ...

    async def log_trade(self, trade: Trade, order_id: str) -> None:
        ...

    async def log_order(self, order: LiveOrder, status: OrderStatus) -> None:
        ...
```

---

## 7. Phase 5: LiveEngine — Main Loop

```python
# src/live/daemon.py

class LiveEngine:
    """Orchestrates the live trading pipeline.

    Flow per tick/bar cycle:
        1. Receive tick from BarAggregator → completed bars
        2. For each completed bar:
            a. Update models (ModelUpdaterFn)
            b. Update candle data in BacktestState
            c. Run StrategyFn.on_tick() → TradeSignal[]
            d. Apply risk checks (RiskCheckFn)
            e. Build orders from signals (LiveOrderBuilder)
            f. Submit orders to IBKR (IbkrOrderClient)
            g. Poll fills (FillConfirmer → FillEvents)
            h. Apply fills to PortfolioState (bt/portfolio/pure)
            i. Reconcile with IBKR state (IbkrPortfolioAdapter)
            j. Persist state snapshot
        3. Sleep until next tick
    """

    def __init__(self, config: LiveConfig):
        self.config = config
        self.aggregator: BarAggregator
        self.strategy: StrategyFn
        self.order_builder: LiveOrderBuilder
        self.order_client: IbkrOrderClient
        self.fill_confirmer: FillConfirmer
        self.pf_adapter: PortfolioAdapter
        self.persistence: SqlitePersistence
        self.state: BacktestState

    async def start(self) -> None:
        """Main daemon loop."""
        # 1. Bootstrap: resolve symbols, load historical data, restore state
        state = await self._bootstrap()

        # 2. Connect to IBKR ws
        async for completed_bars in self._tick_loop():
            for bar in completed_bars:
                state = await self._process_bar(state, bar)

    async def _process_bar(
        self,
        state: BacktestState,
        bar: AggregatedBar,
    ) -> BacktestState:
        """Process one completed bar through the full pipeline."""
        tick = bar.to_tick()  # AggregatedBar → Tick (compatible with strategies)

        # Update models
        if self.model_updater:
            state = self.model_updater(state, tick)

        # Append candle data
        state = self._append_candle(state, tick)

        # Execute any pending signals
        portfolio, pending = self._execute_bt_signals(state, tick)
        state = merge_bt_state(state, dict(portfolio=portfolio, pending_signals=pending))

        # Run strategy
        signals = self.strategy(state, tick, self.config.strategy_params)
        if signals:
            for s in signals:
                order = self.order_builder.build_entry_order(
                    s, self.config.position_size_pct, state.portfolio.cash
                )
                order_id = await self.order_client.place_order(order)
                self._pending_orders[order_id] = (order, s)

        # Poll fills
        fills = await self.fill_confirmer.poll()
        for fill in fills:
            portfolio = self._apply_fill(state.portfolio, fill)
            state = merge_bt_state(state, dict(portfolio=portfolio))

        # Risk checks (SL/TP)
        risk_events = self.risk_checker(state.portfolio, tick, self.risk_config)
        for event in risk_events:
            order = self.order_builder.build_close_order(...)
            await self.order_client.place_order(order)

        # Reconcile with IBKR
        state = merge_bt_state(state, dict(
            portfolio=await self.pf_adapter.reconcile(state.portfolio)
        ))

        # Persist
        await self.persistence.save_snapshot(state)

        return state
```

---

## 8. Integration with Existing `src/bt/` Primitives

| Primitive                                                        | Where it lives             | How live engine uses it                                                                                           |
| ---------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- | --- |
| `StrategyFn` (Protocol)                                          | `src/bt/types.py`          | Injected via config, called on each completed bar                                                                 |
| `ModelUpdaterFn`                                                 | `src/bt/types.py`          | Update z-score / regime / market data per bar                                                                     |
| `RiskCheckFn`                                                    | `src/bt/types.py`          | Applied after each bar (SL/TP triggers)                                                                           |
| `ExecuteSignal` / `ExecuteRiskEvent`                             | `src/bt/execution/pure.py` | **Replaced** by `LiveOrderBuilder` + `IbkrOrderClient` for live. Kept as `BtExecutionAdapter` for simulation mode |
| `apply_fill` (portfolio mutation)                                | `src/bt/portfolio/pure.py` | **Reused** directly — same `PortfolioState` type, same fill logic                                                 |
| `check_risk`                                                     | `src/bt/risk/pure.py`      | **Reused** directly for SL/TP detection                                                                           |
| `Tick`, `TradeSignal`, `FillEvent`, `PortfolioState`, `Position` | `src/bt/state/types.py`    | **Reused** directly — types are identical                                                                         |
| `BacktestState`                                                  | `src/bt/state/types.py`    | **Extended** via `LiveState` (adds order tracking fields)                                                         |
| `StrategyConfig`                                                 | `src/bt/types.py`          | **Reused** as-is (YAML → config)                                                                                  |
| `BarAggregator`                                                  | `src/live/baragg/` (new)   | New — converts 1s ticks to bars                                                                                   |
| `LiveOrderBuilder`                                               | `src/live/order/` (new)    | New — maps `TradeSignal` → IBKR orders                                                                            |
| `IbkrPortfolioAdapter`                                           | `src/live/pf/` (new)       | New — reconciles bt state with IBKR                                                                               | --> |

---

## 9. Config File

Extended YAML config (adds `live` block to existing strategy configs):

```yaml
name: breakout_ema_live
strategy_type: breakout_ema
symbols: [AAPL, MSFT, NVDA]
initial_capital: 100000
position_size: 0.25
stop_loss: 0.05
take_profit: 0.10
commission: 0.5
training_start: "2025-01-01"
training_end: "2025-12-31"
trading_start: "2026-01-01"
trading_end: "2026-12-31"
bar: "5min"
strategy_params:
  fast: 9
  slow: 14

# New: live trading config
live:
  enabled: true
  pf_mode: "ibkr" # "bt" | "ibkr"
  tick_source: "smd" # "smh" | "smd" — tick path or bar path
  bar_interval: "5min" # used when tick_source == "smd"
  order_defaults:
    order_type: "MKT" # "MKT" | "LMT"
    tif: "DAY"
    use_brackets: true # auto-attach SL/TP as bracket orders
  reconcile_interval_s: 60 # how often to poll IBKR for position reconciliation
  state_persistence: true
  account_id: "U1234567"
```

---

## 10. Implementation Sequence

| Step   | Component                                                                   | Estimated LOC | Depends On                  |
| ------ | --------------------------------------------------------------------------- | ------------- | --------------------------- |
| **1**  | `src/live/baragg/types.py` + `TimeBarAggregator` + `registry.py`            | 150           | Nothing                     |
| **2**  | Bar aggregator tests (tick sequences, interval boundaries, edge cases)      | 100           | Step 1                      |
| **3**  | `src/live/order/types.py` — dataclasses (LiveOrder, OrderFill, OrderStatus) | 80            | Nothing                     |
| **4**  | `src/live/order/builder.py` — LiveOrderBuilder (entry, close, bracket)      | 120           | Step 3                      |
| **5**  | `src/live/order/client.py` — IbkrOrderClient (REST place/cancel/status)     | 200           | Step 3                      |
| **6**  | `src/live/order/fills.py` — FillConfirmer                                   | 100           | Steps 3–5                   |
| **7**  | `src/live/pf/ibkr_pf.py` + `bt_pf.py` — PortfolioAdapter implementations    | 150           | Step 3, `bt/state/types.py` |
| **8**  | `src/live/state/persistence.py` — SqlitePersistence                         | 120           | `src/db/models.py`          |
| **9**  | `src/live/state/types.py` — LiveState                                       | 40            | `bt/state/types.py`         |
| **10** | `src/live/config.py` — LiveConfig dataclass                                 | 50            | Nothing                     |
| **11** | `src/live/daemon.py` — LiveEngine main loop                                 | 250           | Steps 1–10                  |
| **12** | `src/live/__init__.py` — run_live() entry point                             | 40            | Step 11                     |
| **13** | CLI integration in `main.py` (new `live` command)                           | 30            | Step 12                     |
| **14** | Integration + smoke tests (mock IBKR, single tick sequence)                 | 150           | Steps 1–13                  |

**Total: ~1,640 LOC of new code.** Zero changes to existing `src/bt/` or `src/screen/`.

---

## 11. Tick Architecture Migration Path (Phase 2+)

The `BarAggregator` abstraction makes tick-migration non-breaking:

| Phase   | Data source                      | Aggregator                                  | Strategy receives                       | User-visible change              |
| ------- | -------------------------------- | ------------------------------------------- | --------------------------------------- | -------------------------------- |
| Today   | IBKR `smh` (pre-computed bars)   | None                                        | Pre-computed `Tick` with `interval` set | Nothing                          |
| Phase 1 | IBKR `smh`                       | Still none                                  | Same as today                           | Internal only                    |
| Phase 2 | IBKR `smd` (1s real-time ticks)  | `TimeBarAggregator`                         | `Tick` with `interval="5min"`           | Set `tick_source: smd` in config |
| Phase 3 | IBKR `smd` + custom tick filters | `TimeBarAggregator` + `TickCountAggregator` | Same                                    | Extended config options          |

Strategies never know the difference: their `on_tick()` always receives a `Tick`
with a meaningful `interval` field.

---

## 12. Risk & Edge Cases

| Concern                      | Mitigation                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Duplicate orders**         | `LiveEngine` tracks `_pending_orders` dict; same signal from same bar is idempotent                               |
| **Engine crash mid-trade**   | `IbkrPortfolioAdapter.snapshot()` on restart discovers open positions from IBKR                                   |
| **IBKR WS disconnect**       | `LiveBarFeed` already has reconnection + exponential backoff                                                      |
| **Order rejected**           | `IbkrOrderClient.place_order()` catches non-200; logs + notifies; does not crash loop                             |
| **Fill confirmation race**   | `FillConfirmer.poll()` on each bar + periodic background reconciliation                                           |
| **Partial fills**            | `LiveOrder` tracks `remaining_qty`; fill events carry partial qty; `PortfolioState` handles partial closes        |
| **Market close**             | `LiveEngine` checks market hours; pauses processing; resumes on next open                                         |
| **Account ID missing**       | `LiveConfig` validates `account_id` at construction time                                                          |
| **Bracket order references** | IBKR requires `parent_id` in the child; the client ensures the entry order's returned ID is threaded to its SL/TP |
