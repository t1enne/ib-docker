# Functional Rewrite of Backtest Engine

## Executive Summary

This plan outlines a comprehensive rewrite of the backtest engine to adopt a **functional programming style**. The current engine relies heavily on mutable state and class methods that mutate objects in place. This makes testing difficult, debugging hard, and reasoning about state changes complex.

**Core Philosophy**: Transform classes into immutable data containers and functions into pure state transformers that take input state, compute output, and return new state.

## Why Functional Style?

### Current Problems

1. **Hidden State Mutations**: Methods like `portfolio.on_fill()` mutate internal state, making it hard to track what changed
2. **Difficult Testing**: Must set up complex object state before testing; can't easily test intermediate states
3. **Debugging Complexity**: Can't easily snapshot and compare states at different points
4. **Race Conditions**: Async code with shared mutable state is error-prone
5. **No State Replay**: Can't easily replay from a specific point in time

### Functional Benefits

1. **Predictability**: Same input → same output, always
2. **Testability**: Pure functions are trivial to unit test
3. **Debuggability**: Can snapshot state at any point, diff states easily
4. **Composability**: Functions compose naturally via state passing
5. **Concurrency Safety**: Immutable data eliminates race conditions
6. **Time Travel**: Can save/restore states, implement undo/redo

## Architecture Transformation

### Before: Class-Based with Mutation

```python
class Portfolio:
    def __init__(self, initial_capital):
        self.cash = initial_capital  # Mutable!
        self.positions = {}          # Mutable!
        
    def on_fill(self, fill: FillEvent) -> None:
        self.cash -= fill.commission  # Mutation!
        self.positions[fill.symbol] = ...  # Mutation!
```

### After: Data + Pure Functions

```python
@dataclass(frozen=True)
class Portfolio:
    cash: float
    positions: Dict[str, Position]
    trades: Tuple[Trade, ...]  # Immutable sequence

def apply_fill(portfolio: Portfolio, fill: FillEvent) -> Portfolio:
    """Pure function: returns new Portfolio, doesn't mutate input."""
    new_cash = portfolio.cash - fill.commission
    new_positions = update_positions(portfolio.positions, fill)
    new_trades = portfolio.trades + (new_trade_from_fill(fill),)
    
    return Portfolio(
        cash=new_cash,
        positions=new_positions,
        trades=new_trades
    )
```

## Component-by-Component Transformation

### 1. State Types (Immutable Data Classes)

```python
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, FrozenSet
from datetime import datetime
import pandas as pd

@dataclass(frozen=True)
class Tick:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float]
    take_profit: Optional[float]

@dataclass(frozen=True)
class PortfolioState:
    cash: float
    positions: Dict[str, Position]  # Symbol -> Position
    trades: Tuple[Trade, ...]
    equity_curve: Tuple[EquityPoint, ...]
    initial_capital: float

@dataclass(frozen=True)
class TradeSignal:
    action: ActionType
    symbol: str
    timestamp: pd.Timestamp
    price: float
    qty: float
    reason: Optional[str]

@dataclass(frozen=True)
class FillEvent:
    signal: TradeSignal
    filled_qty: float
    executed_price: float
    commission: float
    slippage: float
    timestamp: pd.Timestamp

@dataclass(frozen=True)
class BacktestState:
    """Complete state snapshot at any point in time."""
    portfolio: PortfolioState
    timestamp: Optional[pd.Timestamp]
    pending_signals: Tuple[TradeSignal, ...]
    model_state: ModelState  # Strategy model state
    risk_events: Tuple[RiskEvent, ...]
    
@dataclass(frozen=True)
class ModelState:
    """All model computations (z-score, momentum, regime, etc.)."""
    z_score: Optional[float]
    current_regime: Optional[int]
    price_buffers: Tuple[Dict[str, float], ...]
    market_data: MarketDataState
    
@dataclass(frozen=True)
class MarketDataState:
    """Immutable market data history."""
    timestamps: Tuple[pd.Timestamp, ...]
    bars: Tuple[Dict[str, Tick], ...]  # [{sym1: Tick, sym2: Tick}, ...]
```

### 2. Pure Function Signatures

```python
# Portfolio functions
PortfolioState, FillEvent -> PortfolioState
PortfolioState, Dict[str, float] -> PortfolioState  # Update prices
PortfolioState -> PortfolioResult

# Strategy functions  
ModelState, Tick, Optional[Position] -> Tuple[TradeSignal, ...], ModelState

# Risk functions
PortfolioState, Tick -> Tuple[RiskEvent, ...]

# Execution functions
TradeSignal, Tick, ExecutionParams -> FillEvent

# Model update functions
ModelState, Dict[str, Tick] -> ModelState

# Engine step function
BacktestState, Tick -> BacktestState
```

### 3. Engine Loop Transformation

**Current (Stateful)**:
```python
async def _run_backtest(self, feed):
    for tick in feed:
        self.model.update(tick)  # Mutation
        signals = self.strategy.on_tick(tick)  # Uses self.model
        for signal in signals:
            fill = self.execution.execute(signal, tick)
            self.portfolio.on_fill(fill)  # Mutation
```

**Functional (State Transform)**:
```python
async def run_backtest(
    config: StrategyConfig,
    initial_state: BacktestState,
    data_stream: AsyncIterator[Tick]
) -> Iterator[BacktestState]:
    """Yields state after each tick for debugging/analysis."""
    state = initial_state
    
    async for tick in data_stream:
        state = process_tick(state, tick, config)
        yield state  # Can snapshot/save at any point
    
    return state

def process_tick(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig
) -> BacktestState:
    """Pure function: transform state with one tick."""
    # 1. Update models
    new_model_state = update_models(state.model_state, tick)
    
    # 2. Generate signals
    signals, new_model_state = generate_signals(
        new_model_state, tick, config
    )
    
    # 3. Execute pending signals
    fills = execute_signals(state.pending_signals, tick, config.execution)
    new_portfolio = apply_fills(state.portfolio, fills)
    
    # 4. Check risk
    risk_events = check_risk(new_portfolio, tick, config.risk)
    risk_fills = execute_risk_events(risk_events, tick)
    new_portfolio = apply_fills(new_portfolio, risk_fills)
    
    # 5. Update market prices
    new_portfolio = update_prices(new_portfolio, tick)
    
    # 6. Get new signals from strategy
    new_signals, new_model_state = strategy.on_tick(
        new_model_state, tick, new_portfolio
    )
    
    return BacktestState(
        portfolio=new_portfolio,
        timestamp=tick.timestamp,
        pending_signals=new_signals,
        model_state=new_model_state,
        risk_events=risk_events
    )
```

## Implementation Plan

### Phase 1: State Data Classes (Days 1-2)

**Goal**: Convert all mutable state to immutable dataclasses

**Files to Create**:
- `src/bt/state/types.py` - All state dataclasses
- `src/bt/state/factories.py` - Factory functions for initial states

**Key Conversions**:

1. **Portfolio** → `PortfolioState`
   - Replace `Dict[str, Trade]` with `Dict[str, Position]` 
   - Replace `List[Trade]` with `Tuple[Trade, ...]`
   - Remove mutation methods

2. **StrategyModel** → `ModelState`
   - Extract all computed values into frozen dataclass
   - Separate model computation from state storage

3. **MarketDataView** → `MarketDataState`
   - Replace mutable lists with tuples
   - Make slicing return new state, not views

4. **RegimeModel/ZModel** → Model computation functions
   - Separate training/inference from state storage
   - State only stores results, not model objects

**Validation**:
- All dataclasses use `@dataclass(frozen=True)`
- No mutable default arguments
- Use tuples instead of lists for sequences
- Use Mapping/Dict for key-value (still immutable if frozen)

### Phase 2: Portfolio Functions (Days 3-4)

**Goal**: Convert Portfolio class methods to pure functions

**Files to Create**:
- `src/bt/portfolio/pure.py` - Pure portfolio functions
- `src/bt/portfolio/state.py` - PortfolioState dataclass

**Function Implementations**:

```python
# src/bt/portfolio/pure.py

def apply_fill(
    portfolio: PortfolioState,
    fill: FillEvent
) -> PortfolioState:
    """Apply a fill to portfolio, return new state."""
    if fill.signal.action == ActionType.close:
        return _close_position(portfolio, fill)
    else:
        return _open_position(portfolio, fill)

def _open_position(
    portfolio: PortfolioState,
    fill: FillEvent
) -> PortfolioState:
    """Open a new position from fill."""
    signal = fill.signal
    qty = calculate_position_qty(portfolio, signal, fill)
    
    position = Position(
        symbol=signal.symbol,
        qty=qty,
        entry_price=fill.executed_price,
        entry_time=fill.timestamp,
        stop_loss=calculate_stop_loss(signal, fill),
        take_profit=calculate_take_profit(signal, fill)
    )
    
    new_positions = dict(portfolio.positions)
    new_positions[signal.symbol] = position
    
    new_cash = portfolio.cash - (qty * fill.executed_price) - fill.commission
    
    trade = Trade(
        entry_time=signal.timestamp,
        entry_price=fill.executed_price,
        symbol=signal.symbol,
        position=signal.action,
        qty=qty,
        # ... other fields
    )
    
    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=portfolio.trades + (trade,),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital
    )

def _close_position(
    portfolio: PortfolioState,
    fill: FillEvent
) -> PortfolioState:
    """Close a position from fill."""
    symbol = fill.signal.symbol
    position = portfolio.positions.get(symbol)
    
    if not position:
        return portfolio  # No-op if no position
    
    # Calculate PnL
    is_long = position.qty > 0
    qty = abs(position.qty)
    
    if is_long:
        pnl = (fill.executed_price - position.entry_price) * qty
        cash_change = qty * fill.executed_price - fill.commission
    else:
        pnl = (position.entry_price - fill.executed_price) * qty
        cash_change = (qty * position.entry_price) + pnl - fill.commission
    
    # Create closed trade record
    closed_trade = ClosedTrade(
        entry_time=position.entry_time,
        entry_price=position.entry_price,
        exit_time=fill.timestamp,
        exit_price=fill.executed_price,
        symbol=symbol,
        pnl=pnl,
        # ... other fields
    )
    
    # Return new state
    new_positions = {k: v for k, v in portfolio.positions.items() if k != symbol}
    new_cash = portfolio.cash + cash_change
    
    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=portfolio.trades + (closed_trade,),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital
    )

def update_prices(
    portfolio: PortfolioState,
    tick: Tick
) -> PortfolioState:
    """Update position valuations with new prices."""
    if tick.symbol not in portfolio.positions:
        return portfolio
    
    # Update equity curve with new valuations
    # ... implementation
    
    return portfolio  # Or new state with updated equity

def calculate_equity(portfolio: PortfolioState, prices: Dict[str, float]) -> float:
    """Calculate total equity from cash + position values."""
    position_value = sum(
        pos.qty * prices.get(sym, pos.entry_price)
        for sym, pos in portfolio.positions.items()
    )
    return portfolio.cash + position_value
```

**Testing Strategy**:
```python
# Pure functions are trivial to test
def test_apply_fill():
    portfolio = create_test_portfolio(cash=10000)
    fill = create_test_fill(symbol="AAPL", qty=10, price=100)
    
    new_portfolio = apply_fill(portfolio, fill)
    
    # Assertions are straightforward
    assert new_portfolio.cash == 9000  # 10000 - (10 * 100)
    assert "AAPL" in new_portfolio.positions
    assert len(new_portfolio.trades) == 1
    
    # Original portfolio unchanged!
    assert portfolio.cash == 10000
    assert "AAPL" not in portfolio.positions
```

### Phase 3: Risk Management Functions (Days 5-6)

**Goal**: Convert RiskManager to pure functions

**Files to Create**:
- `src/bt/risk/pure.py` - Risk checking functions
- `src/bt/risk/types.py` - Risk-specific types

```python
# src/bt/risk/pure.py

def check_risk(
    portfolio: PortfolioState,
    tick: Tick,
    config: RiskConfig
) -> Tuple[RiskEvent, ...]:
    """Check if any positions hit risk limits."""
    events = []
    
    for symbol, position in portfolio.positions.items():
        if symbol == tick.symbol:
            event = check_position_risk(position, tick, config)
            if event:
                events.append(event)
    
    return tuple(events)

def check_position_risk(
    position: Position,
    tick: Tick,
    config: RiskConfig
) -> Optional[RiskEvent]:
    """Check single position for SL/TP."""
    is_long = position.qty > 0
    
    # Check stop loss
    if is_long and position.stop_loss and tick.close <= position.stop_loss:
        return StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close
        )
    
    if not is_long and position.stop_loss and tick.close >= position.stop_loss:
        return StopLossEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close
        )
    
    # Check take profit
    if is_long and position.take_profit and tick.close >= position.take_profit:
        return TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close
        )
    
    if not is_long and position.take_profit and tick.close <= position.take_profit:
        return TakeProfitEvent(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            trigger_price=tick.close
        )
    
    return None

def update_trailing_stop(
    position: Position,
    tick: Tick,
    config: RiskConfig
) -> Position:
    """Update trailing stop, return new position."""
    if not config.trailing_stop:
        return position
    
    is_long = position.qty > 0
    
    if is_long:
        new_stop = tick.high * (1 - config.stop_loss_pct)
        if new_stop > (position.stop_loss or 0):
            return Position(
                symbol=position.symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                stop_loss=new_stop,
                take_profit=position.take_profit
            )
    else:
        new_stop = tick.low * (1 + config.stop_loss_pct)
        if new_stop < (position.stop_loss or float('inf')):
            return Position(
                symbol=position.symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                stop_loss=new_stop,
                take_profit=position.take_profit
            )
    
    return position
```

### Phase 4: Execution Functions (Days 7-8)

**Goal**: Convert ExecutionHandler to pure functions

**Files to Create**:
- `src/bt/execution/pure.py` - Execution functions
- `src/bt/execution/types.py` - Execution types

```python
# src/bt/execution/pure.py

def execute_signal(
    signal: TradeSignal,
    tick: Tick,
    params: ExecutionParams
) -> FillEvent:
    """Convert signal to fill with slippage/spread."""
    base_spread = signal.price * (params.spread_bps / 10000)
    
    # Calculate base price with spread
    if signal.action == ActionType.long:
        base_price = signal.price + base_spread
    elif signal.action == ActionType.short:
        base_price = signal.price - base_spread
    else:
        base_price = signal.price
    
    # Calculate slippage
    adverse = calculate_adverse_selection(signal, tick)
    slippage_bps = params.slippage_bps * (1.5 if adverse else 1.0)
    slippage = signal.price * (slippage_bps / 10000)
    
    executed_price = base_price + slippage
    commission = calculate_commission(signal, executed_price, params)
    
    return FillEvent(
        signal=signal,
        filled_qty=signal.qty,
        executed_price=executed_price,
        commission=commission,
        slippage=slippage,
        timestamp=tick.timestamp
    )

def execute_risk_event(
    event: RiskEvent,
    tick: Tick,
    params: ExecutionParams
) -> FillEvent:
    """Execute a risk-triggered close."""
    signal = TradeSignal(
        action=ActionType.close,
        symbol=event.symbol,
        timestamp=event.timestamp,
        price=event.trigger_price,
        qty=0,  # Will be determined from position
        reason=event.reason
    )
    
    # Risk events often have worse slippage
    base_price = event.trigger_price - (event.trigger_price * params.spread_bps / 10000)
    slippage = event.trigger_price * (params.slippage_bps * 2 / 10000)
    executed_price = base_price - slippage
    
    return FillEvent(
        signal=signal,
        filled_qty=0,  # To be filled from position
        executed_price=executed_price,
        commission=params.fixed_commission,
        slippage=slippage,
        timestamp=tick.timestamp
    )

def calculate_adverse_selection(
    signal: TradeSignal,
    tick: Tick
) -> bool:
    """Determine if slippage should be adverse."""
    price_move = tick.close - tick.open
    percent_move = price_move / tick.open if tick.open != 0 else 0
    
    if signal.action == ActionType.long:
        return percent_move < -0.001
    elif signal.action == ActionType.short:
        return percent_move > 0.001
    return False
```

### Phase 5: Model Functions (Days 9-10)

**Goal**: Convert StrategyModel to state + pure functions

**Files to Create**:
- `src/bt/models/pure.py` - Model computation functions
- `src/bt/models/state.py` - ModelState dataclass

```python
# src/bt/models/pure.py

def update_models(
    state: ModelState,
    tick_group: Dict[str, Tick],
    config: ModelConfig
) -> ModelState:
    """Update all models with new tick data."""
    # Update market data
    new_market_data = append_bar(state.market_data, tick_group)
    
    # Update z-score if enabled
    new_z = update_z_score(state, tick_group, config)
    
    # Update HMM if enabled
    new_regime = update_regime(state, new_market_data, config)
    
    return ModelState(
        z_score=new_z,
        current_regime=new_regime,
        price_buffers=update_price_buffers(state.price_buffers, tick_group, config),
        market_data=new_market_data
    )

def update_z_score(
    state: ModelState,
    tick_group: Dict[str, Tick],
    config: ModelConfig
) -> Optional[float]:
    """Calculate new z-score from price buffers."""
    if not config.z_score_enabled:
        return None
    
    prices = {sym: tick.close for sym, tick in tick_group.items()}
    new_buffers = state.price_buffers + (prices,)
    
    # Trim to window size
    if len(new_buffers) > config.rolling_window_size:
        new_buffers = new_buffers[-config.rolling_window_size:]
    
    if len(new_buffers) < 2:
        return 0.0
    
    # Calculate z-score
    return calculate_z_score(new_buffers, config)

def update_regime(
    state: ModelState,
    market_data: MarketDataState,
    config: ModelConfig
) -> Optional[int]:
    """Update HMM regime detection."""
    if not config.hmm_enabled:
        return None
    
    # Check if we need to retrain
    if should_retrain_hmm(state, config):
        model = train_hmm(market_data, config)
        return predict_regime(model, market_data)
    
    if state.hmm_model:
        return predict_regime(state.hmm_model, market_data)
    
    return None
```

### Phase 6: Strategy Functions (Days 11-12)

**Goal**: Convert strategies to pure functions

**Files to Create**:
- `src/bt/strategies/pure.py` - Strategy interface
- `src/bt/algos/pairs_trading_pure.py` - Pure pairs strategy

```python
# src/bt/strategies/pure.py

from typing import Protocol

class PureStrategy(Protocol):
    """Protocol for pure functional strategies."""
    
    def on_tick(
        self,
        model_state: ModelState,
        tick: Tick,
        portfolio: PortfolioState
    ) -> Tuple[Tuple[TradeSignal, ...], ModelState]:
        """Generate signals from tick data.
        
        Returns:
            Tuple of (signals, updated_model_state)
        """
        ...

# src/bt/algos/pairs_trading_pure.py

def pairs_trading_strategy(
    model_state: ModelState,
    tick: Tick,
    portfolio: PortfolioState,
    config: PairsConfig
) -> Tuple[Tuple[TradeSignal, ...], ModelState]:
    """Pure pairs trading strategy."""
    z_score = model_state.z_score
    
    # Check for exit
    position = portfolio.positions.get(tick.symbol)
    if position and abs(z_score) < config.exit_z:
        signal = TradeSignal(
            action=ActionType.close,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            qty=abs(position.qty),
            reason="z_regression"
        )
        return (signal,), model_state
    
    # Check for entry
    if len(model_state.price_buffers) < 2:
        return (), model_state
    
    # Need both symbols' ticks to generate entry signals
    pending_ticks = model_state.pending_ticks or {}
    pending_ticks[tick.symbol] = tick.close
    
    if len(pending_ticks) != 2:
        new_state = ModelState(
            **model_state.__dict__,
            pending_ticks=pending_ticks
        )
        return (), new_state
    
    # Generate entry signals
    sym1, sym2 = config.symbols
    
    if z_score < -config.entry_z:
        signals = (
            TradeSignal(ActionType.long, sym1, tick.timestamp, pending_ticks[sym1], 0, None),
            TradeSignal(ActionType.short, sym2, tick.timestamp, pending_ticks[sym2], 0, None)
        )
    elif z_score > config.entry_z:
        signals = (
            TradeSignal(ActionType.short, sym1, tick.timestamp, pending_ticks[sym1], 0, None),
            TradeSignal(ActionType.long, sym2, tick.timestamp, pending_ticks[sym2], 0, None)
        )
    else:
        signals = ()
    
    return signals, model_state
```

### Phase 7: Engine Orchestration (Days 13-14)

**Goal**: Wire everything together with pure engine

**Files to Create**:
- `src/bt/engine/pure_engine.py` - Functional backtest engine
- `src/bt/engine/pipeline.py` - Data processing pipeline

```python
# src/bt/engine/pure_engine.py

from typing import Iterator, Optional
import pandas as pd

async def run_backtest(
    config: StrategyConfig,
    data_stream: AsyncIterator[Tick],
    strategy: PureStrategy,
    initial_state: Optional[BacktestState] = None
) -> Iterator[BacktestState]:
    """Run functional backtest, yielding state after each tick.
    
    Args:
        config: Strategy configuration
        data_stream: Async iterator of ticks
        strategy: Pure strategy function
        initial_state: Optional initial state (for resuming)
    
    Yields:
        BacktestState after each tick
    
    Returns:
        Final BacktestState
    """
    state = initial_state or create_initial_state(config)
    
    async for tick in data_stream:
        # Process one tick through the pipeline
        state = await process_tick_pipeline(state, tick, config, strategy)
        
        # Yield for debugging/snapshotting
        yield state
    
    # Finalize (close positions, etc.)
    final_state = finalize_backtest(state, config)
    yield final_state
    
    return final_state

def process_tick_pipeline(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig,
    strategy: PureStrategy
) -> BacktestState:
    """Process single tick through all pipeline stages.
    
    Each stage is a pure function transforming state.
    """
    # Stage 1: Update models
    state = pipe(
        state,
        lambda s: update_model_state(s, tick, config)
    )
    
    # Stage 2: Execute pending signals
    state = pipe(
        state,
        lambda s: execute_pending_signals(s, tick, config)
    )
    
    # Stage 3: Check risk
    state = pipe(
        state,
        lambda s: check_and_execute_risk(s, tick, config)
    )
    
    # Stage 4: Update prices
    state = pipe(
        state,
        lambda s: update_portfolio_prices(s, tick)
    )
    
    # Stage 5: Generate new signals
    state = pipe(
        state,
        lambda s: generate_strategy_signals(s, tick, strategy)
    )
    
    return state

def pipe(value, *functions):
    """Pipe value through functions left-to-right."""
    for fn in functions:
        value = fn(value)
    return value

# Specific stage functions

def update_model_state(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig
) -> BacktestState:
    """Stage 1: Update all models."""
    new_model = update_models(
        state.model_state,
        {tick.symbol: tick},  # Wrap in dict for consistency
        config
    )
    
    return BacktestState(
        portfolio=state.portfolio,
        timestamp=tick.timestamp,
        pending_signals=state.pending_signals,
        model_state=new_model,
        risk_events=()
    )

def execute_pending_signals(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig
) -> BacktestState:
    """Stage 2: Execute signals matching current tick symbol."""
    relevant_signals = tuple(
        s for s in state.pending_signals
        if s.symbol == tick.symbol
    )
    
    remaining_signals = tuple(
        s for s in state.pending_signals
        if s.symbol != tick.symbol
    )
    
    fills = tuple(
        execute_signal(signal, tick, config.execution)
        for signal in relevant_signals
    )
    
    new_portfolio = reduce(
        apply_fill,
        fills,
        state.portfolio
    )
    
    return BacktestState(
        portfolio=new_portfolio,
        timestamp=state.timestamp,
        pending_signals=remaining_signals,
        model_state=state.model_state,
        risk_events=state.risk_events
    )

def check_and_execute_risk(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig
) -> BacktestState:
    """Stage 3: Check risk and execute closes."""
    risk_events = check_risk(state.portfolio, tick, config.risk)
    
    if not risk_events:
        return state
    
    fills = tuple(
        execute_risk_event(event, tick, config.execution)
        for event in risk_events
    )
    
    new_portfolio = reduce(
        apply_fill,
        fills,
        state.portfolio
    )
    
    return BacktestState(
        portfolio=new_portfolio,
        timestamp=state.timestamp,
        pending_signals=state.pending_signals,
        model_state=state.model_state,
        risk_events=risk_events
    )

def update_portfolio_prices(
    state: BacktestState,
    tick: Tick
) -> BacktestState:
    """Stage 4: Update portfolio with new prices."""
    new_portfolio = update_prices(state.portfolio, tick)
    
    return BacktestState(
        portfolio=new_portfolio,
        timestamp=state.timestamp,
        pending_signals=state.pending_signals,
        model_state=state.model_state,
        risk_events=state.risk_events
    )

def generate_strategy_signals(
    state: BacktestState,
    tick: Tick,
    strategy: PureStrategy
) -> BacktestState:
    """Stage 5: Generate new signals from strategy."""
    position = state.portfolio.positions.get(tick.symbol)
    
    new_signals, new_model = strategy.on_tick(
        state.model_state,
        tick,
        state.portfolio
    )
    
    return BacktestState(
        portfolio=state.portfolio,
        timestamp=state.timestamp,
        pending_signals=state.pending_signals + new_signals,
        model_state=new_model,
        risk_events=state.risk_events
    )

def finalize_backtest(
    state: BacktestState,
    config: StrategyConfig
) -> BacktestState:
    """Close all open positions at end of backtest."""
    # Create close signals for all positions
    close_signals = tuple(
        TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            timestamp=state.timestamp or pd.Timestamp.now(),
            price=position.entry_price,  # Use last known price
            qty=abs(position.qty),
            reason="backtest_end"
        )
        for symbol, position in state.portfolio.positions.items()
    )
    
    # Execute all closes
    new_portfolio = state.portfolio
    for signal in close_signals:
        # Mock fill at entry price
        fill = FillEvent(
            signal=signal,
            filled_qty=signal.qty,
            executed_price=signal.price,
            commission=config.commission,
            slippage=0.0,
            timestamp=signal.timestamp
        )
        new_portfolio = apply_fill(new_portfolio, fill)
    
    return BacktestState(
        portfolio=new_portfolio,
        timestamp=state.timestamp,
        pending_signals=(),
        model_state=state.model_state,
        risk_events=state.risk_events
    )
```

### Phase 8: Testing & Validation (Days 15-16)

**Goal**: Comprehensive testing of pure functions

**Testing Approach**:

```python
# tests/test_portfolio_pure.py

class TestPortfolioPure:
    """Tests for pure portfolio functions."""
    
    def test_apply_fill_open_position(self):
        """Test opening a position."""
        portfolio = create_portfolio(cash=10000)
        fill = create_fill(
            symbol="AAPL",
            qty=10,
            price=100,
            commission=1
        )
        
        new_portfolio = apply_fill(portfolio, fill)
        
        # Assertions
        assert new_portfolio.cash == 8999  # 10000 - 1000 - 1
        assert "AAPL" in new_portfolio.positions
        assert new_portfolio.positions["AAPL"].qty == 10
        assert len(new_portfolio.trades) == 1
        
        # Original unchanged
        assert portfolio.cash == 10000
        assert "AAPL" not in portfolio.positions
    
    def test_apply_fill_close_position(self):
        """Test closing a position."""
        position = Position(
            symbol="AAPL",
            qty=10,
            entry_price=100,
            entry_time=pd.Timestamp("2024-01-01"),
            stop_loss=None,
            take_profit=None
        )
        portfolio = create_portfolio(
            cash=5000,
            positions={"AAPL": position}
        )
        
        fill = create_fill(
            symbol="AAPL",
            action=ActionType.close,
            qty=10,
            price=110,  # Profit!
            commission=1
        )
        
        new_portfolio = apply_fill(portfolio, fill)
        
        assert new_portfolio.cash == 6099  # 5000 + 1100 - 1
        assert "AAPL" not in new_portfolio.positions
        assert len(new_portfolio.trades) == 1
        assert new_portfolio.trades[0].pnl == 100  # (110-100) * 10
    
    def test_apply_fill_idempotent(self):
        """Applying same fill twice gives same result as once."""
        portfolio = create_portfolio(cash=10000)
        fill = create_fill(symbol="AAPL", qty=10, price=100)
        
        result1 = apply_fill(portfolio, fill)
        result2 = apply_fill(apply_fill(portfolio, fill), fill)
        
        # Second apply should be no-op (position already exists)
        assert result1 == result2

# tests/test_engine_pure.py

class TestEnginePure:
    """Tests for pure engine functions."""
    
    def test_process_tick_pipeline(self):
        """Test complete tick processing pipeline."""
        config = create_test_config()
        state = create_initial_state(config)
        tick = create_test_tick(symbol="AAPL", price=100)
        strategy = create_mock_strategy([])
        
        new_state = process_tick_pipeline(state, tick, config, strategy)
        
        # Verify state transformations
        assert new_state.timestamp == tick.timestamp
        assert new_state.model_state is not None
    
    def test_state_snapshots(self):
        """Test that we can snapshot and compare states."""
        config = create_test_config()
        state = create_initial_state(config)
        
        # Record initial state
        snapshot1 = state
        
        # Process some ticks
        ticks = [
            create_test_tick("AAPL", 100),
            create_test_tick("AAPL", 101),
        ]
        
        for tick in ticks:
            state = process_tick_pipeline(state, tick, config, mock_strategy)
        
        # Can diff states
        assert state.timestamp != snapshot1.timestamp
        assert state.portfolio != snapshot1.portfolio
    
    def test_determinism(self):
        """Same inputs produce same outputs."""
        config = create_test_config()
        state = create_initial_state(config)
        tick = create_test_tick("AAPL", 100)
        
        result1 = process_tick_pipeline(state, tick, config, mock_strategy)
        result2 = process_tick_pipeline(state, tick, config, mock_strategy)
        
        assert result1 == result2

# tests/test_state_debugging.py

class TestStateDebugging:
    """Tests demonstrating debuggability improvements."""
    
    def test_state_can_be_serialized(self):
        """States can be pickled/saved for later analysis."""
        import pickle
        
        state = create_test_state()
        serialized = pickle.dumps(state)
        restored = pickle.loads(serialized)
        
        assert state == restored
    
    def test_state_diffing(self):
        """Can diff two states to see what changed."""
        state1 = create_test_state(cash=10000)
        state2 = state1._replace(cash=9000)
        
        diff = state_diff(state1, state2)
        assert diff.changed_fields == {"cash"}
        assert diff.cash.before == 10000
        assert diff.cash.after == 9000
    
    def test_time_travel(self):
        """Can replay from any saved state."""
        states = []
        
        # Run backtest, saving states
        state = create_initial_state(config)
        for tick in test_data:
            state = process_tick(state, tick, config, strategy)
            states.append(state)  # Save snapshot
        
        # Replay from middle
        middle_state = states[50]
        for tick in test_data[51:]:
            middle_state = process_tick(middle_state, tick, config, strategy)
        
        # Should match final state
        assert middle_state == states[-1]
```

### Phase 9: Integration & Migration (Days 17-18)

**Goal**: Maintain backward compatibility while migrating

**Migration Strategy**:

```python
# src/bt/engine/__init__.py

# Old engine (kept for compatibility)
from .backtest_engine import BacktestEngine

# New pure engine
from .pure_engine import run_backtest, process_tick_pipeline

# Adapter to use new engine with old interface
class BacktestEngine:
    """Adapter maintaining old interface, using new pure engine internally."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self._state: Optional[BacktestState] = None
    
    async def run(self) -> BacktestResults:
        """Run backtest using new pure engine."""
        data_feed = DataFeed(self.config, self._build_window())
        strategy = self._create_strategy()
        
        states = []
        async for state in run_backtest(
            self.config,
            data_feed.get_data_stream(),
            strategy
        ):
            states.append(state)
        
        self._state = states[-1] if states else None
        return self._to_results(states[-1])
```

**Gradual Migration Path**:
1. Keep old classes as thin wrappers around pure functions
2. Mark old methods as deprecated
3. Update tests to use pure functions directly
4. Eventually remove old classes

### Phase 10: Performance Optimization (Days 19-20)

**Goal**: Ensure functional style doesn't hurt performance

**Optimizations**:

```python
# 1. Use immutable data structures with structural sharing
from typing import Mapping
from functools import lru_cache

@dataclass(frozen=True)
class PortfolioState:
    # Use persistent data structures
    positions: Mapping[str, Position]  # Could be pyrsistent.PMap
    trades: Tuple[Trade, ...]
    
# 2. Memoize expensive computations
@lru_cache(maxsize=1024)
def calculate_z_score(buffers: Tuple[Dict, ...]) -> float:
    """Cached z-score calculation."""
    ...

# 3. Lazy evaluation for indicators
def lazy_indicator(data: Tuple[float, ...], window: int) -> Iterator[float]:
    """Generate indicator values on-demand."""
    ...

# 4. Batch operations
@dataclass(frozen=True)
class BatchUpdate:
    """Group multiple updates for efficiency."""
    fills: Tuple[FillEvent, ...]
    price_updates: Tuple[Tuple[str, float], ...]

def apply_batch(portfolio: PortfolioState, batch: BatchUpdate) -> PortfolioState:
    """Apply multiple updates in one pass."""
    ...
```

## Key Benefits Demonstration

### Before: Debugging Nightmare

```python
# Can't see what changed
portfolio.on_fill(fill)  # What changed? Who knows!

# Have to set up complex state
portfolio = Portfolio(props)
portfolio.cash = 10000
portfolio.positions["AAPL"] = some_position
# ... 50 more setup lines

# Test is brittle
assert portfolio.cash == expected  # Might fail for many reasons
```

### After: Crystal Clear

```python
# Everything is explicit
new_portfolio = apply_fill(portfolio, fill)

# Can compare states
diff = portfolio_diff(portfolio, new_portfolio)
print(diff)  # cash: 10000 -> 9000, positions: {} -> {"AAPL": ...}

# Easy to test
result = apply_fill(
    create_portfolio(cash=10000),  # Start state
    create_fill(qty=10, price=100)  # Input
)
assert result.cash == 9000  # Output

# Can snapshot at any point
states = []
for tick in data:
    state = process_tick(state, tick, config)
    states.append(state)  # Save for later analysis
    
# Replay from any point
investigate_state = states[1000]  # What happened at tick 1000?
```

## Success Criteria

1. **Functionality**:
   - [ ] All existing tests pass with new pure engine
   - [ ] Results match old engine (deterministic)
   - [ ] Can snapshot/restore state at any point
   - [ ] Backward compatibility maintained

2. **Testability**:
   - [ ] Unit tests don't require complex setup
   - [ ] Can test individual functions in isolation
   - [ ] State diffs show exactly what changed
   - [ ] Property-based tests work easily

3. **Debuggability**:
   - [ ] Can serialize/deserialize states
   - [ ] Can replay from any saved state
   - [ ] Can diff any two states
   - [ ] States are human-readable

4. **Performance**:
   - [ ] No more than 10% slower than old engine
   - [ ] Memory usage reasonable (structural sharing)
   - [ ] Can handle same data sizes

5. **Code Quality**:
   - [ ] All functions are pure (no side effects)
   - [ ] All state is immutable
   - [ ] Functions are composable
   - [ ] Type hints throughout

## File Structure

```
src/bt/
├── state/
│   ├── __init__.py
│   ├── types.py              # All state dataclasses
│   └── factories.py          # State creation functions
├── portfolio/
│   ├── __init__.py           # Re-export for compatibility
│   ├── pure.py               # Pure portfolio functions
│   └── state.py              # PortfolioState
├── risk/
│   ├── __init__.py
│   ├── pure.py               # Pure risk functions
│   └── types.py
├── execution/
│   ├── __init__.py
│   ├── pure.py               # Pure execution functions
│   └── types.py
├── models/
│   ├── __init__.py
│   ├── pure.py               # Pure model functions
│   └── state.py              # ModelState
├── strategies/
│   ├── __init__.py
│   ├── pure.py               # Strategy protocol
│   └── base.py               # Base strategy functions
├── engine/
│   ├── __init__.py           # Re-export for compatibility
│   ├── pure_engine.py        # New functional engine
│   ├── pipeline.py           # Pipeline functions
│   └── adapter.py            # Old interface adapter
└── utils/
    ├── functional.py         # Functional utilities (pipe, reduce, etc.)
    └── debug.py              # State debugging utilities
```

---

**Estimated Effort**: 20 days
**Priority**: High - Foundation for all future work
**Dependencies**: None, but affects everything
**Risk**: Medium - Large refactoring, but well-contained
