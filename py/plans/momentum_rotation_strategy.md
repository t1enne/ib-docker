# Momentum Rotation Strategy Implementation Plan

## Executive Summary

This plan outlines the implementation of a **momentum rotation strategy** that ranks assets by momentum and rotates into top performers. Unlike the current pairs trading strategy (which trades mean reversion on spreads), momentum rotation is a trend-following strategy that requires different portfolio mechanics, risk management, and execution logic.

## Current System Understanding

### Architecture Overview

The system is built around these core components:

1. **BacktestEngine** (`src/bt/engine/backtest_engine.py:40`)
   - Orchestrates the backtest loop
   - Handles data streaming, model updates, and trade execution
   - Currently hardcoded for pairs trading with 2 symbols

2. **StrategyModel** (`src/bt/models/strategy_model.py:33`)
   - Composite facade providing features to strategies
   - Currently provides: z-score, HMM regime, hedge_beta, market_data
   - Designed for pairs trading features

3. **Portfolio** (`src/bt/portfolio/__init__.py:30`)
   - Tracks positions as individual trades
   - Supports long/short positions with collateral-based short accounting
   - Calculates equity curve and PnL

4. **RiskManager** (`src/bt/risk/__init__.py:21`)
   - Monitors stop-loss and take-profit levels
   - Currently per-position fixed percentage SL/TP

5. **ExecutionHandler** (`src/bt/execution/__init__.py:15`)
   - Converts TradeSignals to FillEvents with spread/slippage
   - Supports long, short, and close actions

6. **DataFeed** (`src/bt/data_feed/__init__.py:12`)
   - Async stream of OHLCV ticks from SQLite
   - Currently supports multiple symbols but processes them independently

### Current Strategy Type: Pairs Trading

- **Mechanism**: Long one asset, short another based on z-score divergence
- **Position Structure**: Always market-neutral (2 positions)
- **Entry**: |z-score| > entry_z
- **Exit**: |z-score| < exit_z
- **Risk**: Per-position SL/TP
- **Symbols**: Exactly 2 required

## Momentum Rotation Strategy Requirements

### Strategy Mechanics

Momentum rotation involves:
1. **Ranking**: Calculate momentum for all assets in universe
2. **Selection**: Select top N assets by momentum score
3. **Allocation**: Equal weight or risk-parity allocation
4. **Rebalancing**: Periodic rebalancing (daily, weekly, monthly)
5. **Rotation**: Replace underperformers with new leaders

### Key Differences from Pairs Trading

| Aspect | Pairs Trading | Momentum Rotation |
|--------|---------------|-------------------|
| Direction | Mean reversion | Trend following |
| Position count | Always 2 (market neutral) | Variable (N assets) |
| Entry/exit | Z-score based | Momentum rank based |
| Holding period | Until convergence | Fixed rebalancing period |
| Risk management | Per-position SL/TP | Portfolio-level stops or position sizing |
| Symbol count | Exactly 2 | Variable (>=2) |

### Required Components

#### 1. Momentum Scoring Model

```python
class MomentumModel:
    """Calculate momentum scores for assets."""
    
    # Indicators to support:
    # - Simple returns (N-period)
    # - Risk-adjusted returns (Sharpe-like)
    # - Exponential smoothing
    # - Multi-timeframe momentum
    # - Cross-sectional vs time-series
```

#### 2. Rank-Based Strategy

```python
class MomentumRotationStrategy:
    """Generate signals based on momentum ranking."""
    
    # Logic:
    # - Calculate momentum for all symbols
    # - Rank by momentum
    # - Generate signals to:
    #   - Enter top N assets (long)
    #   - Exit positions no longer in top N
    #   - Rebalance to target weights
```

#### 3. Portfolio Enhancements

Current portfolio tracks positions as individual trades. For rotation:
- Track allocations vs targets
- Support partial closes (for rebalancing)
- Handle multiple simultaneous positions

#### 4. Rebalancing Logic

```python
class RebalancingSchedule:
    """Determine when to rebalance."""
    
    # Types:
    # - Calendar-based (daily, weekly, monthly)
    # - Threshold-based (allocation drift > X%)
    # - Volatility-adjusted
```

## Architecture Modifications

### Phase 1: Generalize StrategyModel

**Current**: StrategyModel is tightly coupled to z-score/HMM
**Required**: Make StrategyModel pluggable for different model types

```python
# New structure
class StrategyModel:
    def __init__(self, symbols, config):
        self.symbols = symbols
        self.market_data = MarketDataView(symbols)
        
        # Feature models (optional based on strategy)
        self._z_model = None  # Only if z-score strategy
        self._momentum_model = None  # Only if momentum strategy
        self._regime_model = None  # Optional for all strategies
        
    @property
    def features(self) -> Dict[str, Any]:
        """Access computed features."""
        # Return dict of available features based on configured models
```

### Phase 2: Strategy-Specific Configuration

Extend `StrategyConfig` to support momentum rotation parameters:

```yaml
# momentum_rotation.yaml
name: sector_momentum
strategy_type: momentum_rotation
symbols:
  - spy
  - qqq
  - iwm
  - vti
  - voo

# Date ranges
training_start: 2022-01-01
training_end: 2022-06-01
trading_start: 2022-06-02
trading_end: 2026-01-01

# Momentum parameters
momentum_lookback: 63  # 3 months (daily bars)
momentum_method: simple_returns  # simple_returns, risk_adjusted, ema_diff
momentum_smoothing: 5  # EMA smoothing period

# Rotation parameters
top_n: 3  # Hold top 3 assets
rebalance_frequency: monthly  # daily, weekly, monthly
rebalance_threshold: 0.05  # Minimum drift to trigger rebalance (optional)

# Risk parameters (different from pairs trading)
max_position_weight: 0.5  # Max 50% in single asset
stop_loss: 0.15  # Portfolio-level stop (optional)
take_profit: 0.25  # Portfolio-level take profit (optional)
volatility_target: 0.15  # Target annual vol for position sizing

# Position sizing
position_sizing: equal_weight  # equal_weight, risk_parity, inverse_vol
initial_capital: 10000
commission: 0.5

# Regime detection (optional)
hmm_floating_window: 252
hmm_retrain_interval: 50

plot: true
```

### Phase 3: Engine Generalization

**Current Issue**: BacktestEngine assumes pairs trading structure

**Changes Required**:

1. **Remove hardcoded pairs assumptions**:
   - Symbol count validation (currently asserts exactly 2)
   - Z-score model initialization (make optional)
   - Hedge beta calculation (pairs-specific)

2. **Strategy factory**:
   ```python
   def create_strategy(config: StrategyConfig, model: StrategyModel):
       """Factory to instantiate appropriate strategy."""
       if config.strategy_type == "pnd":
           return PairsTradingStrategy(...)
       elif config.strategy_type == "momentum_rotation":
           return MomentumRotationStrategy(...)
   ```

3. **DataFeed enhancements**:
   - Currently groups ticks by timestamp (good)
   - Ensure it handles variable symbol counts

### Phase 4: Portfolio Updates

**Current**: Single positions per symbol, binary open/close
**Required**: Support multiple positions, partial closes, allocation tracking

```python
class Portfolio:
    # Current: Dict[str, Trade] (one open trade per symbol)
    # Needed: Track allocations and support rebalancing
    
    def rebalance(self, target_weights: Dict[str, float], current_prices: Dict[str, float]):
        """Rebalance to target weights."""
        # Calculate current weights
        # Determine trades needed
        # Execute partial closes/opens as needed
```

### Phase 5: Risk Management Updates

**Current**: Per-position SL/TP
**Required**: Add portfolio-level risk controls

```python
class RiskManager:
    # Add:
    def check_portfolio_limits(self, portfolio: Portfolio) -> List[RiskEvent]:
        """Check portfolio-level risk limits."""
        # - Max drawdown
        # - Position concentration
        # - Sector exposure
```

## Implementation Plan

### Step 1: Refactor StrategyModel (Day 1-2)

**Goal**: Make models pluggable

1. Create `ModelRegistry` for dynamic model loading
2. Refactor `StrategyModel` to initialize models based on strategy type
3. Update `BacktestEngine` to use factory pattern for model initialization
4. Ensure backward compatibility with existing pairs trading

**Files to modify**:
- `src/bt/models/strategy_model.py`
- `src/bt/engine/backtest_engine.py`

**Testing**: Existing pairs trading tests should pass unchanged

### Step 2: Create Momentum Model (Day 3-4)

**Goal**: Implement momentum calculation

1. Create `MomentumModel` class in `src/bt/models/`
2. Support multiple momentum calculation methods:
   - Simple returns: `(P_t / P_{t-n}) - 1`
   - Risk-adjusted: Returns / Volatility
   - EMA difference: EMA_fast - EMA_slow
   - Cross-sectional: Rank vs universe
3. Add momentum indicators to `src/bt/indicators.py`:
   - `momentum_score(prices, method, params)`
   - `rank_assets(momentum_scores)`

**Files to modify**:
- `src/bt/models/momentum_model.py` (new)
- `src/bt/indicators.py` (add momentum indicators)
- `src/bt/models/__init__.py`

**Testing**: Unit tests for momentum calculations

### Step 3: Create Momentum Rotation Strategy (Day 5-6)

**Goal**: Implement rotation logic

1. Create `MomentumRotationStrategy` class
2. Implement rebalancing logic:
   - Track last rebalance timestamp
   - Determine if rebalance needed (calendar or threshold)
   - Calculate target positions
   - Generate signals for required trades
3. Handle position sizing (equal weight, risk parity, etc.)

**Files to modify**:
- `src/bt/algos/momentum_rotation.py` (new)
- `src/bt/algos/__init__.py`

**Testing**: Strategy unit tests with mock data

### Step 4: Update Portfolio for Multi-Asset (Day 7)

**Goal**: Support multiple positions and rebalancing

1. Add allocation tracking
2. Support partial closes
3. Calculate current weights vs targets
4. Handle simultaneous multi-asset positions

**Files to modify**:
- `src/bt/portfolio/__init__.py`

**Testing**: Portfolio tests with multi-asset scenarios

### Step 5: Add Portfolio-Level Risk Management (Day 8)

**Goal**: Add portfolio-level stops and limits

1. Add portfolio drawdown monitoring
2. Add position concentration checks
3. Add volatility-based position sizing

**Files to modify**:
- `src/bt/risk/__init__.py`
- `src/bt/risk/portfolio_risk.py` (new)

**Testing**: Risk manager tests

### Step 6: Update Configuration and Types (Day 9)

**Goal**: Support momentum rotation config

1. Extend `StrategyConfig` with momentum-specific fields
2. Add validation for momentum config
3. Update `StrategyType` enum

**Files to modify**:
- `src/bt/types.py`
- `src/bt/__init__.py`

### Step 7: Update Plotting (Day 10)

**Goal**: Visualize momentum rotation results

1. Add momentum score subplot
2. Add allocation pie chart over time
3. Add ranking heatmap
4. Update trade markers for multi-asset

**Files to modify**:
- `src/bt/plotting/plotting.py`

### Step 8: Integration and Testing (Day 11-12)

**Goal**: End-to-end testing

1. Create example momentum rotation YAML config
2. Run backtest and verify results
3. Compare with known momentum strategy benchmarks
4. Performance profiling

**Files to create**:
- `configs/momentum_rotation_example.yaml`
- Integration tests

### Step 9: Documentation (Day 13)

**Goal**: Document the new strategy type

1. Update README with momentum rotation section
2. Add docstrings to new classes
3. Create example usage guide

## Configuration Schema

### Momentum Rotation Config

```yaml
name: str                    # Strategy name
strategy_type: str           # "momentum_rotation"
symbols: list[str]           # Universe of assets (2+)

# Date ranges
training_start: str          # YYYY-MM-DD
training_end: str           # YYYY-MM-DD  
trading_start: str          # YYYY-MM-DD
trading_end: str            # YYYY-MM-DD

# Momentum calculation
momentum_lookback: int       # Bars to calculate momentum
momentum_method: str         # "simple_returns", "risk_adjusted", "ema_diff"
momentum_smoothing: int     # Optional EMA smoothing
momentum_weights: list      # Optional multi-lookback weights

# Rotation parameters
top_n: int                  # Number of assets to hold
rebalance_frequency: str     # "daily", "weekly", "monthly"
rebalance_threshold: float  # Min drift to trigger (0.0 = calendar only)

# Position sizing
position_sizing: str         # "equal_weight", "risk_parity", "inverse_vol", "custom"
max_position_weight: float   # Max allocation per asset (0.0-1.0)
min_position_weight: float  # Min allocation (optional)

# Risk parameters
stop_loss: float            # Portfolio stop loss % (optional)
take_profit: float          # Portfolio take profit % (optional)
volatility_target: float    # Target annual vol for sizing (optional)
max_drawdown: float        # Max portfolio drawdown % (optional)
sector_limits: dict         # Sector exposure limits (optional)

# Capital
initial_capital: float
commission: float

# Regime detection (optional)
hmm_floating_window: int
hmm_retrain_interval: int

# Output
plot: bool
```

## Testing Strategy

### Unit Tests

1. **MomentumModel**:
   - Test each momentum calculation method
   - Test edge cases (insufficient data, gaps)
   - Test multi-symbol scoring

2. **MomentumRotationStrategy**:
   - Test ranking logic
   - Test rebalancing trigger
   - Test signal generation

3. **Portfolio**:
   - Test multi-asset positions
   - Test partial closes
   - Test allocation tracking

4. **RiskManager**:
   - Test portfolio-level stops
   - Test concentration limits

### Integration Tests

1. **End-to-End Backtest**:
   - Run full backtest with momentum strategy
   - Verify equity curve calculation
   - Verify trade history

2. **Comparison Tests**:
   - Compare with known momentum strategy implementations
   - Verify similar results with same parameters

3. **Edge Cases**:
   - Universe size < top_n
   - Gaps in data
   - High volatility periods

## Potential Challenges & Mitigations

### Challenge 1: Look-Ahead Bias

**Risk**: Using future data in momentum calculation
**Mitigation**: Ensure all indicators use historical data only, add look-ahead checks in tests

### Challenge 2: Rebalancing Costs

**Risk**: Frequent rebalancing generates high transaction costs
**Mitigation**: 
- Implement threshold-based rebalancing
- Add transaction cost analysis to results
- Allow skip-rebalance logic if costs > expected benefit

### Challenge 3: Overfitting

**Risk**: Multiple parameters to optimize
**Mitigation**:
- Provide sensible defaults
- Include parameter sensitivity analysis
- Implement cross-validation framework

### Challenge 4: Survivorship Bias

**Risk**: Universe only contains current assets
**Mitigation**: Document the limitation, recommend using historical constituents if available

## Future Enhancements

### Phase 2 Features

1. **Multi-Factor Rotation**: Combine momentum with quality, value, low-vol
2. **Dynamic Lookback**: Adaptive momentum windows based on market regime
3. **Machine Learning**: ML-based factor combination
4. **Sector Rotation**: Rotate between sector ETFs based on sector momentum

### Performance Optimizations

1. **Vectorized Calculations**: Use pandas/numpy for bulk operations
2. **Caching**: Cache indicator calculations
3. **Parallel Processing**: Multi-symbol indicator calculation

## Success Criteria

1. **Functionality**:
   - [ ] Momentum rotation strategy executes without errors
   - [ ] Results are reasonable (positive Sharpe for known momentum periods)
   - [ ] Existing pairs trading still works unchanged

2. **Performance**:
   - [ ] Backtest completes in reasonable time (< 2x current)
   - [ ] Memory usage remains acceptable

3. **Code Quality**:
   - [ ] New code follows existing patterns
   - [ ] Functions under 50 LOC, classes under 150 LOC
   - [ ] Test coverage > 80% for new code
   - [ ] Type hints throughout

4. **Documentation**:
   - [ ] README updated
   - [ ] Example configs provided
   - [ ] Inline documentation clear

## Appendix: Code Structure

### New Files

```
src/
├── bt/
│   ├── algos/
│   │   ├── momentum_rotation.py      # New strategy
│   │   └── base_rotation_strategy.py   # Base class for rotation
│   ├── models/
│   │   ├── momentum_model.py           # Momentum calculations
│   │   └── model_registry.py           # Factory for models
│   ├── risk/
│   │   └── portfolio_risk.py           # Portfolio risk management
│   └── indicators.py                   # Add momentum indicators
configs/
├── momentum_rotation_example.yaml      # Example config
```

### Modified Files

```
src/
├── bt/
│   ├── __init__.py                     # Add strategy factory
│   ├── types.py                        # Extend StrategyConfig
│   ├── models/
│   │   ├── strategy_model.py           # Make models pluggable
│   │   └── __init__.py
│   ├── engine/
│   │   └── backtest_engine.py          # Remove pairs assumptions
│   ├── portfolio/
│   │   └── __init__.py                 # Support multi-asset
│   ├── risk/
│   │   └── __init__.py
│   └── plotting/
│       └── plotting.py                 # Add momentum plots
```

---

**Estimated Effort**: 13 days (including testing and documentation)
**Priority**: High - Foundational for multi-asset strategies
**Dependencies**: None (self-contained)
