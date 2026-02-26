# Volatility Expansion Pullback Continuation - Implementation Plan

## Overview

Implement the strategy defined in `vol_extension_pullback_continuation.md` as a new algo in the backtest system.

---

## Phase 1: Correlation Model

### 1.1 Create `src/bt/models/correlation_model.py`

```python
class CorrelationModel:
    def __init__(self, symbols: List[str], rolling_window_size: int):
        self.symbols = symbols
        self.rolling_window_size = rolling_window_size
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def calculate_correlation_matrix(self, buffers: List[Dict[str, float]]) -> pd.DataFrame:
        """Compute rolling correlation matrix from price buffers."""
        # Build price DataFrame from buffers
        # Calculate returns correlation matrix
        # Store in self._correlation_matrix
        return self._correlation_matrix

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if self._correlation_matrix is None:
            return 0.0
        return self._correlation_matrix.loc[symbol1, symbol2]
```

### 1.2 Update `src/bt/state/types.py` - ModelState

Add optional correlation model field:

```python
@dataclass(frozen=True)
class ModelState:
    z_score: Optional[float]
    current_regime: Optional[int]
    price_buffers: Tuple[Dict[str, float], ...]
    market_data: MarketDataState
    hedge_beta: float = 1.0
    correlation_model: Optional[Any] = None  # NEW
```

### 1.3 Update `src/bt/engine/engine.py`

- **Import**: Add `from src.bt.models.correlation_model import CorrelationModel`
- **_update_models()**: Add conditional to create/update correlation model for `volatility_expansion_pullback_continuation` strategy
- Pass correlation model to ModelState

---

## Phase 2: Vol Extension Pullback Algo

### 2.1 Create `src/bt/algos/vol_extension_pullback.py`

#### Signal Phases (per symbol)

| Phase | Description |
|-------|-------------|
| `COMPRESSION` | Watching for volatility compression (ATR ratio < 0.6) |
| `BREAKOUT` | Breakout detected, waiting for pullback |
| `PULLBACK` | In pullback window (2-4 days), waiting for entry trigger |
| `ENTERED` | Position open, managing with trailing stop |
| `CLOSED` | Position closed, reset to COMPRESSION |

#### State Tracking

Store phase and metadata per symbol in a dict, passed via strategy_params or module-level:

```python
signal_state: Dict[str, dict] = {
    symbol: {
        "phase": "COMPRESSION",  # COMPRESSION, BREAKOUT, PULLBACK, ENTERED, CLOSED
        "breakout_high": 0.0,
        "breakout_date": None,
        "pullback_start": None,
        "atr_at_entry": 0.0,
        "entry_price": 0.0,
    }
}
```

#### Indicators Needed

From `strategy_params` in vol_ext.yaml:

| Indicator | Purpose |
|-----------|---------|
| ATR(14), ATR(100) | Compression detection |
| Highest High(20) | Breakout confirmation |
| Volume(30) avg | Volume confirmation |
| EMA(5), EMA(10) | Pullback measurement |
| MA(20), MA(50) | Trend filter |
| 63-day return | Momentum filter |
| EMA(20) | Exit signal |
| 10-day low | Exit signal |

#### Entry Logic (per plan doc)

```
1. COMPRESSION → BREAKOUT:
   - CompressionRatio = ATR(14) / ATR(100) < 0.6
   - AND Close > HighestHigh(20)
   - AND TrueRange > 1.5 × ATR(14)
   - AND Volume > 1.5 × 30-day avg volume

2. BREAKOUT → PULLBACK:
   - Price retraces to EMA(5-10) OR
   - Retraces 30-50% of breakout candle OR
   - 2-4 day consolidation with declining volume

3. PULLBACK → ENTERED:
   - Close > previous day's high OR
   - Bullish engulfing candle
   - AND Trend filter: MA(20) > MA(50) AND MA(50) slope > 0 AND 63-day return > 0

4. ENTERED → CLOSED:
   - Trailing stop = 3 × ATR(14) hit OR
   - Close < 10-day low OR
   - Close < EMA(20) OR
   - Time stop = 20 trading days
```

#### Position Sizing

Calculate in algo (as per plan):

```python
risk_per_trade = 0.01  # from config
stop_distance = 2.5 * atr_14
shares = (capital * risk_per_trade) / stop_distance
```

Set `qty` in TradeSignal.

#### Max Positions Check

```python
max_positions = 5  # from config
if len(state.portfolio.positions) >= max_positions:
    return []  # No entry signals
```

#### Correlation Filter

```python
correlation_model = state.model_state.correlation_model
if correlation_model:
    for existing_sym in state.portfolio.positions:
        corr = correlation_model.get_correlation(symbol, existing_sym)
        if corr > 0.8:
            return []  # Skip entry due to correlation
```

#### Key Functions

```python
def on_tick(state: BacktestState, tick: Tick, strategy_params: dict) -> List[TradeSignal]:
    """Generate trade signals based on strategy logic."""
    # Extract params
    # Get/compute indicators
    # Check phase per symbol
    # Generate signals
    return signals

def plot(state: BacktestState, config: StrategyConfig) -> PlotConfig:
    """Return price overlays for visualization."""
    # Return MAs, ATR, etc. as price overlays
```

---

## Phase 3: Integration

### 3.1 Update `src/bt/algos/__init__.py`

```python
import src.bt.algos.vol_extension_pullback

__all__ = ["pairs_trading_functional", "ema_cross", "vol_extension_pullback"]
```

### 3.2 Update `src/bt/engine/engine.py`

**Import (line ~18)**:
```python
from src.bt.algos import ema_cross, pairs_trading_functional, vol_extension_pullback
```

**_strat_wrap() (line ~215)**:
```python
elif self.config.strategy_type == "volatility_expansion_pullback_continuation":
    return vol_extension_pullback.on_tick(state, tick, self.config.strategy_params)
```

**_get_strategy_plot_fn() (line ~499)**:
```python
elif self.config.strategy_type == "volatility_expansion_pullback_continuation":
    return vol_extension_pullback.plot(state, self.config)
```

---

## Notes

1. **Partial take profit**: Add comment in `src/bt/portfolio/pure.py` - leave for later implementation
2. **Correlation model**: Uses rolling_window_size like z_model for retraining frequency
3. **Position sizing**: Algo calculates qty based on ATR, sets in TradeSignal
4. **Config params**: All strategy parameters come from `vol_ext.yaml` strategy_params section

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/bt/models/correlation_model.py` | Create |
| `src/bt/state/types.py` | Modify - add correlation_model field |
| `src/bt/engine/engine.py` | Modify - integrate correlation model + new strategy |
| `src/bt/algos/vol_extension_pullback.py` | Create |
| `src/bt/algos/__init__.py` | Modify - add export |
