# Multi-Timeframe Strategy Framework - Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to add multi-timeframe (MTF) capabilities to the existing backtesting system, enabling PineScript-style higher timeframe (HTF) analysis, lazy data resampling, and on-demand indicator computation.

---

## 1. Goals & Objectives

### Primary Goals

| Goal | Description | Success Criteria |
|------|-------------|------------------|
| **Lazy Resampling** | Load granular (tick/minute) data once, resample on-demand | Zero upfront resampling; O(1) resample cost per request |
| **HTF Access** | PineScript `security()` equivalent for HTF data access | Syntax: `security("AAPL", "1H", close_prices)` |
| **Lazy Indicators** | Compute indicators only when accessed | No compute until `.value` or `.compute()` called |
| **Zero Breaking Changes** | Existing single-timeframe strategies work unchanged | Current tests pass without modification |

---

## 2. Architecture Overview

### New Module Structure

```
src/bt/
├── timeframe/                    # NEW: Multi-timeframe infrastructure
│   ├── __init__.py
│   ├── types.py                  # Timeframe enum, bar types
│   ├── resampler.py              # Lazy resampling engine
│   ├── security.py               # PineScript security() function
│   ├── indicator.py              # Lazy indicator wrapper
│   └── cache.py                  # LRU cache for resampled data
│
├── engine/
│   ├── backtest_engine.py        # MODIFY: Accept MTF config
│   ├── mtf_datafeed.py           # NEW: Multi-timeframe data feed
│   └── bar_synchronizer.py       # NEW: Sync bars across timeframes
│
├── algos/
│   ├── strategy_protocol.py     # MODIFY: Add MTF interface
│   └── mtf_strategy.py           # NEW: Base MTF strategy class
│
└── indicators/                   # NEW: Indicator library
    ├── __init__.py
    ├── sma.py
    ├── ema.py
    ├── rsi.py
    └── macd.py
```

---

## 3. Implementation Details

### 3.1 Timeframe Types (`src/bt/timeframe/types.py`)

```python
from enum import Enum
from dataclasses import dataclass
import pandas as pd

class Timeframe(Enum):
    """Standardized timeframe enumeration"""
    TICK = "tick"
    SECOND = "1S"
    MINUTE = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    HOUR = "1H"
    FOUR_HOUR = "4H"
    DAY = "1D"
    WEEK = "1W"
    MONTH = "1M"

    @classmethod
    def from_string(cls, s: str) -> "Timeframe":
        mapping = {
            "tick": cls.TICK, "1S": cls.SECOND, "1min": cls.MINUTE,
            "5min": cls.FIVE_MIN, "15min": cls.FIFTEEN_MIN,
            "30min": cls.THIRTY_MIN, "1H": cls.HOUR, "4H": cls.FOUR_HOUR,
            "1D": cls.DAY, "1W": cls.WEEK, "1M": cls.MONTH,
        }
        return mapping[s]

@dataclass
class Bar:
    """OHLCV bar representation"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

@dataclass
class TimeframeData:
    """Container for symbol data at specific timeframe"""
    symbol: str
    timeframe: Timeframe
    df: pd.DataFrame  # Columns: open, high, low, close, volume
```

### 3.2 Lazy Resampler (`src/bt/timeframe/resampler.py`)

**Design Principles:**
- Load granular data once (tick/minute)
- Resample only when requested (lazy)
- Cache resampled results (LRU)
- Support any aggregation: OHLC, OHLCV, VWAP, etc.

```python
class LazyResampler:
    def __init__(self, 
                 granular_data: Dict[str, pd.DataFrame],
                 max_cache_size: int = 100):
        self._granular = granular_data
        self._cache = LRUCache(max_cache_size)

    def resample(self,
                 symbol: str,
                 target_tf: Timeframe,
                 column: str = "close",
                 aggregation: str = "ohlc") -> pd.DataFrame:
        """Lazily resample symbol data to target timeframe with caching."""
        cache_key = (symbol, target_tf, column, aggregation)
        
        if cached := self._cache.get(cache_key):
            return cached
        
        result = self._do_resample(symbol, target_tf, column, aggregation)
        self._cache.set(cache_key, result)
        return result

    def _do_resample(self, symbol: str, tf: Timeframe, 
                     column: str, aggregation: str) -> pd.DataFrame:
        """Perform actual resampling operation using pandas resample."""
        rule = {
            Timeframe.TICK: "ms", Timeframe.SECOND: "1S",
            Timeframe.MINUTE: "1min", Timeframe.FIVE_MIN: "5min",
            Timeframe.FIFTEEN_MIN: "15min", Timeframe.THIRTY_MIN: "30min",
            Timeframe.HOUR: "1H", Timeframe.FOUR_HOUR: "4H",
            Timeframe.DAY: "1D", Timeframe.WEEK: "1W", Timeframe.MONTH: "1M",
        }[tf]

        df = self._granular[symbol]
        
        if aggregation == "close":
            return df[column].resample(rule).last().dropna()

        agg_dict = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
            "volume": ("volume", "sum"),
        }
        return df.resample(rule).agg(**agg_dict).dropna()

    def invalidate_cache(self, symbol: Optional[str] = None):
        """Clear cache for symbol or all."""
        if symbol:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == symbol]
            for k in keys_to_remove:
                self._cache.delete(k)
        else:
            self._cache.clear()
```

### 3.3 LRU Cache (`src/bt/timeframe/cache.py`)

```python
from typing import Any, Dict, Optional
import pandas as pd

class LRUCache:
    """Simple LRU cache for resampled DataFrames."""
    
    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._data: Dict[Any, pd.DataFrame] = {}
        self._order: list = []  # LRU order (oldest → newest)
    
    def get(self, key: Any) -> Optional[pd.DataFrame]:
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]
    
    def set(self, key: Any, value: pd.DataFrame):
        if key in self._data:
            self._order.remove(key)
        elif len(self._order) >= self._max_size:
            evict_key = self._order.pop(0)
            del self._data[evict_key]
        self._data[key] = value
        self._order.append(key)
    
    def delete(self, key: Any):
        if key in self._data:
            del self._data[key]
            self._order.remove(key)
    
    def clear(self):
        self._data.clear()
        self._order.clear()
    
    def keys(self):
        return list(self._data.keys())
```

### 3.4 PineScript `security()` Function (`src/bt/timeframe/security.py`)

```python
import pandas as pd
from typing import Union, Optional
from .types import Timeframe
from .resampler import LazyResampler

class HTFContext:
    """Higher timeframe context passed to strategies."""
    
    def __init__(self, symbol: str, timeframe: Timeframe,
                 resampler: LazyResampler, current_index: int):
        self._symbol = symbol
        self._tf = timeframe
        self._resampler = resampler
        self._current_index = current_index
        self._cached_data: Optional[pd.DataFrame] = None
    
    @property
    def close(self) -> pd.Series:
        data = self._get_data()
        return data["close"].iloc[:self._current_index + 1]
    
    @property
    def current_close(self) -> float:
        return self.close.iloc[-1]
    
    @property
    def previous_close(self) -> float:
        if len(self.close) < 2:
            return self.current_close
        return self.close.iloc[-2]
    
    def _get_data(self) -> pd.DataFrame:
        if self._cached_data is None:
            self._cached_data = self._resampler.resample(
                self._symbol, self._tf, "close", "ohlc"
            )
        return self._cached_data


class HTFDataPool:
    """Central registry for HTF data access."""
    
    def __init__(self, resampler: LazyResampler):
        self._resampler = resampler
        self._contexts: dict = {}
    
    def get_context(self, symbol: str, tf: Timeframe, 
                    current_idx: int) -> HTFContext:
        key = (symbol, tf)
        if key not in self._contexts:
            self._contexts[key] = HTFContext(symbol, tf, self._resampler, current_idx)
        else:
            self._contexts[key]._current_index = current_idx
        return self._contexts[key]


def security(symbol: str, timeframe: Timeframe,
             data: Union[pd.Series, str],
             resampler: Optional[LazyResampler] = None) -> pd.Series:
    """
    PineScript-equivalent security() function.
    
    Returns symbol data at specified timeframe.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL")
        timeframe: Target timeframe
        data: Source data (Series or column name)
        resampler: LazyResampler instance (auto-injected by engine)
    
    Returns:
        Series at target timeframe, aligned to source timestamps
    """
    if resampler is None:
        raise ValueError("security() requires LazyResampler (auto-injected in strategy)")
    
    if isinstance(data, str):
        granular = resampler._granular.get(symbol)
        if granular is None:
            raise ValueError(f"No data for symbol: {symbol}")
        source = granular[data]
    else:
        source = data
    
    result = resampler.resample(symbol, timeframe, column=data, aggregation="close")
    aligned = result.reindex(source.index, method='ffill')
    return aligned
```

### 3.5 Lazy Indicator Framework (`src/bt/timeframe/indicator.py`)

```python
import pandas as pd
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod

class LazyIndicator(ABC):
    """Base class for lazily-computed indicators."""
    
    def __init__(self, source: Union[pd.Series, np.ndarray, "LazyIndicator"],
                 cached: bool = True):
        self._source = source
        self._cached = cached
        self._computed: Optional[pd.Series] = None
    
    @abstractmethod
    def _compute(self, source: pd.Series) -> pd.Series:
        pass
    
    @property
    def series(self) -> pd.Series:
        if self._computed is None:
            source_series = self._get_source_series()
            self._computed = self._compute(source_series)
        return self._computed
    
    @property
    def value(self) -> float:
        series = self.series
        return series.iloc[-1] if len(series) > 0 else np.nan
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return LazySlice(self, key.start, key.stop)
        return self.series.iloc[key]
    
    def _get_source_series(self) -> pd.Series:
        if isinstance(self._source, LazyIndicator):
            return self._source.series
        return pd.Series(self._source) if not isinstance(self._source, pd.Series) else self._source


class LazySMA(LazyIndicator):
    """Lazy Simple Moving Average"""
    
    def __init__(self, source, window: int, cached: bool = True):
        super().__init__(source, cached=cached)
        self._window = window
    
    def _compute(self, source: pd.Series) -> pd.Series:
        return source.rolling(window=self._window, min_periods=1).mean()


class LazyEMA(LazyIndicator):
    """Lazy Exponential Moving Average"""
    
    def __init__(self, source, window: int, cached: bool = True):
        super().__init__(source, cached=cached)
        self._window = window
    
    def _compute(self, source: pd.Series) -> pd.Series:
        return source.ewm(span=self._window, adjust=False).mean()


class LazyRSI(LazyIndicator):
    """Lazy Relative Strength Index"""
    
    def __init__(self, source, window: int = 14, cached: bool = True):
        super().__init__(source, cached=cached)
        self._window = window
    
    def _compute(self, source: pd.Series) -> pd.Series:
        delta = source.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=self._window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self._window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class LazyMACD(LazyIndicator):
    """Lazy MACD"""
    
    def __init__(self, source, fast: int = 12, slow: int = 26,
                 signal: int = 9, cached: bool = True):
        super().__init__(source, cached=cached)
        self._fast = fast
        self._slow = slow
        self._signal = signal
    
    def _compute(self, source: pd.Series) -> dict:
        fast_ema = source.ewm(span=self._fast, adjust=False).mean()
        slow_ema = source.ewm(span=self._slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_ema.ewm(span=self._signal, adjust=False).mean()
        return {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}


# Convenience functions
def sma(source: Union[pd.Series, LazyIndicator], window: int) -> LazySMA:
    return LazySMA(source, window)

def ema(source: Union[pd.Series, LazyIndicator], window: int) -> LazyEMA:
    return LazyEMA(source, window)

def rsi(source: Union[pd.Series, LazyIndicator], window: int = 14) -> LazyRSI:
    return LazyRSI(source, window)

def macd(source: Union[pd.Series, LazyIndicator],
         fast: int = 12, slow: int = 26, signal: int = 9) -> LazyMACD:
    return LazyMACD(source, fast, slow, signal)
```

### 3.6 MTF Strategy Base (`src/bt/algos/mtf_strategy.py`)

```python
from typing import Dict, Optional, List, Any
from ..engine.types import Tick, TradeSignal
from ..timeframe.types import Timeframe, HTFContext
from ..timeframe.resampler import LazyResampler
from .strategy_protocol import StrategyProtocol

class MTFStrategy(StrategyProtocol):
    """Base class for multi-timeframe strategies."""
    
    def __init__(self):
        self._resampler: Optional[LazyResampler] = None
        self._htf_contexts: Dict[str, Dict[Timeframe, HTFContext]] = {}
    
    def _set_resampler(self, resampler: LazyResampler):
        self._resampler = resampler
    
    def _update_htf_context(self, symbol: str, tf: Timeframe, ctx: HTFContext):
        if symbol not in self._htf_contexts:
            self._htf_contexts[symbol] = {}
        self._htf_contexts[symbol][tf] = ctx
    
    def _get_htf_context(self, symbol: str, tf: Timeframe) -> Optional[HTFContext]:
        return self._htf_contexts.get(symbol, {}).get(tf)
    
    def security(self, symbol: str, tf: Timeframe, data: Any) -> Any:
        """PineScript-equivalent security() with automatic injection."""
        if self._resampler is None:
            raise RuntimeError("security() requires MTFStrategy base class")
        from ..timeframe.security import security as _security
        return _security(symbol, tf, data, resampler=self._resampler)
    
    def get_htf_trend(self, symbol: str, tf: Timeframe, window: int = 20) -> str:
        """Get HTF trend direction: UP, DOWN, or NEUTRAL."""
        ctx = self._get_htf_context(symbol, tf)
        if ctx is None:
            return "NEUTRAL"
        close = ctx.close
        if len(close) < window:
            return "NEUTRAL"
        sma_val = close.rolling(window).mean().iloc[-1]
        current = ctx.current_close
        if current > sma_val * 1.02:
            return "UP"
        elif current < sma_val * 0.98:
            return "DOWN"
        return "NEUTRAL"
```

### 3.7 Engine Integration (`src/bt/engine/backtest_engine.py`)

```python
from ..timeframe.types import Timeframe
from ..timeframe.resampler import LazyResampler
from ..timeframe.security import HTFDataPool

class BacktestEngine:
    def __init__(self, # ... existing params ...
                 secondary_timeframes: Optional[List[Timeframe]] = None,
                 htf_cache_size: int = 50):
        # ... existing init ...
        self._secondary_tfs = secondary_timeframes or []
        self._htf_pool: Optional[HTFDataPool] = None
        self._resampler: Optional[LazyResampler] = None
        self._htf_cache_size = htf_cache_size
    
    def run(self):
        # ... existing data loading ...
        
        # NEW: Initialize resampler and HTF pool
        self._resampler = LazyResampler(
            granular_data=self._granular_data,
            max_cache_size=self._htf_cache_size
        )
        self._htf_pool = HTFDataPool(self._resampler)
        
        # Inject resampler into strategy
        if hasattr(self.strategy, '_set_resampler'):
            self.strategy._set_resampler(self._resampler)
        
        # ... existing tick loop ...
        
        # NEW: Update HTF contexts on each tick
        for tf in self._secondary_tfs:
            for symbol in self.symbols:
                htf_data = self._resampler.resample(symbol, tf, "close", "ohlc")
                current_idx = len(htf_data) - 1
                ctx = self._htf_pool.get_context(symbol, tf, current_idx)
                if hasattr(self.strategy, '_update_htf_context'):
                    self.strategy._update_htf_context(symbol, tf, ctx)
```

---

## 4. Example Usage

### MTF Pairs Strategy

```python
from bt.algos.mtf_strategy import MTFStrategy
from bt.timeframe.types import Timeframe
from bt.timeframe.indicator import sma, rsi

class MTFPairsStrategy(MTFStrategy):
    def __init__(self, htf: Timeframe = Timeframe.DAY):
        super().__init__()
        self._htf = htf
    
    def on_tick(self, tick, z_score, open_trade):
        # Use security() with auto-injected resampler
        daily_close = self.security("SPY", Timeframe.DAY, "close")
        daily_sma = sma(daily_close, 20)
        
        # Check HTF trend
        trend = self.get_htf_trend("SPY", self._htf)
        
        # Entry logic with HTF filter
        if z_score > 2.0 and trend != "DOWN":
            return [self._long(tick)]
        elif z_score < -2.0 and trend != "UP":
            return [self._short(tick)]
        
        return []
```

---

## 5. API Reference

### Lazy Resampler

```python
resampler = LazyResampler(granular_data={"AAPL": df_minute}, max_cache_size=100)
hourly = resampler.resample("AAPL", Timeframe.HOUR, "close", "ohlc")
resampler.invalidate_cache("AAPL")
```

### security() Function

```python
daily_close = security("AAPL", Timeframe.DAY, "close")
daily_sma = sma(daily_close, 20)
```

### Lazy Indicators

```python
sma20 = sma(close_series, 20)     # Deferred
current = sma20.value             # Compute current only
full = sma20.series               # Compute full series
slice_val = sma20[10:20]         # Partial
```

---

## 6. Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/bt/timeframe/__init__.py` | Module exports |
| `src/bt/timeframe/types.py` | Timeframe enum, types |
| `src/bt/timeframe/cache.py` | LRU cache |
| `src/bt/timeframe/resampler.py` | Lazy resampling |
| `src/bt/timeframe/security.py` | security() function |
| `src/bt/timeframe/indicator.py` | Lazy indicators |
| `src/bt/engine/mtf_datafeed.py` | MTF data feed |
| `src/bt/engine/bar_synchronizer.py` | Bar sync |
| `src/bt/algos/mtf_strategy.py` | MTF strategy base |

### Modified Files

| File | Changes |
|------|---------|
| `src/bt/engine/backtest_engine.py` | Add MTF support |
| `src/bt/algos/strategy_protocol.py` | Add MTF interface |

---

## 7. Testing

```bash
uv run pytest src/bt/ -v
uv tool run ty check src/bt/
```

---

## 8. Summary

The implementation adds:

1. **LazyResampler**: O(1) resampling with LRU cache
2. **security()**: PineScript-equivalent for HTF access
3. **LazyIndicator**: Compute-on-access indicators
4. **MTFStrategy**: Base class with HTF context injection
5. **Backward compatibility**: Existing strategies work unchanged
