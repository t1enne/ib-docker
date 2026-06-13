"""Live trading engine — production-grade trading on top of the bt backtesting primitives.

Provides:
- BarAggregator: tick → bar conversion (time-based, tick-count, volume)
- LiveOrderBuilder: TradeSignal → IBKR order lifecycle
- PortfolioAdapter: reconcile bt PortfolioState with IBKR account
- LiveEngine: main daemon loop orchestrating the pipeline
"""
