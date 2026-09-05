"""Cross-sectional research statistics for strategy construction.

`ibkr research -U <universe.json> --bench <TICKER>` scans a universe of
single names against a mandatory benchmark ETF and reports per-edge-family
statistics a quantitative researcher uses to decide whether (and which)
strategy to build.

This is pre-strategy exploration only — it reports numbers on the local DB,
never fills, costs, or backtest simulation.
"""
