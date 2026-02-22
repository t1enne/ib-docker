"""Execution module - functional implementation.

This module provides pure functions for order execution.
All functions are immutable - they return new state rather than mutating input.
"""

# State types
from src.bt.state import (
    ExecutionParams,
    create_execution_params,
)

# Pure functions
from src.bt.execution.pure import (
    execute_signal,
    execute_risk_event,
    calculate_adverse_selection,
)

# For backward compatibility - these are deprecated
# Use the functional versions above instead
ExecutionHandler = None  # Removed - use pure functions

__all__ = [
    # State types
    "ExecutionParams",
    # Factories
    "create_execution_params",
    # Pure functions
    "execute_signal",
    "execute_risk_event",
    "calculate_adverse_selection",
    # Deprecated (for migration only)
    "ExecutionHandler",
]
