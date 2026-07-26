"""Shared NYSE calendar singleton via exchange_calendars.

Lazy-loaded — costs nothing if unused.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exchange_calendars import ExchangeCalendar

_xcal: ExchangeCalendar | None = None


def get_xcal() -> ExchangeCalendar:
    """Return the cached XNYS calendar, loading it on first call."""
    global _xcal
    if _xcal is None:
        import exchange_calendars as xcals

        _xcal = xcals.get_calendar("XNYS")
    return _xcal


def is_non_trading_day(d: datetime) -> bool:
    """Check if a datetime falls on a non-trading day (weekend or NYSE holiday).

    Falls back to weekday-only check for dates outside the calendar's
    range (exchange_calendars typically covers ~3 years forward).
    """
    if d.weekday() >= 5:
        return True
    try:
        return not get_xcal().is_session(d.strftime("%Y-%m-%d"))
    except Exception:
        return False
