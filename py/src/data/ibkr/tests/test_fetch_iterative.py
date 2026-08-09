"""Test _fetch_candles_iterative — forward chunking for multi-year backfills.

Regression coverage for the fix that routes historical fetches through
``direction=1`` (forward) anchored at ``from_datetime``. The previous backward
variant anchored ``startTime`` at the walk cursor, which drifted into the deep
past on the second chunk of a multi-year range and triggered IBKR 50X errors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import ModuleType
from unittest.mock import AsyncMock

from ib_rest_api_client.models import (
    IserverHistoryBidAskResponse,
    SingleHistoricalBarBidAsk,
)

import pytest

from src.data.ibkr.candles import _fetch_candles_iterative, MAX_PERIOD_DAYS


def _candles_module() -> ModuleType:
    """Return the real candles submodule (unshadowed by package __init__)."""
    return import_module("src.data.ibkr.candles")


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


class _FakeResp:
    status_code = 200

    def __init__(self, items: list[SingleHistoricalBarBidAsk]):
        # A real typed response so the isinstance guard in the fetcher passes.
        self.parsed = IserverHistoryBidAskResponse(data=items)
        self.content = b""


def _spanned_items(
    span_start: datetime, span_end: datetime, step_h: float = 1.0
) -> list[SingleHistoricalBarBidAsk]:
    """Hourly bars from span_start to span_end (inclusive seam)."""
    items: list[SingleHistoricalBarBidAsk] = []
    cur = span_start
    while cur <= span_end:
        items.append(
            SingleHistoricalBarBidAsk(
                o=100.0, h=101.0, l=99.0, c=100.5, v=1000, t=_ms(cur)
            )
        )
        cur += timedelta(hours=step_h)
    return items


def _fake_history() -> AsyncMock:
    """Return an async mock that serves one 365d forward chunk per call."""

    async def handler(**_kwargs: object) -> _FakeResp:
        start = datetime.strptime(str(_kwargs["start_time"]), "%Y%m%d-%H:%M:%S")
        period_days = int(str(_kwargs["period"]).rstrip("d"))
        direction = _kwargs["direction"]
        assert getattr(direction, "value", None) == 1, "fetch must walk forward"
        # Server returns up to period_days forward from start_time.
        span_end = start + timedelta(days=period_days)
        return _FakeResp(_spanned_items(start, span_end, step_h=1.0))

    return AsyncMock(side_effect=handler)


def _collect_requests(mock: AsyncMock) -> list[tuple[str, str, int]]:
    """Return [(start_time, period, direction)] captured per call."""
    out: list[tuple[str, str, int]] = []
    for c in mock.call_args_list:
        kw = c.kwargs
        direction = kw["direction"]
        out.append(
            (
                str(kw["start_time"]),
                str(kw["period"]),
                int(getattr(direction, "value", 0)),
            )
        )
    return out


@pytest.mark.asyncio
async def test_multi_year_fetches_in_forward_chunks_within_limit(monkeypatch):
    """A ~3-year range must chunk so no single request exceeds MAX_PERIOD_DAYS
    and every start_time anchors a real past boundary (never a stale past cursor)."""
    from_dt = datetime(2020, 1, 1)
    to_dt = datetime(2023, 1, 1)

    mock = _fake_history()
    monkeypatch.setattr(
        _candles_module().get_iserver_marketdata_history,
        "asyncio_detailed",
        mock,
    )

    result = await _fetch_candles_iterative(123, "AAPL", "1h", from_dt, to_dt)

    # Non-empty result covering the full 3-year range.
    assert result, "expected candles to be fetched"
    stamps = sorted(_ts(c["timestamp"]) for c in result)
    assert stamps[0] <= from_dt
    assert stamps[-1] >= to_dt

    # Every request stays within the max period.
    requests = _collect_requests(mock)
    assert len(requests) > 1, "multi-year range must be chunked"
    for start_time, period, _dir in requests:
        period_days = int(period.rstrip("d"))
        assert period_days <= MAX_PERIOD_DAYS

    # Requests walk strictly forward from the from boundary.
    first = datetime.strptime(requests[0][0], "%Y%m%d-%H:%M:%S")
    assert first == from_dt
    for start_time, _, _dir in requests[1:]:
        dt = datetime.strptime(start_time, "%Y%m%d-%H:%M:%S")
        assert dt > from_dt, "subsequent chunks must advance past from_datetime"


@pytest.mark.asyncio
async def test_single_year_single_request_unchanged(monkeypatch):
    """A sub-365-day range still resolves to a single forward request."""
    from_dt = datetime(2025, 1, 1)
    to_dt = datetime(2025, 6, 1)

    mock = _fake_history()
    monkeypatch.setattr(
        _candles_module().get_iserver_marketdata_history,
        "asyncio_detailed",
        mock,
    )

    result = await _fetch_candles_iterative(456, "MSFT", "1h", from_dt, to_dt)

    requests = _collect_requests(mock)
    assert len(requests) == 1
    period_days = int(requests[0][1].rstrip("d"))
    assert period_days <= MAX_PERIOD_DAYS
    assert result
