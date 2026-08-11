"""Test _fetch_candles_iterative — backward chunking for multi-year backfills.

Regression coverage for the fix that routes historical fetches through
``direction=-1`` (backward) anchored at the newest boundary (``to_datetime``),
walking toward the past.

Why backward: IBKR's history endpoint serves ``direction=-1`` reliably for any
date range (verified live back to 2015+ for hourly bars), whereas the forward
variant (``direction=1``) returns HTTP 500 ``Chart data unavailable`` for all
requests on common gateways. Anchoring ``startTime`` at the newest boundary and
walking backward keeps the cursor advancing monotonically toward the past, so
multi-year backfills stay within the per-request period cap.
"""

from __future__ import annotations

import asyncio
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
from src.data.types import CandleDict


@pytest.fixture(autouse=True)
def _no_sleep():
    """No-op the inter-chunk cooldown so these tests run fast.

    ``_fetch_candles_iterative`` sleeps ``CHUNK_DELAY_S`` (1s) between chunks
    and backs off on 503s. Those wall-clock delays are real when the tests
    exercise a multi-year hourly range (~3-5s across a handful of chunks) and
    push ``make check`` far past its budget. The retry/backoff *logic* is what
    matters here, not the elapsed time, so patch ``asyncio.sleep`` to return
    immediately.
    """

    async def _instant(_secs: float) -> None:
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(asyncio, "sleep", _instant)
    yield
    monkeypatch.undo()


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
    """Return an async mock that serves one 365d backward chunk per call."""

    async def handler(**_kwargs: object) -> _FakeResp:
        end = datetime.strptime(str(_kwargs["start_time"]), "%Y%m%d-%H:%M:%S")
        period_days = int(str(_kwargs["period"]).rstrip("d"))
        direction = _kwargs["direction"]
        assert getattr(direction, "value", None) == -1, "fetch must walk backward"
        # Server returns up to period_days backward from end, topmost bar at end.
        span_start = end - timedelta(days=period_days, hours=1)
        span_end = end
        return _FakeResp(_spanned_items(span_start, span_end, step_h=1.0))

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
async def test_multi_year_fetches_in_backward_chunks_within_limit(monkeypatch):
    """A ~3-year range must chunk so no single request exceeds MAX_PERIOD_DAYS
    and every start_time anchors a real new boundary (never a stale forward cursor)."""
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

    # Requests walk strictly backward from the to boundary.
    first = datetime.strptime(requests[0][0], "%Y%m%d-%H:%M:%S")
    assert first == to_dt
    for start_time, _, _dir in requests[1:]:
        dt = datetime.strptime(start_time, "%Y%m%d-%H:%M:%S")
        assert dt < to_dt, "subsequent chunks must retreat past to_datetime"


@pytest.mark.asyncio
async def test_single_year_single_request_unchanged(monkeypatch):
    """A sub-365-day range still resolves to a single backward request."""
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


class _RateLimitedResp:
    status_code = 503
    parsed = None
    content = b'{"error":"Service Unavailable"}'


@pytest.mark.asyncio
async def test_503_throttles_then_retries_same_window(monkeypatch):
    """A transient 503 must throttle and retry the SAME window, not abort.

    The first call returns HTTP 503 (rate-limit); the fetcher doubles its
    inter-chunk delay and retries the identical start_time/period. Once it gets
    a 200 it proceeds and completes the range.
    """
    from_dt = datetime(2020, 1, 1)
    to_dt = datetime(2021, 1, 1)

    real = _fake_history()
    calls: list[tuple[str, str]] = []

    async def flaky_handler(
        **_kwargs: object,
    ) -> _FakeResp | _RateLimitedResp:
        start_time = str(_kwargs["start_time"])
        period = str(_kwargs["period"])
        calls.append((start_time, period))
        if len(calls) == 1:
            # First chunk is rate-limited; subsequent ones succeed.
            return _RateLimitedResp()
        return await real.side_effect(**_kwargs)

    mock = AsyncMock(side_effect=flaky_handler)
    monkeypatch.setattr(
        _candles_module().get_iserver_marketdata_history,
        "asyncio_detailed",
        mock,
    )

    result = await _fetch_candles_iterative(789, "GLD", "1h", from_dt, to_dt)

    assert result, "fetch must recover from a transient 503 and return candles"
    # The same window must have been retried after the throttle.
    assert calls[0] == calls[1], "503 must retry the identical window"
    stamps = sorted(_ts(c["timestamp"]) for c in result)
    assert stamps[0] <= from_dt
    assert stamps[-1] >= to_dt


@pytest.mark.asyncio
async def test_on_chunk_writes_each_chunk_and_returns_empty(monkeypatch):
    """When an ``on_chunk`` sink is provided, every chunk is written immediately
    and nothing is buffered: the function returns an empty list and the sink
    receives exactly one write per fetched chunk covering the full range."""
    from_dt = datetime(2020, 1, 1)
    to_dt = datetime(2023, 1, 1)

    mock = _fake_history()
    monkeypatch.setattr(
        _candles_module().get_iserver_marketdata_history,
        "asyncio_detailed",
        mock,
    )

    written: list[list[CandleDict]] = []

    def sink(chunk: list[CandleDict]) -> None:
        written.append(list(chunk))

    result = await _fetch_candles_iterative(
        999, "NVDA", "1h", from_dt, to_dt, on_chunk=sink
    )

    # Memory-backed accumulation is disabled when a sink is present.
    assert result == [], "expected no buffered candles when on_chunk is provided"

    # One write per fetched chunk, covering the full range.
    n_requests = len(mock.call_args_list)
    assert len(written) == n_requests, "one sink write per API chunk"
    all_candles = [c for chunk in written for c in chunk]
    stamps = sorted(_ts(c["timestamp"]) for c in all_candles)
    assert stamps[0] <= from_dt
    assert stamps[-1] >= to_dt
