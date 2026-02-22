import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.syncm.ibkr_layer.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    with_rate_limiter,
    with_retry,
    RequestQueue,
    batch_items,
)


class TestRateLimitConfig:
    def test_defaults(self):
        config = RateLimitConfig()
        assert config.max_concurrent == 2
        assert config.min_delay_ms == 200
        assert config.max_retries == 3
        assert config.base_delay_ms == 500
        assert config.max_delay_ms == 10000

    def test_custom_config(self):
        config = RateLimitConfig(
            max_concurrent=5,
            min_delay_ms=100,
            max_retries=2,
            base_delay_ms=200,
            max_delay_ms=5000,
        )
        assert config.max_concurrent == 5
        assert config.min_delay_ms == 100
        assert config.max_retries == 2
        assert config.base_delay_ms == 200
        assert config.max_delay_ms == 5000


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_releases(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=1, min_delay_ms=0))
        async with limiter:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=2, min_delay_ms=0))
        active = 0
        max_active = 0

        async def task(i: int):
            nonlocal active, max_active
            async with limiter:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                active -= 1

        await asyncio.gather(*[task(i) for i in range(4)])
        assert max_active == 2

    @pytest.mark.asyncio
    async def test_min_delay_enforced(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=1, min_delay_ms=100))
        times = []

        async def task(i: int):
            async with limiter:
                times.append(asyncio.get_event_loop().time())
                await asyncio.sleep(0.01)

        await asyncio.gather(*[task(i) for i in range(3)])
        delays = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert all(d >= 0.08 for d in delays)

    @pytest.mark.asyncio
    async def test_context_manager_returns_none(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=1, min_delay_ms=0))
        async with limiter as result:
            assert result is None


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        call_count = 0

        @with_retry(max_retries=3, base_delay_ms=10)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        call_count = 0

        @with_retry(max_retries=3, base_delay_ms=10)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        call_count = 0

        @with_retry(max_retries=2, base_delay_ms=10)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await always_fails()
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_custom_exceptions(self):
        call_count = 0

        @with_retry(max_retries=1, base_delay_ms=10, retryable_exceptions=(ValueError,))
        async def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await fails_with_type_error()
        assert call_count == 1


class TestWithRateLimiter:
    @pytest.mark.asyncio
    async def test_decorator(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=1, min_delay_ms=0))
        call_times = []

        @with_rate_limiter(limiter)
        async def tracked_call():
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)

        await asyncio.gather(*[tracked_call() for _ in range(3)])
        assert len(call_times) == 3


class TestRequestQueue:
    @pytest.mark.asyncio
    async def test_enqueue_returns_result(self):
        queue = RequestQueue(max_concurrent=2)

        async def get_result():
            await asyncio.sleep(0)
            return "result"

        result = await queue.enqueue(get_result())
        assert result == "result"

    @pytest.mark.asyncio
    async def test_enqueue_many(self):
        queue = RequestQueue(max_concurrent=2)

        async def slow_add(x):
            await asyncio.sleep(0.01)
            return x + 1

        results = await queue.enqueue_many([slow_add(i) for i in range(3)])
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_enqueue_with_callback(self):
        queue = RequestQueue(max_concurrent=2)
        callback_results = []

        async def on_complete(result):
            callback_results.append(result)

        async def get_result():
            await asyncio.sleep(0)
            return "callback_test"

        await queue.enqueue_with_callback(get_result(), on_complete)
        assert callback_results == ["callback_test"]


class TestBatchItems:
    @pytest.mark.asyncio
    async def test_batch_items_basic(self):
        results = []

        async def processor(batch: list[int]) -> list[int]:
            results.extend(batch)
            return batch

        items = [1, 2, 3, 4, 5]
        batched = await batch_items(
            items, batch_size=2, processor=processor, max_concurrent=2
        )
        assert batched == [1, 2, 3, 4, 5]
        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batch_items_returns_processed(self):
        async def processor(batch: list[int]) -> list[int]:
            return [x * 2 for x in batch]

        items = [1, 2, 3]
        result = await batch_items(items, batch_size=2, processor=processor)
        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_empty_list(self):
        async def processor(batch: list[int]) -> list[int]:
            return batch

        result = await batch_items([], batch_size=2, processor=processor)
        assert result == []
