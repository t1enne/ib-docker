import pytest
import asyncio
from unittest.mock import AsyncMock

from src.syncm.ibkr_layer.rate_limiter import (
    RateLimitConfig,
    with_retry,
    _set_monotonic_sleep,
)


@pytest.fixture(autouse=True)
def _mock_sleep():
    """Override the monotonic sleep with an instant mock for all retry tests."""
    mock = AsyncMock()
    _set_monotonic_sleep(mock)
    yield mock
    _set_monotonic_sleep(asyncio.sleep)


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


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self, _mock_sleep):
        call_count = 0

        @with_retry(max_retries=3, base_delay_ms=10)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1
        assert _mock_sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_retry_then_success(self, _mock_sleep):
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
        # 2 failures before success = 2 retry sleeps
        assert _mock_sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, _mock_sleep):
        call_count = 0

        @with_retry(max_retries=2, base_delay_ms=10)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await always_fails()
        assert call_count == 3
        # 3 calls (1 original + 2 retries) = 2 sleeps
        assert _mock_sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_custom_exceptions(self, _mock_sleep):
        call_count = 0

        @with_retry(max_retries=1, base_delay_ms=10, retryable_exceptions=(ValueError,))
        async def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await fails_with_type_error()
        assert call_count == 1
        # TypeError is not retryable → no retry, no sleep
        assert _mock_sleep.await_count == 0
