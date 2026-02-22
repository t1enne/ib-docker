import asyncio
from functools import wraps
from typing import Callable, TypeVar, Generic, Awaitable, ParamSpec
from dataclasses import dataclass
from collections import deque

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class RateLimitConfig:
    max_concurrent: int = 2
    min_delay_ms: int = 200
    max_retries: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 10000


class RateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self._config = config if config is not None else RateLimitConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._request_timestamps: deque[float] = deque(
            maxlen=self._config.max_concurrent * 2
        )

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    async def _throttle(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            min_interval = self._config.min_delay_ms / 1000.0

            if self._last_request_time > 0:
                elapsed = now - self._last_request_time
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

            self._last_request_time = asyncio.get_event_loop().time()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        await self._throttle()

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def with_rate_limiter(limiter: RateLimiter):
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_retry(
    max_retries: int | None = None,
    base_delay_ms: int | None = None,
    max_delay_ms: int | None = None,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    config = RateLimitConfig(
        max_retries=max_retries or 3,
        base_delay_ms=base_delay_ms or 500,
        max_delay_ms=max_delay_ms or 10000,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: BaseException | None = None
            name = func.__name__  # ty: ignore[unresolved-attribute]
            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        print(f"[{name}] Succeeded on attempt {attempt + 1}")
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    print(
                        f"[{name}] Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}"
                    )
                    if attempt < config.max_retries:
                        delay = (
                            min(
                                config.base_delay_ms * (2**attempt),
                                config.max_delay_ms,
                            )
                            / 1000.0
                        )
                        print(f"[{name}] Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"[{name}] All retries exhausted, raising: {e}")
                        raise last_exception
            raise last_exception  # type: ignore[possibly-undefined]

        return wrapper

    return decorator


class RequestQueue:
    def __init__(self, max_concurrent: int = 2):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

    async def enqueue(self, coro: Awaitable[T]) -> T:
        async def _run():
            async with self._semaphore:
                return await coro

        future = asyncio.create_task(_run())
        return await future

    async def enqueue_many(
        self,
        coros: list[Awaitable[T]],
    ) -> list[T]:
        return await asyncio.gather(*[self.enqueue(coro) for coro in coros])

    async def enqueue_with_callback(
        self,
        coro: Awaitable[T],
        on_complete: Callable[[T], Awaitable[None]],
    ) -> T:
        result = await self.enqueue(coro)
        await on_complete(result)
        return result


async def batch_items(
    items: list[T],
    batch_size: int,
    processor: Callable[[list[T]], Awaitable[list[T]]],
    max_concurrent: int = 2,
) -> list[T]:
    semaphore = asyncio.Semaphore(max_concurrent)
    semaphore_lock = asyncio.Lock()
    last_batch_time = 0.0
    min_delay = 0.2

    async def process_with_limit(batch: list[T]) -> list[T]:
        nonlocal last_batch_time
        async with semaphore:
            async with semaphore_lock:
                now = asyncio.get_event_loop().time()
                if last_batch_time > 0:
                    elapsed = now - last_batch_time
                    if elapsed < min_delay:
                        await asyncio.sleep(min_delay - elapsed)
                last_batch_time = asyncio.get_event_loop().time()
            return await processor(batch)

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results = await asyncio.gather(*[process_with_limit(b) for b in batches])
    return [item for batch_result in results for item in batch_result]
