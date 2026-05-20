import asyncio
from functools import wraps
from typing import Callable, TypeVar, Awaitable, ParamSpec
from dataclasses import dataclass

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class RateLimitConfig:
    max_concurrent: int = 2
    min_delay_ms: int = 200
    max_retries: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 10000


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
