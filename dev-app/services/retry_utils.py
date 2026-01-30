# services/retry_utils.py
"""
Retry Utilities - Exponential backoff for IG API calls

Created: Jan 2026 as part of bulletproof trailing system

This module provides retry logic for transient failures in IG API calls.
Based on patterns from ig_auth.py but generalized for all API operations.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Tuple, Optional, Any
from functools import wraps
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger("retry_utils")

# Type variable for generic return type
T = TypeVar('T')

# Default configuration
STOP_UPDATE_TIMEOUT = 10.0  # seconds
STOP_UPDATE_MAX_RETRIES = 3
STOP_UPDATE_BACKOFF = [2, 5, 10]  # seconds between retries

# Exceptions that are safe to retry
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


@dataclass
class RetryResult:
    """Result of a retried operation"""
    success: bool
    result: Any
    attempts: int
    total_time_ms: float
    last_error: Optional[Exception] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_time_ms": self.total_time_ms,
            "last_error": str(self.last_error) if self.last_error else None
        }


async def retry_with_backoff(
    operation: Callable[[], T],
    max_retries: int = STOP_UPDATE_MAX_RETRIES,
    backoff_times: list = None,
    retryable_exceptions: Tuple = RETRYABLE_EXCEPTIONS,
    operation_name: str = "operation"
) -> RetryResult:
    """
    Execute an async operation with exponential backoff on failure.

    Args:
        operation: Async callable to execute
        max_retries: Maximum number of retry attempts
        backoff_times: List of wait times in seconds between retries
        retryable_exceptions: Tuple of exceptions that should trigger retry
        operation_name: Name for logging purposes

    Returns:
        RetryResult with success status, result, and metrics

    Example:
        result = await retry_with_backoff(
            lambda: client.put(url, data=payload),
            max_retries=3,
            operation_name="adjust_stop"
        )
        if result.success:
            return result.result
    """
    if backoff_times is None:
        backoff_times = STOP_UPDATE_BACKOFF

    start_time = datetime.utcnow()
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if attempt > 0:
                wait_time = backoff_times[min(attempt - 1, len(backoff_times) - 1)]
                logger.warning(
                    f"[RETRY] {operation_name}: Attempt {attempt + 1}/{max_retries + 1} "
                    f"after {wait_time}s backoff"
                )
                await asyncio.sleep(wait_time)

            # Execute the operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()

            # If we get here, operation succeeded
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if attempt > 0:
                logger.info(
                    f"[RETRY SUCCESS] {operation_name}: Succeeded on attempt {attempt + 1} "
                    f"after {elapsed_ms:.0f}ms total"
                )

            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                total_time_ms=elapsed_ms,
                last_error=None
            )

        except retryable_exceptions as e:
            last_error = e
            logger.warning(
                f"[RETRY ERROR] {operation_name}: Attempt {attempt + 1} failed with "
                f"{type(e).__name__}: {str(e)[:100]}"
            )

            # If this was the last attempt, don't continue
            if attempt >= max_retries:
                break

        except Exception as e:
            # Non-retryable exception - fail immediately
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                f"[RETRY ABORT] {operation_name}: Non-retryable error {type(e).__name__}: {e}"
            )
            return RetryResult(
                success=False,
                result=None,
                attempts=attempt + 1,
                total_time_ms=elapsed_ms,
                last_error=e
            )

    # All retries exhausted
    elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    logger.error(
        f"[RETRY EXHAUSTED] {operation_name}: Failed after {max_retries + 1} attempts "
        f"({elapsed_ms:.0f}ms total). Last error: {last_error}"
    )

    return RetryResult(
        success=False,
        result=None,
        attempts=max_retries + 1,
        total_time_ms=elapsed_ms,
        last_error=last_error
    )


def retry_decorator(
    max_retries: int = STOP_UPDATE_MAX_RETRIES,
    backoff_times: list = None,
    retryable_exceptions: Tuple = RETRYABLE_EXCEPTIONS
):
    """
    Decorator version of retry_with_backoff for cleaner code.

    Example:
        @retry_decorator(max_retries=3)
        async def my_api_call():
            return await client.get(url)
    """
    if backoff_times is None:
        backoff_times = STOP_UPDATE_BACKOFF

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                backoff_times=backoff_times,
                retryable_exceptions=retryable_exceptions,
                operation_name=func.__name__
            )
            if result.success:
                return result.result
            else:
                raise result.last_error or Exception(f"{func.__name__} failed after retries")

        return wrapper
    return decorator


class RetryableHttpClient:
    """
    Wrapper around httpx.AsyncClient that adds automatic retry.

    Usage:
        client = RetryableHttpClient()
        response = await client.put(url, headers=headers, data=data)
    """

    def __init__(
        self,
        timeout: float = STOP_UPDATE_TIMEOUT,
        max_retries: int = STOP_UPDATE_MAX_RETRIES,
        backoff_times: list = None
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_times = backoff_times or STOP_UPDATE_BACKOFF

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Execute HTTP request with retry"""
        kwargs.setdefault('timeout', self.timeout)

        async def do_request():
            async with httpx.AsyncClient() as client:
                return await client.request(method, url, **kwargs)

        result = await retry_with_backoff(
            do_request,
            max_retries=self.max_retries,
            backoff_times=self.backoff_times,
            operation_name=f"{method} {url.split('/')[-1]}"
        )

        if result.success:
            return result.result
        else:
            # Re-raise the last error
            raise result.last_error or httpx.NetworkError(f"Request failed after retries")

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("DELETE", url, **kwargs)


# Convenience function for simple usage
async def retry_http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = STOP_UPDATE_MAX_RETRIES,
    **kwargs
) -> httpx.Response:
    """
    Retry an HTTP request using an existing client.

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (GET, POST, PUT, DELETE)
        url: Request URL
        max_retries: Maximum retry attempts
        **kwargs: Additional request arguments

    Returns:
        httpx.Response on success

    Raises:
        Last exception encountered on failure
    """
    async def do_request():
        return await client.request(method, url, **kwargs)

    result = await retry_with_backoff(
        do_request,
        max_retries=max_retries,
        operation_name=f"{method} {url.split('/')[-1]}"
    )

    if result.success:
        return result.result
    else:
        raise result.last_error or httpx.NetworkError(f"Request failed after {max_retries} retries")
