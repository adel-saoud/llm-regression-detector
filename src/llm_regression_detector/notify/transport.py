"""Shared HTTP transport for webhook delivery.

Webhook delivery must survive transient 5xx and connection errors — losing an
alert is the worst possible outcome for a regression detector. This module
centralises retry policy in one place so every concrete notifier behaves
identically: three attempts, exponential backoff with jitter (avoids
thundering-herd collisions when multiple processes retry simultaneously),
retries on 408 / 425 / 429 / 5xx and ``httpx`` connection errors only.
"""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_regression_detector.notify.base import NotifierError

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _RetryableHTTPError(Exception):
    """Internal sentinel — raised on a retryable HTTP status to drive tenacity."""


def _retry_policy() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=2.0, jitter=0.25),
        retry=retry_if_exception_type((httpx.HTTPError, _RetryableHTTPError)),
        reraise=True,
    )


async def post_with_retry(
    url: str,
    *,
    body: dict[str, object],
    timeout_s: float,
    platform_name: str,
) -> None:
    """POST ``body`` to ``url`` with bounded retries on transient failures.

    Raises :class:`NotifierError` if all attempts fail. Used by every concrete
    notifier so retry logic lives in exactly one place.
    """
    try:
        async for attempt in _retry_policy():
            with attempt:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(url, json=body)
                if response.status_code in _RETRYABLE_STATUS:
                    raise _RetryableHTTPError(
                        f"{platform_name} returned retryable {response.status_code}"
                    )
                if response.status_code >= 400:
                    raise NotifierError(
                        f"{platform_name} webhook failed: {response.status_code} {response.text}"
                    )
    except RetryError as exc:
        raise NotifierError(
            f"{platform_name} webhook failed after retries: {exc.last_attempt.exception()}"
        ) from exc
    except (httpx.HTTPError, _RetryableHTTPError) as exc:
        raise NotifierError(f"{platform_name} webhook failed: {exc}") from exc
