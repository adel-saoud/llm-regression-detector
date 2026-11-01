"""Shared HTTP transport for webhook delivery.

Losing an alert is the worst outcome for a regression detector, so delivery
retries transient failures: three attempts with exponential backoff and jitter
(jitter keeps several processes from retrying the same webhook in lockstep).
Only 408, 425, 429, 500, 502, 503, 504 and ``httpx`` transport errors are
retried; any other 4xx or 5xx fails at once because a retry cannot fix it.
Every notifier posts through ``post_with_retry`` so they all share one policy.
"""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_regression_detector.notify.base import NotifierError

_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
"""Statuses that mean "try again later": timeout, too early, rate limit, server or gateway down."""


class _RetryableHTTPError(Exception):
    """Raised on a retryable status so tenacity, which retries on exceptions, sees it."""


def _retry_policy() -> AsyncRetrying:
    """A fresh tenacity controller per delivery; ``reraise`` surfaces the last real error.

    Worst case is three attempts and under 2 s of waiting (0.5 s then 1 s, plus jitter),
    short enough for a CI job.
    """
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
    """POST ``body`` as JSON to ``url``, retrying transient failures.

    Raises:
        NotifierError: on a non-retryable status, or once every attempt failed.
            ``platform_name`` prefixes the message so the log names the webhook.
    """
    try:
        async for attempt in _retry_policy():
            with attempt:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(url, json=body)
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    raise _RetryableHTTPError(
                        f"{platform_name} returned retryable {response.status_code}"
                    )
                if response.status_code >= 400:
                    raise NotifierError(
                        f"{platform_name} webhook failed: {response.status_code} {response.text}"
                    )
    except (httpx.HTTPError, _RetryableHTTPError) as exc:
        raise NotifierError(f"{platform_name} webhook failed after retries: {exc}") from exc
