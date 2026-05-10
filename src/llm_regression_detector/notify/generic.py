"""Generic JSON-webhook notifier — posts the raw ``AlertPayload`` to any endpoint."""

from __future__ import annotations

from llm_regression_detector.notify.base import AlertPayload
from llm_regression_detector.notify.transport import post_with_retry


class GenericNotifier:
    def __init__(self, *, webhook_url: str, timeout_s: float = 10.0) -> None:
        self._webhook_url = webhook_url
        self._timeout_s = timeout_s

    async def send(self, payload: AlertPayload) -> None:
        await post_with_retry(
            self._webhook_url,
            body=payload.model_dump(mode="json"),
            timeout_s=self._timeout_s,
            platform_name="Generic",
        )
