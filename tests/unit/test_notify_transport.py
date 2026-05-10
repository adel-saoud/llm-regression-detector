"""Tests for the shared retry policy on webhook delivery."""

from __future__ import annotations

import httpx
import pytest

from llm_regression_detector.notify.base import AlertPayload, AlertSeverity, NotifierError
from llm_regression_detector.notify.transport import post_with_retry


def _payload() -> AlertPayload:
    return AlertPayload(
        severity=AlertSeverity.WARNING,
        title="t",
        summary="s",
        accuracy_delta=-0.05,
        run_id="r",
    )


class _FakeAsyncClient:
    """Minimal context-manager stub that returns a configurable response sequence."""

    def __init__(
        self,
        statuses: list[int],
        *,
        raise_first_n: int = 0,
        timeout: float | None = None,
    ) -> None:
        self._statuses = list(statuses)
        self._raise_first_n = raise_first_n
        self.calls = 0

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
        self.calls += 1
        if self.calls <= self._raise_first_n:
            raise httpx.ConnectError("network down")
        status = self._statuses.pop(0) if self._statuses else 200
        return httpx.Response(status_code=status, request=httpx.Request("POST", url))


async def test_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAsyncClient(statuses=[503, 503, 200])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)  # type: ignore[arg-type]
    await post_with_retry(
        "https://example.com/x",
        body={},
        timeout_s=1.0,
        platform_name="Test",
    )
    assert fake.calls == 3


async def test_retries_on_connection_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAsyncClient(statuses=[200], raise_first_n=2)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)  # type: ignore[arg-type]
    await post_with_retry(
        "https://example.com/x",
        body={},
        timeout_s=1.0,
        platform_name="Test",
    )
    assert fake.calls == 3


async def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAsyncClient(statuses=[503, 503, 503, 503])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)  # type: ignore[arg-type]
    with pytest.raises(NotifierError):
        await post_with_retry(
            "https://example.com/x",
            body={},
            timeout_s=1.0,
            platform_name="Test",
        )
    assert fake.calls == 3  # stop_after_attempt(3)


async def test_non_retryable_4xx_fails_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAsyncClient(statuses=[404])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)  # type: ignore[arg-type]
    with pytest.raises(NotifierError, match="404"):
        await post_with_retry(
            "https://example.com/x",
            body={},
            timeout_s=1.0,
            platform_name="Test",
        )
    assert fake.calls == 1


def test_payload_constructible() -> None:
    # sanity — used as input across the retry tests
    assert _payload().run_id == "r"
