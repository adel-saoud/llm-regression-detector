from __future__ import annotations

import httpx
import pytest

from llm_regression_detector.notify.base import AlertPayload, AlertSeverity, NotifierError
from llm_regression_detector.notify.discord import DiscordNotifier
from llm_regression_detector.notify.generic import GenericNotifier
from llm_regression_detector.notify.google_chat import GoogleChatNotifier
from llm_regression_detector.notify.slack import SlackNotifier


def _payload() -> AlertPayload:
    return AlertPayload(
        severity=AlertSeverity.CRITICAL,
        title="🚨 Critical regression",
        summary="2 regressions",
        accuracy_delta=-0.10,
        regressions=["t001", "b002"],
        improvements=[],
        report_url="https://example.com/r",
        run_id="run-xyz",
    )


@pytest.mark.parametrize(
    "notifier_cls",
    [SlackNotifier, GoogleChatNotifier, DiscordNotifier, GenericNotifier],
)
async def test_notifier_posts_payload(monkeypatch: pytest.MonkeyPatch, notifier_cls: type) -> None:
    captured: dict[str, object] = {}

    class _MockResp:
        status_code = 200
        text = ""

    class _MockClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def __aenter__(self) -> _MockClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResp:
            captured["url"] = url
            captured["json"] = json
            return _MockResp()

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    notifier = notifier_cls(webhook_url="https://example.com/hook")
    await notifier.send(_payload())
    assert captured["url"] == "https://example.com/hook"
    assert isinstance(captured["json"], dict)


async def test_notifier_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockResp:
        status_code = 500
        text = "boom"

    class _MockClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def __aenter__(self) -> _MockClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResp:
            return _MockResp()

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/x")
    with pytest.raises(NotifierError):
        await notifier.send(_payload())
