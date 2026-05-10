from __future__ import annotations

import pytest

from llm_regression_detector.config import Settings, WebhookPlatform, detect_platform


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://hooks.slack.com/services/T0/B0/xxx", WebhookPlatform.SLACK),
        ("https://chat.googleapis.com/v1/spaces/AAA/messages?key=k", WebhookPlatform.GOOGLE_CHAT),
        ("https://discord.com/api/webhooks/123/abc", WebhookPlatform.DISCORD),
        ("https://discordapp.com/api/webhooks/123/abc", WebhookPlatform.DISCORD),
        ("https://example.com/webhook", WebhookPlatform.GENERIC),
    ],
)
def test_detect_platform(url: str, expected: WebhookPlatform) -> None:
    assert detect_platform(url) == expected


def test_settings_mock_mode_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_TOKEN", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.is_mock_mode is True
    assert settings.resolved_webhook_platform is None


def test_settings_real_mode_with_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.is_mock_mode is False


def test_webhook_platform_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/x")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.resolved_webhook_platform == WebhookPlatform.SLACK
