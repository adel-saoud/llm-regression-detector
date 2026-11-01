from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
    for var in ("HF_TOKEN", "GEMINI_API_KEY", "LRD_CUSTOM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.is_mock_mode is True
    assert settings.resolved_webhook_platform is None


def test_settings_real_mode_with_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.is_mock_mode is False


def test_webhook_platform_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LRD_WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/x")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.resolved_webhook_platform == WebhookPlatform.SLACK


def test_misspelled_env_var_is_rejected_with_closest_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LRD_CONCURENCY", "4")
    with pytest.raises(ValidationError, match=r"LRD_CONCURENCY \(did you mean LRD_CONCURRENCY\?\)"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_old_unprefixed_name_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Names renamed to the LRD_ prefix are not read under their old name."""
    monkeypatch.setenv("WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/x")
    monkeypatch.delenv("LRD_WEBHOOK_URL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.webhook_url is None


def test_unknown_key_in_dotenv_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LRD_LOG_LEVL=DEBUG\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="lrd_log_levl"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_vendor_credential_names_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g_test_value")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.gemini_api_key is not None


def test_judge_equal_to_model_under_test_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-preference bias: a model rates its own answers higher."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama_chat/qwen3.5:4b")
    monkeypatch.setenv("LRD_JUDGE_MODEL", "ollama/qwen3.5:4b")
    with pytest.raises(ValidationError, match="LRD_JUDGE_MODEL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_default_judge_differs_from_default_model_under_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.primary_model == settings.ollama_model
    assert settings.judge_model != settings.primary_model
