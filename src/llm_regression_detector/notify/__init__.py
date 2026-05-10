"""Webhook notifications — platform-agnostic via the ``Notifier`` Protocol."""

from __future__ import annotations

from collections.abc import Callable

from llm_regression_detector.config import Settings, WebhookPlatform
from llm_regression_detector.notify.base import (
    AlertPayload,
    AlertSeverity,
    Notifier,
    NullNotifier,
)
from llm_regression_detector.notify.discord import DiscordNotifier
from llm_regression_detector.notify.generic import GenericNotifier
from llm_regression_detector.notify.google_chat import GoogleChatNotifier
from llm_regression_detector.notify.slack import SlackNotifier

NotifierFactory = Callable[[str], Notifier]

_PLATFORM_MAP: dict[WebhookPlatform, NotifierFactory] = {
    WebhookPlatform.SLACK: lambda url: SlackNotifier(webhook_url=url),
    WebhookPlatform.GOOGLE_CHAT: lambda url: GoogleChatNotifier(webhook_url=url),
    WebhookPlatform.DISCORD: lambda url: DiscordNotifier(webhook_url=url),
    WebhookPlatform.GENERIC: lambda url: GenericNotifier(webhook_url=url),
}


def build_notifier(settings: Settings) -> Notifier:
    """Pick the right notifier from settings, or a no-op if no webhook is configured."""
    platform = settings.resolved_webhook_platform
    if platform is None or settings.webhook_url is None:
        return NullNotifier()
    return _PLATFORM_MAP[platform](settings.webhook_url)


__all__ = [
    "AlertPayload",
    "AlertSeverity",
    "DiscordNotifier",
    "GenericNotifier",
    "GoogleChatNotifier",
    "Notifier",
    "NullNotifier",
    "SlackNotifier",
    "build_notifier",
]
