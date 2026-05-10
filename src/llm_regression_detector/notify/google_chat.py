"""Google Chat incoming-webhook notifier (Card V2 payload)."""

from __future__ import annotations

from typing import Any

from llm_regression_detector.notify.base import AlertPayload, AlertSeverity
from llm_regression_detector.notify.transport import post_with_retry

_ICON: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "✅",
    AlertSeverity.WARNING: "⚠️",
    AlertSeverity.CRITICAL: "🚨",
}


class GoogleChatNotifier:
    def __init__(self, *, webhook_url: str, timeout_s: float = 10.0) -> None:
        self._webhook_url = webhook_url
        self._timeout_s = timeout_s

    async def send(self, payload: AlertPayload) -> None:
        await post_with_retry(
            self._webhook_url,
            body=self._format(payload),
            timeout_s=self._timeout_s,
            platform_name="Google Chat",
        )

    @staticmethod
    def _format(payload: AlertPayload) -> dict[str, Any]:
        delta_pp = payload.accuracy_delta * 100
        widgets: list[dict[str, Any]] = [
            {"textParagraph": {"text": payload.summary}},
            {"decoratedText": {"topLabel": "Accuracy Δ", "text": f"{delta_pp:+.2f} pp"}},
            {"decoratedText": {"topLabel": "Run", "text": payload.run_id}},
        ]
        if payload.regressions:
            widgets.append(
                {
                    "decoratedText": {
                        "topLabel": f"Regressions ({len(payload.regressions)})",
                        "text": ", ".join(payload.regressions[:10]),
                    }
                }
            )
        if payload.improvements:
            widgets.append(
                {
                    "decoratedText": {
                        "topLabel": f"Improvements ({len(payload.improvements)})",
                        "text": ", ".join(payload.improvements[:10]),
                    }
                }
            )
        if payload.report_url:
            widgets.append(
                {
                    "buttonList": {
                        "buttons": [
                            {
                                "text": "View report",
                                "onClick": {"openLink": {"url": payload.report_url}},
                            }
                        ]
                    }
                }
            )

        return {
            "cardsV2": [
                {
                    "cardId": payload.run_id,
                    "card": {
                        "header": {
                            "title": f"{_ICON[payload.severity]} {payload.title}",
                            "subtitle": payload.timestamp.isoformat(timespec="seconds"),
                        },
                        "sections": [{"widgets": widgets}],
                    },
                }
            ]
        }
