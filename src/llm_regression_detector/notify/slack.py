"""Slack incoming-webhook notifier (Block Kit payload)."""

from __future__ import annotations

from typing import Any

from llm_regression_detector.notify.base import AlertPayload, AlertSeverity
from llm_regression_detector.notify.transport import post_with_retry

_COLOUR: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "#2eb67d",
    AlertSeverity.WARNING: "#ecb22e",
    AlertSeverity.CRITICAL: "#e01e5a",
}


class SlackNotifier:
    """Posts alerts to a Slack incoming webhook as Block Kit in a severity-coloured attachment."""

    def __init__(self, *, webhook_url: str, timeout_s: float = 10.0) -> None:
        self._webhook_url = webhook_url
        self._timeout_s = timeout_s

    async def send(self, payload: AlertPayload) -> None:
        """Deliver with the shared retry policy; ``NotifierError`` when it gives up."""
        await post_with_retry(
            self._webhook_url,
            body=self._format(payload),
            timeout_s=self._timeout_s,
            platform_name="Slack",
        )

    @staticmethod
    def _format(payload: AlertPayload) -> dict[str, Any]:
        """Platform body; ID lists are cut at 10 so long runs stay within message limits."""
        delta_pp = payload.accuracy_delta * 100
        fields: list[dict[str, Any]] = [
            {"type": "mrkdwn", "text": f"*Accuracy Δ*\n{delta_pp:+.2f} pp"},
            {"type": "mrkdwn", "text": f"*Run*\n`{payload.run_id}`"},
        ]
        if payload.regressions:
            fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Regressions ({len(payload.regressions)})*\n"
                    + ", ".join(f"`{r}`" for r in payload.regressions[:10]),
                }
            )
        if payload.improvements:
            fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Improvements ({len(payload.improvements)})*\n"
                    + ", ".join(f"`{i}`" for i in payload.improvements[:10]),
                }
            )

        blocks: list[dict[str, Any]] = [
            {"type": "header", "text": {"type": "plain_text", "text": payload.title}},
            {"type": "section", "text": {"type": "mrkdwn", "text": payload.summary}},
            {"type": "section", "fields": fields},
        ]
        if payload.report_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View report"},
                            "url": payload.report_url,
                        }
                    ],
                }
            )

        return {
            "attachments": [
                {
                    "color": _COLOUR[payload.severity],
                    "blocks": blocks,
                }
            ]
        }
