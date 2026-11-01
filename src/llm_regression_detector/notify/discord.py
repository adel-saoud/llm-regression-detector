"""Discord incoming-webhook notifier (embed payload)."""

from __future__ import annotations

from typing import Any

from llm_regression_detector.notify.base import AlertPayload, AlertSeverity
from llm_regression_detector.notify.transport import post_with_retry

_COLOUR: dict[AlertSeverity, int] = {
    AlertSeverity.INFO: 0x2EB67D,
    AlertSeverity.WARNING: 0xECB22E,
    AlertSeverity.CRITICAL: 0xE01E5A,
}


class DiscordNotifier:
    """Posts alerts to a Discord incoming webhook as one embed coloured by severity."""

    def __init__(self, *, webhook_url: str, timeout_s: float = 10.0) -> None:
        self._webhook_url = webhook_url
        self._timeout_s = timeout_s

    async def send(self, payload: AlertPayload) -> None:
        """Deliver with the shared retry policy; ``NotifierError`` when it gives up."""
        await post_with_retry(
            self._webhook_url,
            body=self._format(payload),
            timeout_s=self._timeout_s,
            platform_name="Discord",
        )

    @staticmethod
    def _format(payload: AlertPayload) -> dict[str, Any]:
        """Platform body; ID lists are cut at 10 so long runs stay within message limits."""
        delta_pp = payload.accuracy_delta * 100
        fields: list[dict[str, Any]] = [
            {"name": "Accuracy Δ", "value": f"{delta_pp:+.2f} pp", "inline": True},
            {"name": "Run", "value": f"`{payload.run_id}`", "inline": True},
        ]
        if payload.regressions:
            fields.append(
                {
                    "name": f"Regressions ({len(payload.regressions)})",
                    "value": ", ".join(f"`{r}`" for r in payload.regressions[:10]),
                    "inline": False,
                }
            )
        if payload.improvements:
            fields.append(
                {
                    "name": f"Improvements ({len(payload.improvements)})",
                    "value": ", ".join(f"`{i}`" for i in payload.improvements[:10]),
                    "inline": False,
                }
            )

        embed: dict[str, Any] = {
            "title": payload.title,
            "description": payload.summary,
            "color": _COLOUR[payload.severity],
            "fields": fields,
            "timestamp": payload.timestamp.isoformat(),
        }
        if payload.report_url:
            embed["url"] = payload.report_url

        return {"embeds": [embed]}
