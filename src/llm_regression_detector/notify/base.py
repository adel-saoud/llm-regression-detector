"""Notifier Protocol and shared payload types.

Transport-level concerns (retries, HTTP error mapping) live in
:mod:`llm_regression_detector.notify.transport` so this module stays focused on
the contract every notifier must satisfy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertPayload(BaseModel):
    """Platform-neutral alert produced by the diff stage."""

    severity: AlertSeverity
    title: str
    summary: str
    accuracy_delta: float = Field(description="Signed change vs baseline (e.g. -0.08 = -8 pp)")
    regressions: list[str] = Field(
        default_factory=list[str],
        description="Case IDs that flipped pass→fail",
    )
    improvements: list[str] = Field(
        default_factory=list[str],
        description="Case IDs that flipped fail→pass",
    )
    report_url: str | None = None
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NotifierError(Exception):
    """Raised when a webhook delivery fails after exhausting retries."""


@runtime_checkable
class Notifier(Protocol):
    """Send an alert payload to an external system. Implementations format per-platform."""

    async def send(self, payload: AlertPayload) -> None: ...


class NullNotifier:
    """No-op notifier — used when no webhook URL is configured."""

    async def send(self, payload: AlertPayload) -> None:
        return None
