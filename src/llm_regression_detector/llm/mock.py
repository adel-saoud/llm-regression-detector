"""Deterministic in-process LLM stub for offline development and tests.

The mock inspects the conversation, decides whether it looks like a classifier
call or a judge call, and returns a stable JSON response in the matching shape.
This lets the entire pipeline — runner, judge, diff, alerting — run end-to-end
with no API keys and no network.

When the prompt under test ships without few-shot examples, the mock simulates
a weaker model by drifting ~25% of cases to ``"general"``. That asymmetry is
what makes the self-eval demo (``v1`` vs ``v2-degraded``) detect a real
regression even in offline mode.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from llm_regression_detector.llm.client import LLMResponse

_INCIDENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "p0": (
        "100% of",
        "all users",
        "all api",
        "all checkout",
        "all logins",
        "all login",
        "all write",
        "entire app",
        "data loss",
        "data breach",
        "data deletion",
        "exfiltration",
        "corruption",
        "corrupted",
        "not resolving",
        "nxdomain",
        "ssl cert",
        "gateway is down",
        "primary node",
        "revenue impact",
        "all write operations",
        "every api endpoint",
        "all edge nodes",
        "domain app.",
        "permanently deleted",
        "deleting act",
        "session secret",
        "critical transactional",
        "password resets",
        "wrong query",
        "logged out",
    ),
    "p1": (
        "significant portion",
        "large portion",
        "many users",
        "majority",
        "mobile app crash",
        "ios app crash",
        "ios 17",
        "2fa",
        "two-factor",
        "sms not",
        "search is returning zero",
        "zero results for every",
        "search returning no results",
        "search returning no",
        "misfiring",
        "wrong amounts",
        "incorrect amounts",
        "bulk data export",
        "bulk export",
        "all bulk",
        "loading >30s",
        "30-90 seconds",
        "30 seconds to load",
        "okta",
        "sso integration",
        "all enterprise",
        "newly created accounts",
        "video playback",
        "file upload",
        "file uploads failing",
        "over 1mb",
        "notification worker",
        "email notification worker",
        "silently failing",
        "rate limiter",
        "redis key",
        "timing out regardless",
        "stuck in 'processing'",
        "stuck in processing",
        "report generation",
        "timing out for all",
    ),
    "p2": (
        "over 10k rows",
        "over 10,000",
        "subset",
        "only page 1",
        "weekends",
        "german locale",
        "gmail users",
        "billing subscription",
        "stale results",
        "delayed by",
        "wrong completion",
        "deprecated endpoint",
        "api v1",
        "dark mode toggle",
        "pagination broken",
        "slow query",
        "accounts with >",
        "slack integration",
        "all users who connected",
    ),
    "p3": (
        "tooltip",
        "typo",
        "avatar",
        "keyboard shortcut",
        "cmd+b",
        "ctrl+b",
        "empty state illustration",
        "hover state",
        "same tab instead",
        "duplicate entries",
        "disappears too quickly",
        "off-center",
        "off by exactly 1 hour",
        "off by 1 hour",
        "ie11",
        "toast notification",
        "terms of service link",
        "filter dropdown closes",
        "secondary call-to-action",
        "missing the hover",
        "says 'you wi",
        "link expires says",
        "last seen",
    ),
}


def classify_incident(text: str) -> _MockClassification:
    """Keyword-based severity classifier for incident triage prompts."""
    lowered = text.lower()
    summaries: dict[str, str] = {
        "p0": "Complete outage — all users affected, immediate response required.",
        "p1": "Major feature broken — significant user impact, page on-call.",
        "p2": "Partial degradation — subset of users affected, same-day fix needed.",
        "p3": "Minor issue — cosmetic or edge case, few users affected.",
    }
    for category, keywords in _INCIDENT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return _MockClassification(
                category=category,
                summary=summaries[category],
                judge_score=4,
            )
    return _MockClassification(
        category="p2",
        summary=summaries["p2"],
        judge_score=3,
    )


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "invoice",
        "charge",
        "charged",
        "refund",
        "payment",
        "subscription",
        "billed",
        "billing",
        "credit card",
        "plan",
        "discount",
        "vat",
        "tax",
        "proration",
        "cancel",
        "pay",
        "w-9",
        "annual",
        "upgrade",
    ),
    "technical": (
        "error",
        "bug",
        "crash",
        "crashes",
        "broken",
        "not working",
        "doesn't load",
        "500",
        "404",
        "webhook",
        "api",
        "rate limit",
        "safari",
        "iphone",
        "import",
        "stuck",
        "two-factor",
        "2fa",
        "typeerror",
        "save",
        "rolls back",
        "latency",
        "search endpoint",
        "blank",
        "console",
        "button",
        "render",
        "export",
    ),
    "account": (
        "log in",
        "login",
        "password",
        "sign in",
        "username",
        "locked out",
        "access",
        "email address",
        "teammate",
        "admin",
        "delete",
        "deleted",
        "gdpr",
        "sso",
        "okta",
        "transfer",
        "ownership",
        "remove me",
        "workspace",
        "phone",
        "profile",
        "forgot",
        "reset",
    ),
    "general": (
        "non-profit",
        "data retention",
        "soc 2",
        "compliant",
        "redesign",
        "love",
        "webinar",
        "documentation",
        "docs",
        "roadmap",
        "partnership",
        "competitor",
        "thanks",
        "appreciate",
        "wondering",
        "hello",
        "hi",
        "question",
        "info",
    ),
}


@dataclass(frozen=True, slots=True)
class _MockClassification:
    category: str
    summary: str
    judge_score: int


def _classify(text: str) -> _MockClassification:
    """First matching category wins — keyword order in ``_CATEGORY_KEYWORDS`` matters."""
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return _MockClassification(
                category=category,
                summary=f"Customer reports a {category} issue.",
                judge_score=4,
            )
    return _MockClassification(
        category="general",
        summary="Customer has a general inquiry.",
        judge_score=3,
    )


def _stable_seed(messages: list[dict[str, str]]) -> int:
    """Deterministic seed derived from the message contents."""
    payload = json.dumps(messages, sort_keys=True).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


class MockLLMClient:
    """Implements ``LLMClient`` without any network I/O."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        full_context = "\n".join(m.get("content", "") for m in messages)
        has_few_shots = any(m.get("role") == "assistant" for m in messages)
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        is_incident_mode = bool(re.search(r"\bp[0-3]\b", system_msg))
        classification = classify_incident(user_msg) if is_incident_mode else _classify(user_msg)

        if _looks_like_judge_prompt(full_context):
            # Mock judge: parse expected vs predicted category from the user message
            # and compute a real comparison. Judge quality is independent of the
            # prompt under test, so this stays the same across v1/v2.
            expected = _extract_field(user_msg, "Expected category")
            predicted = _extract_field(user_msg, "Predicted category")
            category_match = bool(expected and predicted and expected == predicted)
            score = 4 if category_match else 1
            content = json.dumps(
                {
                    "category_match": category_match,
                    "summary_score": score,
                    "rationale": f"Mock judge: expected={expected}, predicted={predicted}.",
                }
            )
        elif has_few_shots:
            content = json.dumps(
                {"category": classification.category, "summary": classification.summary},
            )
        else:
            # No few-shots → simulate a weaker prompt: misclassify harder/adversarial
            # cases. Deterministic via the input hash — same case always degrades the
            # same way. This is what makes the self-eval demo show a real regression.
            seed = _stable_seed(messages)
            if seed % 5 < 2:  # ~40% of cases drift — cleanly significant at N=53
                degraded_category = "p3" if is_incident_mode else "general"
                degraded_summary = (
                    "Severity unclear — defaulting to p3."
                    if is_incident_mode
                    else "Customer message — unclear category."
                )
                content = json.dumps(
                    {"category": degraded_category, "summary": degraded_summary},
                )
            else:
                content = json.dumps(
                    {"category": classification.category, "summary": classification.summary},
                )

        prompt_tokens = max(1, len(user_msg) // 4)
        completion_tokens = max(1, len(content) // 4)
        seed = _stable_seed(messages)
        latency_ms = 5.0 + (seed % 20)  # 5–25 ms, deterministic per input

        return LLMResponse(
            content=content,
            model="mock/deterministic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )


_JUDGE_MARKERS = re.compile(r"\b(judge|score|rate|evaluate|grading)\b", re.IGNORECASE)


def _looks_like_judge_prompt(text: str) -> bool:
    return bool(_JUDGE_MARKERS.search(text))


def _extract_field(text: str, label: str) -> str:
    """Pull the value of a ``Label: value`` line from a free-form prompt."""
    for line in text.splitlines():
        if line.startswith(f"{label}:"):
            return line.split(":", 1)[1].strip()
    return ""
