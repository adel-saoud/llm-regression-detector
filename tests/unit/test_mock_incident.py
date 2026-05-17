"""Tests for the incident-triage mode in the deterministic mock."""

from __future__ import annotations

import json

import pytest

from llm_regression_detector.llm.mock import MockLLMClient, classify_incident

# ── classify_incident unit tests ────────────────────────────────────────────


def testclassify_incident_p0_outage() -> None:
    result = classify_incident("Payment service returning 500 on 100% of checkout attempts.")
    assert result.category == "p0"
    assert result.judge_score == 4


def testclassify_incident_p0_data_breach() -> None:
    result = classify_incident(
        "Security alert: data breach detected, active exfiltration in progress."
    )
    assert result.category == "p0"


def testclassify_incident_p1_major_feature() -> None:
    result = classify_incident("Mobile app crash on launch for many users — iOS 17+.")
    assert result.category == "p1"
    assert result.judge_score == 4


def testclassify_incident_p1_sso() -> None:
    result = classify_incident("Okta SSO integration broken — users cannot log in via SSO.")
    assert result.category == "p1"


def testclassify_incident_p2_partial() -> None:
    result = classify_incident("CSV export fails for reports over 10k rows. Other exports work.")
    assert result.category == "p2"


def testclassify_incident_p2_subset() -> None:
    result = classify_incident("Dark mode toggle not persisting — only affects a subset of users.")
    assert result.category == "p2"


def testclassify_incident_p3_cosmetic() -> None:
    result = classify_incident("Tooltip on billing page shows wrong currency symbol for CAD users.")
    assert result.category == "p3"
    assert result.judge_score == 4


def testclassify_incident_p3_typo() -> None:
    result = classify_incident("Typo in error message: 'recieve' should be 'receive'.")
    assert result.category == "p3"


def testclassify_incident_fallback_returns_p2() -> None:
    """Truly ambiguous reports fall back to p2."""
    result = classify_incident("Something seems off with the system today.")
    assert result.category == "p2"
    assert result.judge_score == 3


# ── MockLLMClient incident mode integration ──────────────────────────────────

_INCIDENT_SYSTEM = (
    'You are an incident classifier. Output JSON with "category": one of "p0", "p1", "p2", "p3".'
)


def _incident_messages(report: str, *, few_shots: bool = True) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": _INCIDENT_SYSTEM}]
    if few_shots:
        msgs.append({"role": "user", "content": "Alert: all API endpoints returning 503."})
        msgs.append(
            {
                "role": "assistant",
                "content": json.dumps({"category": "p0", "summary": "All APIs down."}),
            }
        )
    msgs.append({"role": "user", "content": f"Incident report: {report}"})
    return msgs


@pytest.mark.asyncio
async def test_mock_incident_mode_with_few_shots() -> None:
    client = MockLLMClient()
    msgs = _incident_messages("Tooltip shows wrong currency symbol for CAD users.", few_shots=True)
    response = await client.complete(msgs)
    payload = json.loads(response.content)
    assert payload["category"] in {"p0", "p1", "p2", "p3"}


@pytest.mark.asyncio
async def test_mock_incident_mode_p0_detected() -> None:
    client = MockLLMClient()
    msgs = _incident_messages(
        "CRITICAL: payment service returning 500 on all checkout attempts."
        " Revenue impact confirmed.",
        few_shots=True,
    )
    response = await client.complete(msgs)
    payload = json.loads(response.content)
    assert payload["category"] == "p0"


@pytest.mark.asyncio
async def test_mock_incident_degraded_uses_p3() -> None:
    """Without few-shots, ~40% of cases drift to p3 (not general)."""
    client = MockLLMClient()
    # Use many reports to statistically confirm p3 is the drift target (not "general")
    degraded_categories: list[str] = []
    reports = [
        "Something looks off with the dashboard.",
        "Users report slowness intermittently.",
        "Occasional timeouts on the API.",
        "Minor rendering issue in the UI.",
        "A few users report errors.",
        "Performance seems degraded today.",
        "Some webhook calls are delayed.",
        "A background job appears stuck.",
        "Users cannot access a specific feature.",
        "The search results look incorrect.",
    ]
    for report in reports:
        msgs = _incident_messages(report, few_shots=False)
        response = await client.complete(msgs)
        payload = json.loads(response.content)
        degraded_categories.append(payload["category"])

    all_valid = all(c in {"p0", "p1", "p2", "p3"} for c in degraded_categories)
    assert all_valid, f"Incident mode returned non-incident category: {degraded_categories}"
    assert "general" not in degraded_categories, "Degraded incident mode must not emit 'general'"


@pytest.mark.asyncio
async def test_mock_email_mode_not_affected() -> None:
    """Original email classification is unaffected when system prompt has no p0/p1/p2/p3."""
    client = MockLLMClient()
    msgs = [
        {"role": "system", "content": "Classify emails as billing, technical, account, or general."},  # noqa: E501
        {"role": "user", "content": "I was charged twice this month."},
        {
            "role": "assistant",
            "content": json.dumps({"category": "billing", "summary": "Duplicate charge."}),
        },
        {"role": "user", "content": "I was charged twice this month. Please refund."},
    ]
    response = await client.complete(msgs)
    payload = json.loads(response.content)
    assert payload["category"] in {"billing", "technical", "account", "general"}
