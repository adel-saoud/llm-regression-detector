"""Shared fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_regression_detector.eval.dataset import (
    CaseResult,
    CategoryAccuracy,
    EvalRun,
    GoldenCase,
    JudgeScore,
    PromptSpec,
    RunSummary,
    load_dataset,
    load_prompt,
)
from llm_regression_detector.eval.stats import wilson_interval

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def golden_dataset() -> list[GoldenCase]:
    return load_dataset(REPO_ROOT / "golden_dataset" / "support_emails.json")


@pytest.fixture
def prompt_v1() -> PromptSpec:
    return load_prompt(REPO_ROOT / "prompts" / "classifier_v1.yaml")


def _make_case(case_id: str, *, passed: bool, score: int = 4) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        predicted_category="billing",
        predicted_summary="x",
        judge=JudgeScore(category_match=passed, summary_score=score, rationale=""),
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=5,
        raw_output="{}",
    )


def make_run(
    run_id: str,
    *,
    pass_ids: list[str],
    fail_ids: list[str],
    prompt_name: str = "support-email-classifier",
    prompt_version: str = "v1",
    per_category: list[CategoryAccuracy] | None = None,
) -> EvalRun:
    results = [_make_case(i, passed=True) for i in pass_ids] + [
        _make_case(i, passed=False, score=1) for i in fail_ids
    ]
    total = len(results)
    passed = len(pass_ids)
    ci = wilson_interval(passed, total)
    return EvalRun(
        run_id=run_id,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        timestamp=datetime.now(UTC),
        results=results,
        summary=RunSummary(
            accuracy=ci.point,
            accuracy_ci_low=ci.low,
            accuracy_ci_high=ci.high,
            avg_summary_score=sum(r.judge.summary_score for r in results) / total,
            latency_p50_ms=10.0,
            latency_p95_ms=10.0,
            latency_p99_ms=10.0,
            total_prompt_tokens=10 * total,
            total_completion_tokens=5 * total,
            estimated_cost_usd=0.0,
            cases_total=total,
            cases_passed=passed,
            per_category=per_category or [],
        ),
    )
