"""Async batched eval runner.

Runs every golden case through:
  1. the prompt under test (model produces category + summary)
  2. the LLM-as-Judge (judge scores the prediction vs the gold label)

Concurrency is bounded by a semaphore. Failures are isolated per-case — one bad
LLM response never sinks the whole run.

The summary block computed at the end is rich on purpose: percentile latency,
Wilson 95% CI, per-category accuracy, and an optional cost estimate. Aggregate
accuracy alone is an unreliable regression signal — the report consumers
(diff/analyzer.py, report/html.py, dashboard) use these extra dimensions.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import UTC, datetime

import structlog

from llm_regression_detector.eval.dataset import (
    CaseResult,
    CategoryAccuracy,
    EvalRun,
    GoldenCase,
    JudgeScore,
    PromptSpec,
    RunSummary,
)
from llm_regression_detector.eval.parsing import extract_json_object
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.eval.stats import percentile, wilson_interval
from llm_regression_detector.llm.client import LLMClient

_log = structlog.get_logger(__name__)


class Runner:
    """Executes a prompt against the golden dataset and returns an ``EvalRun``."""

    def __init__(
        self,
        *,
        client: LLMClient,
        judge: Judge,
        concurrency: int = 10,
        cost_per_million_input_usd: float = 0.0,
        cost_per_million_output_usd: float = 0.0,
    ) -> None:
        self._client = client
        self._judge = judge
        self._semaphore = asyncio.Semaphore(concurrency)
        self._cost_in = cost_per_million_input_usd
        self._cost_out = cost_per_million_output_usd

    async def run(
        self,
        *,
        prompt: PromptSpec,
        cases: list[GoldenCase],
        run_id: str | None = None,
    ) -> EvalRun:
        run_id = run_id or _make_run_id(prompt)
        log = _log.bind(run_id=run_id, prompt=prompt.name, version=prompt.version)
        log.info("eval.run.start", cases=len(cases))

        results = await asyncio.gather(*(self._evaluate(prompt, c, log) for c in cases))
        case_lookup = {c.id: c for c in cases}
        summary = self._summarize(results, case_lookup)

        log.info(
            "eval.run.done",
            accuracy=round(summary.accuracy, 4),
            ci_low=round(summary.accuracy_ci_low, 4),
            ci_high=round(summary.accuracy_ci_high, 4),
            passed=summary.cases_passed,
            total=summary.cases_total,
            cost_usd=round(summary.estimated_cost_usd, 6),
        )
        return EvalRun(
            run_id=run_id,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            timestamp=datetime.now(UTC),
            results=results,
            summary=summary,
        )

    async def _evaluate(
        self,
        prompt: PromptSpec,
        case: GoldenCase,
        log: structlog.stdlib.BoundLogger,
    ) -> CaseResult:
        async with self._semaphore:
            messages = prompt.render_messages(case.input_email)
            try:
                response = await self._client.complete(messages, temperature=0.0)
            except Exception as exc:
                log.warning("eval.case.llm_failed", case_id=case.id, err=str(exc))
                return _error_result(case.id, f"LLM call failed: {exc}")

            predicted = _parse_prediction(response.content)
            if predicted is None:
                return _error_result(
                    case.id,
                    "Model output was not valid classifier JSON",
                    raw=response.content,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )

            try:
                judge_score = await self._judge.score(
                    case,
                    predicted_category=predicted["category"],
                    predicted_summary=predicted["summary"],
                )
            except Exception as exc:
                log.warning("eval.case.judge_failed", case_id=case.id, err=str(exc))
                judge_score = JudgeScore(
                    category_match=predicted["category"] == case.expected_category,
                    summary_score=1,
                    rationale=f"Judge failed: {exc}",
                )

            return CaseResult(
                case_id=case.id,
                topic=case.topic,
                predicted_category=predicted["category"],
                predicted_summary=predicted["summary"],
                judge=judge_score,
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                raw_output=response.content,
            )

    def _summarize(
        self,
        results: list[CaseResult],
        cases: dict[str, GoldenCase],
    ) -> RunSummary:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        # Latency percentiles only over successful calls — failed cases are
        # recorded with latency_ms=0 by ``_error_result`` and would skew the
        # distribution toward zero.
        successful_latencies = [r.latency_ms for r in results if r.error is None]
        ci = wilson_interval(passed, total)

        prompt_tokens = sum(r.prompt_tokens for r in results)
        completion_tokens = sum(r.completion_tokens for r in results)
        cost = (
            prompt_tokens / 1_000_000 * self._cost_in
            + completion_tokens / 1_000_000 * self._cost_out
        )

        return RunSummary(
            accuracy=ci.point,
            accuracy_ci_low=ci.low,
            accuracy_ci_high=ci.high,
            avg_summary_score=sum(r.judge.summary_score for r in results) / total,
            latency_p50_ms=percentile(successful_latencies, 50),
            latency_p95_ms=percentile(successful_latencies, 95),
            latency_p99_ms=percentile(successful_latencies, 99),
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            cases_total=total,
            cases_passed=passed,
            per_category=_per_category_breakdown(results, cases),
        )


def _per_category_breakdown(
    results: list[CaseResult],
    cases: dict[str, GoldenCase],
) -> list[CategoryAccuracy]:
    """Compute pass-rate per gold category. Reveals regressions the aggregate hides."""
    bucket_total: dict[str, int] = defaultdict(int)
    bucket_passed: dict[str, int] = defaultdict(int)
    for result in results:
        case = cases.get(result.case_id)
        if case is None:
            continue
        category = case.expected_category
        bucket_total[category] += 1
        if result.passed:
            bucket_passed[category] += 1
    return [
        CategoryAccuracy(
            category=category,
            accuracy=bucket_passed[category] / bucket_total[category],
            cases_total=bucket_total[category],
            cases_passed=bucket_passed[category],
        )
        for category in sorted(bucket_total)
    ]


def _parse_prediction(raw: str) -> dict[str, str] | None:
    """Extract ``{category, summary}`` from the model's raw output."""
    parsed = extract_json_object(raw)
    if parsed is None:
        return None
    category = parsed.get("category")
    summary = parsed.get("summary")
    if not isinstance(category, str) or not isinstance(summary, str):
        return None
    return {"category": category, "summary": summary}


def _error_result(
    case_id: str,
    error: str,
    *,
    raw: str = "",
    latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        predicted_category="",
        predicted_summary="",
        judge=JudgeScore(category_match=False, summary_score=1, rationale=error),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw_output=raw,
        error=error,
    )


def _make_run_id(prompt: PromptSpec) -> str:
    return f"{prompt.name}-{prompt.version}-{uuid.uuid4().hex[:8]}"
