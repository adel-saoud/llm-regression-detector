"""Async batched eval runner.

Runs every golden case through:
  1. an output source: the prompt under test (``run``), or answers the system
     under test already produced, read from a predictions file (``score``)
  2. the judge (scores the prediction against the checks the golden case defines)

Both sources share the judge, summary and error handling, so a run scored offline
diffs against a baseline exactly like a prompt run.

Concurrency is bounded by a semaphore. Failures are isolated per-case: one bad
LLM response never sinks the whole run.

The summary block computed at the end is rich on purpose: percentile latency,
Wilson 95% CI, per-category accuracy, and an optional cost estimate. Aggregate
accuracy alone is an unreliable regression signal: the report consumers
(diff/analyzer.py, report/html.py, dashboard) use these extra dimensions.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import structlog

from llm_regression_detector.eval.dataset import (
    CaseResult,
    CategoryAccuracy,
    EvalRun,
    GoldenCase,
    PredictionRecord,
    PromptSpec,
    RunSummary,
)
from llm_regression_detector.eval.parsing import Prediction, parse_prediction
from llm_regression_detector.eval.scorer import Judge, fallback_score
from llm_regression_detector.eval.stats import percentile, wilson_interval
from llm_regression_detector.llm.client import LLMClient

_log = structlog.get_logger(__name__)

UNCATEGORIZED = "uncategorized"
"""``per_category`` bucket for cases without an ``expected_category``."""

MISSING_PREDICTION_ERROR = "No prediction for this case id"
"""Error recorded on a case that the predictions file does not answer."""


@dataclass(frozen=True)
class _CaseOutput:
    """What an output source produced for one case, before judging.

    ``error`` is set when there is nothing to judge; ``prediction`` is ``None``
    when JSON output could not be parsed.
    """

    raw: str = ""
    prediction: Prediction | None = None
    error: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_OutputSource = Callable[[GoldenCase, structlog.stdlib.BoundLogger], Awaitable[_CaseOutput]]
"""Produces the output for one case. Receives the run-scoped logger so a source's
own warnings carry ``run_id``."""


class Runner:
    """Scores golden cases and returns an ``EvalRun``.

    ``concurrency`` bounds the cases in flight. Consensus judge calls run inside
    their case's slot, so model calls in flight can reach ``concurrency * consensus_n``.
    """

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
        self._cost_per_million_input_usd = cost_per_million_input_usd
        self._cost_per_million_output_usd = cost_per_million_output_usd

    async def run(
        self,
        *,
        prompt: PromptSpec,
        cases: list[GoldenCase],
        run_id: str | None = None,
    ) -> EvalRun:
        """Generate an answer per case with the prompt under test, then judge it."""

        async def generate(case: GoldenCase, log: structlog.stdlib.BoundLogger) -> _CaseOutput:
            messages = prompt.render_messages(case.input_text)
            try:
                response = await self._client.complete(messages, temperature=0.0)
            except Exception as exc:
                log.warning("eval.case.llm_failed", case_id=case.id, err=str(exc))
                return _CaseOutput(error=f"LLM call failed: {exc}")
            return _CaseOutput(
                raw=response.content,
                prediction=parse_prediction(response.content, prompt.output_format),
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

        return await self._execute(
            name=prompt.name, version=prompt.version, cases=cases, source=generate, run_id=run_id
        )

    async def score(
        self,
        *,
        name: str,
        version: str,
        cases: list[GoldenCase],
        predictions: list[PredictionRecord],
        run_id: str | None = None,
    ) -> EvalRun:
        """Judge answers the system under test already produced (offline scoring).

        ``name`` plays the role of the prompt name: it keys the stored baseline that
        later runs diff against. A case without a prediction fails as an error, so a
        crashed system cannot pass by answering fewer cases.
        """
        by_id = {p.id: p for p in predictions}
        unknown = sorted(set(by_id) - {c.id for c in cases})
        if unknown:
            raise ValueError(f"Predictions for ids not in the dataset: {', '.join(unknown)}")

        async def read(case: GoldenCase, log: structlog.stdlib.BoundLogger) -> _CaseOutput:
            return _output_from_record(by_id.get(case.id))

        return await self._execute(
            name=name, version=version, cases=cases, source=read, run_id=run_id
        )

    async def _execute(
        self,
        *,
        name: str,
        version: str,
        cases: list[GoldenCase],
        source: _OutputSource,
        run_id: str | None,
    ) -> EvalRun:
        """Run every case through ``source`` and the judge, then summarize the run."""
        run_id = run_id or _make_run_id(name, version)
        log = _log.bind(run_id=run_id, prompt=name, version=version)
        log.info("eval.run.start", cases=len(cases))

        results = await asyncio.gather(*(self._evaluate(c, source, log) for c in cases))
        summary = self._summarize(results, {c.id: c for c in cases})

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
            prompt_name=name,
            prompt_version=version,
            timestamp=datetime.now(UTC),
            results=results,
            summary=summary,
        )

    async def _evaluate(
        self,
        case: GoldenCase,
        source: _OutputSource,
        log: structlog.stdlib.BoundLogger,
    ) -> CaseResult:
        """One case under the semaphore; any failure becomes a failed result, never a raise."""
        async with self._semaphore:
            output = await source(case, log)
            error = output.error or _output_error(case, output.prediction)
            if output.prediction is None or error is not None:
                return _error_result(case, error or "", output)
            prediction = output.prediction

            try:
                judge_score = await self._judge.score(case, prediction)
            except Exception as exc:
                log.warning("eval.case.judge_failed", case_id=case.id, err=str(exc))
                judge_score = fallback_score(case, prediction.category, f"Judge failed: {exc}")

            return CaseResult(
                case_id=case.id,
                topic=case.topic,
                predicted_category=prediction.category or "",
                predicted_summary=prediction.summary,
                predicted_answer=prediction.answer,
                judge=judge_score,
                latency_ms=output.latency_ms,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                raw_output=output.raw,
            )

    def _summarize(
        self,
        results: list[CaseResult],
        cases: dict[str, GoldenCase],
    ) -> RunSummary:
        """Aggregate metrics; cost is tokens times the configured price per million."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        # Latency percentiles only over successful calls: failed cases are
        # recorded with latency_ms=0 by ``_error_result`` and would skew the
        # distribution toward zero.
        successful_latencies = [r.latency_ms for r in results if r.error is None]
        ci = wilson_interval(passed, total)

        prompt_tokens = sum(r.prompt_tokens for r in results)
        completion_tokens = sum(r.completion_tokens for r in results)
        cost = (
            prompt_tokens / 1_000_000 * self._cost_per_million_input_usd
            + completion_tokens / 1_000_000 * self._cost_per_million_output_usd
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
    """Compute pass-rate per gold category. Reveals regressions the aggregate hides.

    Cases without an ``expected_category`` are grouped under ``UNCATEGORIZED``.
    """
    bucket_total: dict[str, int] = defaultdict(int)
    bucket_passed: dict[str, int] = defaultdict(int)
    for result in results:
        case = cases.get(result.case_id)
        if case is None:
            continue
        category = case.expected_category or UNCATEGORIZED
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


def _output_error(case: GoldenCase, prediction: Prediction | None) -> str | None:
    """Why the model output cannot be scored for this case, or ``None`` when it can."""
    if prediction is None:
        return "Model output was not a JSON object"
    if case.expected_category is not None and prediction.category is None:
        return "Model output has no string 'category' but the case expects one"
    return None


def _output_from_record(record: PredictionRecord | None) -> _CaseOutput:
    """Turn one predictions-file line into a case output; a missing line is an error.

    ``raw`` keeps the record as JSON so reports show exactly what the system produced.
    """
    if record is None:
        return _CaseOutput(error=MISSING_PREDICTION_ERROR)
    recorded = _CaseOutput(
        raw=record.model_dump_json(exclude_defaults=True),
        latency_ms=record.latency_ms,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
    )
    if record.error is not None:
        return replace(recorded, error=f"Prediction recorded an error: {record.error}")
    prediction = Prediction(
        category=record.category, summary=record.summary or "", answer=record.answer
    )
    return replace(recorded, prediction=prediction)


def _error_result(case: GoldenCase, error: str, output: _CaseOutput) -> CaseResult:
    """A failed result that keeps the raw output and token use, so reports show the cause."""
    return CaseResult(
        case_id=case.id,
        topic=case.topic,
        predicted_category="",
        predicted_summary="",
        judge=fallback_score(case, None, error),
        latency_ms=output.latency_ms,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        raw_output=output.raw,
        error=error,
    )


def _make_run_id(name: str, version: str) -> str:
    """Readable and unique: name and version for humans, 8 hex chars against collisions."""
    return f"{name}-{version}-{uuid.uuid4().hex[:8]}"
