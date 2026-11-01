"""LLM-as-Judge scorer with optional consensus voting.

Each case is scored on the checks its golden fields define:
  - ``category_match``  exact string match against ``expected_category``. Computed
                        in code, not by the judge: an equality check needs no model
                        and gains nothing but noise from one.
  - ``summary_score``   1 to 5 judge rating of the predicted summary or answer
                        against ``expected_summary_keywords`` and ``reference_answer``.
  - ``criteria``        one pass or fail judge verdict per ``rubric`` criterion, the
                        same shape as rubric-based metrics in Vertex AI evaluation.

Single-call judging is fast but noisy: the same model can return different scores
for the same input. ``consensus_n > 1`` runs the judge ``N`` times and takes the
**median** ``summary_score`` and a **majority vote** per rubric criterion. This
dampens variance at linear cost.
"""

from __future__ import annotations

import asyncio
from statistics import median_low

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_regression_detector.eval.dataset import CriterionVerdict, GoldenCase, JudgeScore
from llm_regression_detector.eval.parsing import Prediction, extract_json_object
from llm_regression_detector.llm.client import LLMClient

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial grader for an LLM benchmark.

You will be given the input sent to the model, the gold references available for
this case (expected summary keywords, a reference answer, rubric criteria), and the
model's predicted answer.

Output a JSON object with exactly these fields:
  "summary_score": integer 1 to 5, how well the predicted answer matches the references
                   1 = unrelated or wrong
                   2 = vaguely related
                   3 = partially correct (some keywords or intent captured)
                   4 = mostly correct (most keywords or the full intent)
                   5 = perfect (all keywords and the correct intent)
  "rationale":     one short sentence explaining the score
  "criteria":      one object per rubric criterion, in the order given:
                   {"passed": true or false, "rationale": "one short sentence"}
                   Use an empty list when there are no rubric criteria.

Output JSON only. No prose.
"""

_JUDGE_BASE_MAX_TOKENS = 200
"""Judge ``max_tokens`` for the score and rationale alone; the v1 judge budget."""

_JUDGE_MAX_TOKENS_PER_CRITERION = 60
"""Extra ``max_tokens`` per rubric criterion: one ``{"passed", "rationale"}`` object
with a one-sentence rationale, so long rubrics are not cut off mid-JSON."""

_OMITTED_CRITERION_RATIONALE = "Judge omitted this criterion; counted as failed."
"""Rationale for a criterion the judge left out, so a short reply cannot pass a case."""


class _RawCriterion(BaseModel):
    """One verdict as the judge returns it; matched to its criterion by position."""

    model_config = ConfigDict(extra="ignore")
    passed: bool
    rationale: str = ""


class _RawJudgeOutput(BaseModel):
    """What the judge model returns. Extra keys are ignored so a judge that still
    emits v1 fields such as ``category_match`` does not fail the case."""

    model_config = ConfigDict(extra="ignore")
    summary_score: int = Field(ge=1, le=5)
    rationale: str = ""
    criteria: list[_RawCriterion] = Field(default_factory=list[_RawCriterion])

    @field_validator("criteria", mode="before")
    @classmethod
    def _null_criteria_is_empty(cls, value: object) -> object:
        """Small judges write ``"criteria": null`` for a case without a rubric."""
        return [] if value is None else value


class ScorerError(Exception):
    """Raised when the judge response cannot be parsed into a ``JudgeScore``."""


def category_match(case: GoldenCase, predicted_category: str | None) -> bool | None:
    """Exact-match verdict, or ``None`` when the case defines no category."""
    if case.expected_category is None:
        return None
    return predicted_category == case.expected_category


def fallback_score(case: GoldenCase, predicted_category: str | None, reason: str) -> JudgeScore:
    """Lowest score for a case the judge could not grade; every criterion fails."""
    return JudgeScore(
        category_match=category_match(case, predicted_category),
        summary_score=1,
        rationale=reason,
        criteria=[
            CriterionVerdict(criterion=c, passed=False, rationale=reason) for c in case.rubric or []
        ],
    )


class Judge:
    """Wraps an ``LLMClient`` to score one prediction at a time.

    ``consensus_n`` controls how many judge calls are made per case.
    """

    def __init__(self, client: LLMClient, *, consensus_n: int = 1) -> None:
        if consensus_n < 1:
            raise ValueError(f"consensus_n must be >= 1, got {consensus_n}")
        self._client = client
        self._consensus_n = consensus_n

    async def score(self, case: GoldenCase, prediction: Prediction) -> JudgeScore:
        """Score one prediction with ``consensus_n`` parallel judge calls."""
        outputs = await asyncio.gather(
            *(self._single_call(case, prediction) for _ in range(self._consensus_n))
        )
        return _aggregate(case, prediction, list(outputs))

    async def _single_call(self, case: GoldenCase, prediction: Prediction) -> JudgeScore:
        """One judge call at temperature 0, with room for one verdict per criterion."""
        criteria_count = len(case.rubric or [])
        response = await self._client.complete(
            _build_messages(case, prediction),
            temperature=0.0,
            max_tokens=_JUDGE_BASE_MAX_TOKENS + _JUDGE_MAX_TOKENS_PER_CRITERION * criteria_count,
        )
        return _parse_judge_response(response.content, case, prediction)


def _aggregate(case: GoldenCase, prediction: Prediction, scores: list[JudgeScore]) -> JudgeScore:
    """Combine N judge scores: median quality score, strict majority per criterion.

    ``median_low`` keeps the score an integer on even N; a tie on a criterion fails it.
    """
    if len(scores) == 1:
        return scores[0]
    n = len(scores)
    votes = [sum(s.criteria[i].passed for s in scores) for i in range(len(case.rubric or []))]
    criteria = [
        CriterionVerdict(
            criterion=criterion,
            passed=votes[i] > n // 2,
            rationale=f"{votes[i]}/{n} judges passed it.",
        )
        for i, criterion in enumerate(case.rubric or [])
    ]
    summary_score = median_low([s.summary_score for s in scores])
    return JudgeScore(
        category_match=category_match(case, prediction.category),
        summary_score=summary_score,
        rationale=f"Consensus over {n} judges. Median summary score = {summary_score}.",
        criteria=criteria,
    )


def _build_messages(case: GoldenCase, prediction: Prediction) -> list[dict[str, str]]:
    """Judge input with only the references the case sets, so absent fields add no noise.

    Criteria are numbered because verdicts are matched back to them by position.
    """
    sections = [f"Input:\n{case.input_text}"]
    if case.expected_summary_keywords:
        sections.append(f"Expected summary keywords: {', '.join(case.expected_summary_keywords)}")
    if case.reference_answer is not None:
        sections.append(f"Reference answer:\n{case.reference_answer}")
    if case.rubric:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(case.rubric, start=1))
        sections.append(f"Rubric criteria ({len(case.rubric)}), judge each in order:\n{numbered}")
    sections.append(f"Predicted answer:\n{prediction.judged_text}")
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _parse_judge_response(raw: str, case: GoldenCase, prediction: Prediction) -> JudgeScore:
    """Parse judge output and align its verdicts with the case's rubric by position.

    Unparseable output falls back to the lowest score so one bad judge response
    doesn't crash the whole eval run. Criteria the judge left out count as failed.
    """
    payload = extract_json_object(raw)
    if payload is None:
        return fallback_score(
            case, prediction.category, "Judge output was not valid JSON; defaulted to score=1."
        )
    try:
        output = _RawJudgeOutput.model_validate(payload)
    except ValidationError as exc:
        raise ScorerError(f"Judge response failed validation: {exc}") from exc
    criteria = [
        _align_criterion(criterion, output.criteria[i] if i < len(output.criteria) else None)
        for i, criterion in enumerate(case.rubric or [])
    ]
    return JudgeScore(
        category_match=category_match(case, prediction.category),
        summary_score=output.summary_score,
        rationale=output.rationale,
        criteria=criteria,
    )


def _align_criterion(criterion: str, verdict: _RawCriterion | None) -> CriterionVerdict:
    """Attach the criterion text to the judge's verdict; a missing verdict fails."""
    if verdict is None:
        return CriterionVerdict(
            criterion=criterion, passed=False, rationale=_OMITTED_CRITERION_RATIONALE
        )
    return CriterionVerdict(criterion=criterion, passed=verdict.passed, rationale=verdict.rationale)
