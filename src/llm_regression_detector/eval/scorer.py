"""LLM-as-Judge scorer with optional consensus voting.

Asks a (cheap) judge model to grade two dimensions:
  - ``category_match``  — does the predicted category equal the expected one?
  - ``summary_score``   — 1–5 how well does the predicted summary cover the
                          expected keywords / capture the customer's intent?

Single-call judging is fast and free but noisy: the same model can return
different scores for the same input. ``consensus_n > 1`` runs the judge ``N``
times and takes the **majority vote** for ``category_match`` and the **median**
for ``summary_score``. This dampens variance at linear cost — the standard
pattern for production eval pipelines that need calibrated alerts.
"""

from __future__ import annotations

import asyncio
from statistics import median_low

from pydantic import ValidationError

from llm_regression_detector.eval.dataset import GoldenCase, JudgeScore
from llm_regression_detector.eval.parsing import extract_json_object
from llm_regression_detector.llm.client import LLMClient

_SYSTEM_PROMPT = """\
You are an impartial grader for an LLM classifier benchmark.

You will be given:
  - the original customer email
  - the expected category (gold label)
  - the expected summary keywords (gold label)
  - the model's predicted category
  - the model's predicted summary

Output a JSON object with exactly these fields:
  "category_match": true if predicted_category equals expected_category, else false
  "summary_score":  integer 1–5
                    1 = unrelated / wrong
                    2 = vaguely related
                    3 = partially correct (some keywords or intent captured)
                    4 = mostly correct (most keywords or full intent)
                    5 = perfect (all keywords + correct intent)
  "rationale":      one short sentence explaining the score

Output JSON only. No prose.
"""

_USER_TEMPLATE = """\
Email: {email}
Expected category: {expected_category}
Expected summary keywords: {expected_keywords}

Predicted category: {predicted_category}
Predicted summary:  {predicted_summary}
"""


class ScorerError(Exception):
    """Raised when the judge response cannot be parsed into a ``JudgeScore``."""


class Judge:
    """Wraps an ``LLMClient`` to score one prediction at a time.

    ``consensus_n`` controls how many judge calls are made per case; the final
    score is a majority vote on category match and a median on summary score.
    """

    def __init__(self, client: LLMClient, *, consensus_n: int = 1) -> None:
        if consensus_n < 1:
            raise ValueError(f"consensus_n must be >= 1, got {consensus_n}")
        self._client = client
        self._consensus_n = consensus_n

    async def score(
        self,
        case: GoldenCase,
        predicted_category: str,
        predicted_summary: str,
    ) -> JudgeScore:
        if self._consensus_n == 1:
            return await self._single_call(case, predicted_category, predicted_summary)
        return await self._consensus_call(case, predicted_category, predicted_summary)

    async def _single_call(
        self,
        case: GoldenCase,
        predicted_category: str,
        predicted_summary: str,
    ) -> JudgeScore:
        messages = _build_messages(case, predicted_category, predicted_summary)
        response = await self._client.complete(messages, temperature=0.0, max_tokens=200)
        return _parse_judge_response(response.content, predicted_category, case)

    async def _consensus_call(
        self,
        case: GoldenCase,
        predicted_category: str,
        predicted_summary: str,
    ) -> JudgeScore:
        # Run N independent judge calls in parallel; aggregate.
        scores = await asyncio.gather(
            *(
                self._single_call(case, predicted_category, predicted_summary)
                for _ in range(self._consensus_n)
            )
        )
        match_votes = sum(1 for s in scores if s.category_match)
        category_match = match_votes > self._consensus_n // 2
        summary_score = median_low([s.summary_score for s in scores])
        rationale = (
            f"Consensus over {self._consensus_n} judges: "
            f"{match_votes}/{self._consensus_n} agreed on category match. "
            f"Median summary score = {summary_score}."
        )
        return JudgeScore(
            category_match=category_match,
            summary_score=summary_score,
            rationale=rationale,
        )


def _build_messages(
    case: GoldenCase, predicted_category: str, predicted_summary: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                email=case.input_email,
                expected_category=case.expected_category,
                expected_keywords=", ".join(case.expected_summary_keywords) or "(none)",
                predicted_category=predicted_category,
                predicted_summary=predicted_summary,
            ),
        },
    ]


def _parse_judge_response(
    raw: str,
    predicted_category: str,
    case: GoldenCase,
) -> JudgeScore:
    """Parse judge output. Falls back to a deterministic match on parse failure
    so a single bad judge response doesn't crash the whole eval run."""
    payload = extract_json_object(raw)
    if payload is None:
        return JudgeScore(
            category_match=predicted_category == case.expected_category,
            summary_score=1,
            rationale="Judge output was not valid JSON; defaulted to score=1.",
        )
    try:
        return JudgeScore.model_validate(payload)
    except ValidationError as exc:
        raise ScorerError(f"Judge response failed validation: {exc}") from exc
