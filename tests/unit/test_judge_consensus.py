"""Tests for the multi-call judge with majority vote / median aggregation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llm_regression_detector.eval.dataset import Category, Difficulty, GoldenCase, JudgeScore
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.llm.client import LLMResponse


class _ScriptedClient:
    """LLM stub that returns a pre-canned sequence of judge JSON responses."""

    def __init__(self, judge_responses: list[JudgeScore]) -> None:
        self._iter: Iterator[JudgeScore] = iter(judge_responses)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        score = next(self._iter)
        content = score.model_dump_json()
        return LLMResponse(
            content=content,
            model="scripted",
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
        )


def _case() -> GoldenCase:
    return GoldenCase(
        id="x",
        input_email="e",
        expected_category=Category.P2,
        expected_summary_keywords=["k"],
        difficulty=Difficulty.EASY,
    )


def test_consensus_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="consensus_n"):
        Judge(_ScriptedClient([]), consensus_n=0)


async def test_single_call_is_passthrough() -> None:
    client = _ScriptedClient([JudgeScore(category_match=True, summary_score=4, rationale="ok")])
    judge = Judge(client, consensus_n=1)
    score = await judge.score(_case(), "billing", "summary")
    assert score.category_match is True
    assert score.summary_score == 4


async def test_consensus_majority_vote_with_split_outcomes() -> None:
    # 2 of 3 say match=True → consensus=True; scores [5, 4, 1] → median_low = 4
    client = _ScriptedClient(
        [
            JudgeScore(category_match=True, summary_score=5, rationale=""),
            JudgeScore(category_match=True, summary_score=4, rationale=""),
            JudgeScore(category_match=False, summary_score=1, rationale=""),
        ]
    )
    judge = Judge(client, consensus_n=3)
    score = await judge.score(_case(), "billing", "summary")
    assert score.category_match is True
    assert score.summary_score == 4
    assert "Consensus over 3 judges" in score.rationale


async def test_consensus_majority_rejects_minority_match() -> None:
    # 1 of 3 says match=True → consensus=False
    client = _ScriptedClient(
        [
            JudgeScore(category_match=True, summary_score=5, rationale=""),
            JudgeScore(category_match=False, summary_score=2, rationale=""),
            JudgeScore(category_match=False, summary_score=1, rationale=""),
        ]
    )
    judge = Judge(client, consensus_n=3)
    score = await judge.score(_case(), "billing", "summary")
    assert score.category_match is False
