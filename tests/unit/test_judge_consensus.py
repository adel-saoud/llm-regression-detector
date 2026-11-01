"""Tests for the judge: deterministic category match, rubric verdicts and consensus voting."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from llm_regression_detector.eval.dataset import Difficulty, GoldenCase
from llm_regression_detector.eval.parsing import Prediction
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.llm.client import LLMResponse


class _ScriptedClient:
    """LLM stub that returns a pre-canned sequence of judge JSON payloads."""

    def __init__(self, judge_payloads: list[dict[str, Any] | str]) -> None:
        self._iter: Iterator[dict[str, Any] | str] = iter(judge_payloads)
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.prompts.append(messages[-1]["content"])
        payload = next(self._iter)
        return LLMResponse(
            content=payload if isinstance(payload, str) else json.dumps(payload),
            model="scripted",
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
        )


def _classification_case() -> GoldenCase:
    return GoldenCase(
        id="x",
        input_text="e",
        expected_category="p2",
        expected_summary_keywords=["k"],
        difficulty=Difficulty.EASY,
    )


def _rubric_case() -> GoldenCase:
    return GoldenCase(
        id="r",
        input_text="Can I return my kettle?",
        reference_answer="Yes, within 30 days.",
        rubric=["Allows the return", "Mentions 30 days"],
    )


def _verdict(score: int, *criteria: bool) -> dict[str, Any]:
    return {
        "summary_score": score,
        "rationale": "",
        "criteria": [{"passed": c, "rationale": ""} for c in criteria],
    }


def test_consensus_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="consensus_n"):
        Judge(_ScriptedClient([]), consensus_n=0)


async def test_category_match_is_deterministic_not_judged() -> None:
    client = _ScriptedClient([{"summary_score": 4, "category_match": True}])
    score = await Judge(client).score(
        _classification_case(), Prediction(category="billing", summary="s")
    )
    assert score.category_match is False  # judge's opinion on the category is ignored
    assert score.summary_score == 4
    assert score.criteria == []


async def test_category_match_true_on_exact_label() -> None:
    client = _ScriptedClient([{"summary_score": 4}])
    score = await Judge(client).score(
        _classification_case(), Prediction(category="p2", summary="s")
    )
    assert score.category_match is True


async def test_case_without_category_has_no_category_verdict() -> None:
    client = _ScriptedClient([_verdict(5, True, True)])
    score = await Judge(client).score(_rubric_case(), Prediction(answer="Yes, within 30 days."))
    assert score.category_match is None
    assert [c.criterion for c in score.criteria] == ["Allows the return", "Mentions 30 days"]
    assert all(c.passed for c in score.criteria)
    assert score.rubric_pass_rate == 1.0


async def test_judge_prompt_carries_reference_and_rubric() -> None:
    client = _ScriptedClient([_verdict(5, True, True)])
    await Judge(client).score(_rubric_case(), Prediction(answer="Sure."))
    [prompt] = client.prompts
    assert "Yes, within 30 days." in prompt
    assert "1. Allows the return" in prompt
    assert "2. Mentions 30 days" in prompt
    assert prompt.rstrip().endswith("Sure.")


async def test_criteria_missing_from_judge_output_count_as_failed() -> None:
    client = _ScriptedClient([_verdict(4, True)])  # judge answered 1 of 2 criteria
    score = await Judge(client).score(_rubric_case(), Prediction(answer="a"))
    assert [c.passed for c in score.criteria] == [True, False]
    assert "omitted" in score.criteria[1].rationale
    assert score.rubric_pass_rate == 0.5


async def test_null_criteria_reads_as_no_criteria() -> None:
    """qwen3.5:4b answers ``"criteria": null`` on classification cases."""
    client = _ScriptedClient([{"summary_score": 4, "criteria": None}])
    score = await Judge(client).score(
        _classification_case(), Prediction(category="p2", summary="s")
    )
    assert score.summary_score == 4
    assert score.criteria == []


async def test_extra_criteria_from_judge_are_ignored() -> None:
    client = _ScriptedClient([_verdict(4, True, True, False)])
    score = await Judge(client).score(_rubric_case(), Prediction(answer="a"))
    assert len(score.criteria) == 2


async def test_invalid_judge_json_falls_back_to_failing_score() -> None:
    client = _ScriptedClient(["not json at all"])
    score = await Judge(client).score(_rubric_case(), Prediction(answer="a"))
    assert score.summary_score == 1
    assert [c.passed for c in score.criteria] == [False, False]


async def test_consensus_median_score_and_per_criterion_majority() -> None:
    client = _ScriptedClient(
        [
            _verdict(5, True, False),
            _verdict(4, True, True),
            _verdict(1, False, False),
        ]
    )
    score = await Judge(client, consensus_n=3).score(_rubric_case(), Prediction(answer="a"))
    assert score.summary_score == 4  # median_low of [5, 4, 1]
    assert [c.passed for c in score.criteria] == [True, False]  # 2/3 and 1/3 votes
    assert "Consensus over 3 judges" in score.rationale


async def test_consensus_keeps_deterministic_category_match() -> None:
    client = _ScriptedClient([{"summary_score": 3}] * 3)
    score = await Judge(client, consensus_n=3).score(
        _classification_case(), Prediction(category="p2", summary="s")
    )
    assert score.category_match is True
    assert score.summary_score == 3
