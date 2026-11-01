"""Prediction parsing per output format, and the per-case pass rule across scoring modes."""

from __future__ import annotations

from llm_regression_detector.eval.dataset import (
    CaseResult,
    CriterionVerdict,
    JudgeScore,
    OutputFormat,
)
from llm_regression_detector.eval.parsing import Prediction, parse_prediction


def test_json_output_reads_category_summary_and_answer() -> None:
    raw = '{"category": "ANSWER", "summary": "s", "answer": "Yes, within 30 days."}'
    assert parse_prediction(raw, OutputFormat.JSON) == Prediction(
        category="ANSWER", summary="s", answer="Yes, within 30 days."
    )


def test_json_output_ignores_non_string_fields() -> None:
    assert parse_prediction('{"category": 3, "summary": null}', OutputFormat.JSON) == Prediction()


def test_json_output_that_is_not_json_is_none() -> None:
    assert parse_prediction("plain prose", OutputFormat.JSON) is None


def test_text_output_is_the_whole_answer() -> None:
    prediction = parse_prediction("  Yes, you can return it.\n", OutputFormat.TEXT)
    assert prediction == Prediction(answer="Yes, you can return it.")


def test_judged_text_prefers_answer_over_summary() -> None:
    assert Prediction(summary="s", answer="a").judged_text == "a"
    assert Prediction(summary="s").judged_text == "s"


def _result(judge: JudgeScore) -> CaseResult:
    return CaseResult(
        case_id="c",
        predicted_category="",
        predicted_summary="",
        judge=judge,
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        raw_output="",
    )


def _criterion(passed: bool) -> CriterionVerdict:
    return CriterionVerdict(criterion="c", passed=passed)


def test_passes_without_category_when_all_criteria_pass() -> None:
    judge = JudgeScore(summary_score=4, criteria=[_criterion(True), _criterion(True)])
    assert _result(judge).passed


def test_fails_when_any_criterion_fails() -> None:
    judge = JudgeScore(summary_score=5, criteria=[_criterion(True), _criterion(False)])
    assert not _result(judge).passed
    assert judge.rubric_pass_rate == 0.5


def test_fails_when_category_mismatches_even_if_rubric_passes() -> None:
    judge = JudgeScore(category_match=False, summary_score=5, criteria=[_criterion(True)])
    assert not _result(judge).passed


def test_fails_on_low_quality_score() -> None:
    assert not _result(JudgeScore(category_match=True, summary_score=2)).passed


def test_rubric_pass_rate_is_none_without_rubric() -> None:
    assert JudgeScore(category_match=True, summary_score=4).rubric_pass_rate is None
