"""Offline scoring: the predictions file contract and ``Runner.score``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_regression_detector.eval.dataset import GoldenCase, PredictionRecord, load_predictions
from llm_regression_detector.eval.runner import Runner
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.llm.mock import MockLLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "predictions.v1.schema.json"


def _write_jsonl(tmp_path: Path, lines: list[Any]) -> Path:
    path = tmp_path / "suite.predictions.jsonl"
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_predictions_reads_lines_and_skips_blanks(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {"id": "a", "category": "ANSWER", "answer": "Yes."},
            "",
            {"id": "b", "answer": "No.", "latency_ms": 120.5, "prompt_tokens": 40},
        ],
    )
    records = load_predictions(path)
    assert [r.id for r in records] == ["a", "b"]
    assert records[1].latency_ms == 120.5
    assert records[1].prompt_tokens == 40
    assert records[1].category is None


def test_load_predictions_rejects_duplicate_ids_with_line_number(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [{"id": "a", "answer": "x"}, {"id": "a", "answer": "y"}])
    with pytest.raises(ValueError, match=r"line 2.*duplicate id 'a'"):
        load_predictions(path)


def test_load_predictions_reports_bad_json_line(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [{"id": "a", "answer": "x"}, "{not json"])
    with pytest.raises(ValueError, match="line 2"):
        load_predictions(path)


def test_load_predictions_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [{"id": "a", "answr": "typo"}])
    with pytest.raises(ValueError, match="line 1"):
        load_predictions(path)


def test_committed_predictions_schema_matches_model() -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == PredictionRecord.model_json_schema(), (
        "schemas/predictions.v1.schema.json is stale. Regenerate it with: "
        "uv run lrd schema --kind predictions --output schemas/predictions.v1.schema.json"
    )


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase(id="cat", input_text="q1", expected_category="ANSWER"),
        GoldenCase(id="rub", input_text="q2", rubric=["Is polite", "Is short"]),
        GoldenCase(id="missing", input_text="q3", reference_answer="r"),
    ]


def _runner() -> Runner:
    client = MockLLMClient()
    return Runner(client=client, judge=Judge(client), concurrency=2)


async def test_score_grades_existing_answers_without_calling_a_prompt() -> None:
    predictions = [
        PredictionRecord(id="cat", category="ANSWER", answer="Here you go.", latency_ms=80.0),
        PredictionRecord(id="rub", answer="Thanks, done.", prompt_tokens=12, completion_tokens=5),
    ]
    run = await _runner().score(
        name="coordinator", version="abc123", cases=_cases(), predictions=predictions
    )
    by_id = {r.case_id: r for r in run.results}

    assert run.prompt_name == "coordinator"
    assert run.prompt_version == "abc123"
    assert by_id["cat"].passed
    assert by_id["cat"].latency_ms == 80.0
    assert by_id["rub"].passed
    assert len(by_id["rub"].judge.criteria) == 2
    assert by_id["rub"].predicted_answer == "Thanks, done."
    assert run.summary.total_prompt_tokens == 12


async def test_case_without_prediction_fails_as_error() -> None:
    run = await _runner().score(
        name="s", version="v", cases=_cases(), predictions=[PredictionRecord(id="cat")]
    )
    missing = next(r for r in run.results if r.case_id == "missing")
    assert not missing.passed
    assert missing.error == "No prediction for this case id"


async def test_prediction_with_recorded_error_fails() -> None:
    predictions = [PredictionRecord(id="cat", category="ANSWER", error="agent timed out")]
    run = await _runner().score(name="s", version="v", cases=_cases()[:1], predictions=predictions)
    [result] = run.results
    assert not result.passed
    assert result.error == "Prediction recorded an error: agent timed out"


async def test_prediction_for_unknown_case_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="ghost"):
        await _runner().score(
            name="s",
            version="v",
            cases=_cases(),
            predictions=[PredictionRecord(id="ghost", answer="x")],
        )
