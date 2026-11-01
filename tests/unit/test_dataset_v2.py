"""Dataset schema v2: free-text cases, the versioned file wrapper and v1 compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llm_regression_detector.eval.dataset import (
    GoldenCase,
    GoldenDataset,
    OutputFormat,
    PromptSpec,
    load_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "golden-dataset.v2.schema.json"


def _write(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v1_bare_list_still_loads() -> None:
    cases = load_dataset(REPO_ROOT / "golden_dataset" / "incidents.json")
    assert len(cases) >= 50
    assert all(c.expected_category is not None for c in cases)
    assert all(c.rubric is None and c.reference_answer is None for c in cases)


def test_v2_sample_loads_with_free_text_fields() -> None:
    cases = load_dataset(REPO_ROOT / "golden_dataset" / "store_support_answers.json")
    by_id = {c.id: c for c in cases}
    assert by_id["ret001"].rubric is not None
    assert by_id["prd001"].expected_category is None
    assert by_id["prd001"].reference_answer is not None
    assert by_id["pii001"].source == {"origin": "trace-to-golden", "trace": "example-0001"}


def test_v2_wrapper_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "schema_version": 2,
            "cases": [{"id": "a", "input_text": "q", "rubric": ["is polite"]}],
        },
    )
    [case] = load_dataset(path)
    assert case.rubric == ["is polite"]


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"schema_version": 3, "cases": []})
    with pytest.raises(ValueError, match="schema_version 3"):
        load_dataset(path)


def test_non_list_non_object_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "just a string")
    with pytest.raises(ValueError, match="JSON array"):
        load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    case = {"id": "dup", "input_text": "q", "expected_category": "p0"}
    path = _write(tmp_path, [case, case])
    with pytest.raises(ValueError, match="dup"):
        load_dataset(path)


def test_case_without_anything_to_judge_is_rejected() -> None:
    with pytest.raises(ValidationError, match="nothing to judge"):
        GoldenCase(id="x", input_text="q")


def test_empty_rubric_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenCase(id="x", input_text="q", rubric=[])


def test_blank_rubric_criterion_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenCase(id="x", input_text="q", rubric=[""])


def test_unknown_case_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(
            {"id": "x", "input_text": "q", "expected_category": "p0", "rubrik": ["typo"]}
        )


def test_wrapper_needs_at_least_one_case() -> None:
    with pytest.raises(ValidationError):
        GoldenDataset(schema_version=2, cases=[])


def test_committed_json_schema_matches_model() -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == GoldenDataset.model_json_schema(), (
        "schemas/golden-dataset.v2.schema.json is stale. Regenerate it with: "
        "uv run lrd schema --output schemas/golden-dataset.v2.schema.json"
    )


def _prompt(user_template: str, **extra: Any) -> PromptSpec:
    return PromptSpec.model_validate(
        {"name": "p", "version": "v1", "system": "s", "user_template": user_template, **extra}
    )


def test_prompt_renders_input_text_placeholder() -> None:
    prompt = _prompt("Q: {input_text}")
    assert prompt.render_messages("hello")[-1]["content"] == "Q: hello"
    assert prompt.output_format is OutputFormat.JSON


def test_prompt_email_placeholder_still_renders_with_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="input_text"):
        prompt = _prompt("Q: {email}")
    assert prompt.render_messages("hello")[-1]["content"] == "Q: hello"


def test_prompt_accepts_text_output_format() -> None:
    prompt = _prompt("{input_text}", output_format="text")
    assert prompt.output_format is OutputFormat.TEXT
