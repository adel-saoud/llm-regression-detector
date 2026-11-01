from __future__ import annotations

import pytest

from llm_regression_detector.eval.parsing import extract_json_object


@pytest.mark.parametrize(
    "raw",
    [
        '{"category_match": true, "summary_score": 4, "rationale": "ok"}',
        '```json\n{"category_match": true, "summary_score": 4, "rationale": "ok"}\n```',
        'Some prose\n{"category_match": true, "summary_score": 4, "rationale": "ok"}\nmore',
    ],
)
def test_extract_json_object_handles_common_shapes(raw: str) -> None:
    parsed = extract_json_object(raw)
    assert parsed is not None
    assert parsed["summary_score"] == 4


def test_extract_json_object_returns_none_for_garbage() -> None:
    assert extract_json_object("") is None
    assert extract_json_object("no json here") is None
    assert extract_json_object("[1, 2, 3]") is None  # array not object
