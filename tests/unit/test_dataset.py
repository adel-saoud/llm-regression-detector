from __future__ import annotations

from llm_regression_detector.eval.dataset import (
    Category,
    GoldenCase,
    PromptSpec,
)


def test_golden_dataset_has_all_categories(golden_dataset: list[GoldenCase]) -> None:
    categories = {case.expected_category for case in golden_dataset}
    assert categories == set(Category)


def test_golden_dataset_min_size(golden_dataset: list[GoldenCase]) -> None:
    assert len(golden_dataset) >= 50, "Acceptance criterion: 50+ cases"


def test_golden_dataset_ids_unique(golden_dataset: list[GoldenCase]) -> None:
    ids = [c.id for c in golden_dataset]
    assert len(ids) == len(set(ids))


def test_prompt_renders_messages(prompt_v1: PromptSpec) -> None:
    messages = prompt_v1.render_messages("Test email body")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Test email body" in messages[-1]["content"]
    # Few-shots interleaved as user/assistant
    fs_count = len(prompt_v1.few_shots)
    if fs_count:
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert len(messages) == 1 + fs_count * 2 + 1
