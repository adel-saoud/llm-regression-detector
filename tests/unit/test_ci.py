"""Prompt-to-dataset pairing and the CI helper that picks which prompts a change affects."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_regression_detector.ci import affected_prompts
from llm_regression_detector.eval.dataset import load_prompt, prompt_dataset_path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _prompt(path: Path, dataset: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"dataset: {dataset}\n" if dataset is not None else ""
    path.write_text(
        f"name: {path.stem}\nversion: v1\nsystem: s\nuser_template: '{{input_text}}'\n{key}",
        encoding="utf-8",
    )
    return path


def test_dataset_key_resolves_relative_to_the_prompt_file(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path / "prompts" / "a.yaml", "../data/a.json")
    resolved = prompt_dataset_path(prompt, load_prompt(prompt))
    assert resolved is not None
    assert resolved.resolve() == (tmp_path / "data" / "a.json").resolve()


def test_prompt_without_dataset_key_has_no_dataset(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path / "a.yaml", None)
    assert prompt_dataset_path(prompt, load_prompt(prompt)) is None


@pytest.mark.parametrize(
    ("prompt_file", "dataset_file"),
    [
        ("incident_triage_v1.yaml", "incidents.json"),
        ("demo/incident_triage_v2_degraded.yaml", "incidents.json"),
        ("store_support_v1.yaml", "store_support_answers.json"),
    ],
)
def test_repo_prompts_point_to_their_dataset(prompt_file: str, dataset_file: str) -> None:
    prompt = REPO_ROOT / "prompts" / prompt_file
    resolved = prompt_dataset_path(prompt, load_prompt(prompt))
    assert resolved is not None
    assert resolved.resolve() == (REPO_ROOT / "golden_dataset" / dataset_file).resolve()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny repo: two prompts on one dataset, one prompt on another, one without a key."""
    _prompt(tmp_path / "prompts" / "a_v1.yaml", "../golden_dataset/shared.json")
    _prompt(tmp_path / "prompts" / "a_v2.yaml", "../golden_dataset/shared.json")
    _prompt(tmp_path / "prompts" / "b_v1.yaml", "../golden_dataset/b.json")
    _prompt(tmp_path / "prompts" / "c_v1.yml", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _affected(changed: list[str]) -> list[str]:
    return [p.as_posix() for p in affected_prompts([Path(c) for c in changed], Path("prompts"))]


def test_changed_prompt_is_affected(repo: Path) -> None:
    assert _affected(["prompts/b_v1.yaml"]) == ["prompts/b_v1.yaml"]


def test_changed_dataset_affects_every_prompt_that_uses_it(repo: Path) -> None:
    assert _affected(["golden_dataset/shared.json"]) == ["prompts/a_v1.yaml", "prompts/a_v2.yaml"]


def test_yml_extension_and_prompt_without_dataset_key(repo: Path) -> None:
    assert _affected(["prompts/c_v1.yml"]) == ["prompts/c_v1.yml"]


def test_unrelated_and_deleted_files_affect_nothing(repo: Path) -> None:
    assert _affected(["README.md", "prompts/deleted_v9.yaml"]) == []


def test_each_prompt_is_listed_once(repo: Path) -> None:
    changed = ["prompts/a_v1.yaml", "golden_dataset/shared.json"]
    assert _affected(changed) == ["prompts/a_v1.yaml", "prompts/a_v2.yaml"]


def test_repo_demo_prompt_is_never_picked_by_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """The degraded demo prompt is built to regress; a dataset PR must not run it."""
    monkeypatch.chdir(REPO_ROOT)
    picked = affected_prompts([Path("golden_dataset/incidents.json")], Path("prompts"))
    assert [p.as_posix() for p in picked] == ["prompts/incident_triage_v1.yaml"]
