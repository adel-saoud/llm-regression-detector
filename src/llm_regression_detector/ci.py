"""CI helpers: decide which prompts a change set needs to re-evaluate."""

from __future__ import annotations

from pathlib import Path

from llm_regression_detector.eval.dataset import load_prompt, prompt_dataset_path

PROMPT_SUFFIXES = (".yaml", ".yml")
"""File extensions read as prompt files; anything else in the prompts folder is ignored."""


def affected_prompts(changed_paths: list[Path], prompts_dir: Path) -> list[Path]:
    """Prompts in ``prompts_dir`` whose file changed or whose ``dataset`` changed.

    A dataset-only change re-runs exactly the prompts evaluated on it, instead of
    a fixed default prompt. Paths are compared after resolving, so ``changed_paths``
    can come straight from ``git diff --name-only`` run at the repository root.
    Changed files that no longer exist (deleted prompts) simply match nothing.
    Only the top level of ``prompts_dir`` is scanned, so prompts kept in subfolders
    are never selected.
    """
    changed = {p.resolve() for p in changed_paths}
    affected: list[Path] = []
    for prompt_path in sorted(prompts_dir.iterdir()):
        if prompt_path.suffix not in PROMPT_SUFFIXES:
            continue
        dataset = prompt_dataset_path(prompt_path, load_prompt(prompt_path))
        if prompt_path.resolve() in changed or (
            dataset is not None and dataset.resolve() in changed
        ):
            affected.append(prompt_path)
    return affected
