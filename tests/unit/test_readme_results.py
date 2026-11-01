"""Every result the README quotes must match ``results/manifest.json``.

The manifest is written by ``scripts/reproduce.py``; this test fails when one is
regenerated without the other. The Results section quotes the median and range of
the repeated local Ollama runs; the ``Try it`` section quotes the mock, which is what
those commands run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
GUIDE = (ROOT / "docs" / "golden-dataset-guide.md").read_text(encoding="utf-8")
MANIFEST: dict[str, Any] = json.loads((ROOT / "results" / "manifest.json").read_text("utf-8"))
OLLAMA: dict[str, Any] = MANIFEST["runs"]["ollama"]
SUMMARY: dict[str, Any] = OLLAMA["summary"]
MOCK_SUMMARY: dict[str, Any] = MANIFEST["runs"]["mock"]["summary"]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


@pytest.mark.parametrize("label", ["baseline", "candidate"])
def test_results_table_row_matches_manifest(label: str) -> None:
    acc = SUMMARY[f"{label}.accuracy"]
    cats = [SUMMARY[f"{label}.accuracy.p{i}"]["median"] for i in range(4)]
    judge = SUMMARY[f"{label}.judge_score_mean"]["median"]
    row = (
        f"| {_pct(acc['median'])} ({_pct(acc['min'])} to {_pct(acc['max'])}) | "
        + " | ".join(_pct(c) for c in cats)
        + f" | {judge:.2f} |"
    )
    assert row in README


def test_verdict_matches_manifest() -> None:
    severities: dict[str, int] = SUMMARY["diff.severity"]
    n = SUMMARY["repeats"]
    delta = SUMMARY["diff.accuracy_delta_pp"]
    counts = ", ".join(f"{name.upper()} in {k} of {n}" for name, k in severities.items())
    assert f"Verdict per run: {counts}" in README
    assert f"accuracy delta {delta['min']:+.2f} to {delta['max']:+.2f} pp" in README


def test_try_it_output_matches_mock_run() -> None:
    delta = MOCK_SUMMARY["diff.accuracy_delta_pp"]["median"]
    (severity,) = MOCK_SUMMARY["diff.severity"]
    assert f"`{severity.upper()} · accuracy {delta:+.2f} pp significant`" in README


def test_models_named_in_readme_match_manifest() -> None:
    for role in ("under_test", "judge"):
        assert f"`{OLLAMA['model'][role]['name'].split('/', 1)[1]}`" in README
    assert f"over {SUMMARY['repeats']} runs" in README


def test_sensitivity_numbers_match_manifest() -> None:
    sens = MANIFEST["sensitivity"]
    min_drop = sens["min_drop_pp_by_cases"]
    low, high = sorted((min_drop["0.95"]["50"], min_drop["0.85"]["50"]))
    assert f"separate at a {low:.0f} to {high:.0f} point drop" in README
    assert f"drop under {low:.0f} points cannot be told apart from noise" in README
    cases = sens["min_cases_for_5pp_drop"]
    assert f"about {cases['0.95']} cases at a 95% baseline and {cases['0.85']} at 85%" in README
    for n, at85 in min_drop["0.85"].items():
        at95 = min_drop["0.95"][n]
        assert f"| {n} | {at85:g} pp | {at95:g} pp |" in GUIDE
