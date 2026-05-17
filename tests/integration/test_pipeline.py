"""End-to-end pipeline test using the deterministic mock LLM.

Exercises: dataset load → runner → judge → diff → alert payload → storage roundtrip.
No network, no API keys, no flakiness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_regression_detector.diff.analyzer import Analyzer, to_alert_payload
from llm_regression_detector.eval.dataset import load_dataset, load_prompt
from llm_regression_detector.eval.runner import Runner
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.llm.mock import MockLLMClient
from llm_regression_detector.storage.sqlite import SQLiteStorage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.integration
async def test_full_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    cases = load_dataset(REPO_ROOT / "golden_dataset" / "incidents.json")
    prompt = load_prompt(REPO_ROOT / "prompts" / "incident_triage_v1.yaml")

    client = MockLLMClient()
    runner = Runner(client=client, judge=Judge(client), concurrency=8)
    run = await runner.run(prompt=prompt, cases=cases[:20])  # subset for speed

    assert run.summary.cases_total == 20
    assert 0.0 <= run.summary.accuracy <= 1.0
    assert all(r.latency_ms >= 0 for r in run.results)

    storage = SQLiteStorage(tmp_path / "runs.db")
    await storage.initialize()
    await storage.save(run)
    loaded = await storage.get(run.run_id)
    assert loaded is not None
    assert loaded.summary.cases_passed == run.summary.cases_passed


@pytest.mark.integration
async def test_diff_detects_changes_between_two_runs() -> None:
    cases = load_dataset(REPO_ROOT / "golden_dataset" / "incidents.json")
    prompt = load_prompt(REPO_ROOT / "prompts" / "incident_triage_v1.yaml")

    client = MockLLMClient()
    runner = Runner(client=client, judge=Judge(client), concurrency=8)
    run_a = await runner.run(prompt=prompt, cases=cases[:15])
    run_b = await runner.run(prompt=prompt, cases=cases[:15])

    diff = Analyzer().diff(run_a, run_b)
    # Mock is deterministic — same input → same output → no regressions
    assert diff.regressions == []
    assert diff.improvements == []
    payload = to_alert_payload(diff)
    assert payload.run_id == run_b.run_id
