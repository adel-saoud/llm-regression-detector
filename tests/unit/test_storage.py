from __future__ import annotations

from pathlib import Path

from llm_regression_detector.storage.sqlite import SQLiteStorage
from tests.conftest import make_run


async def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "runs.db")
    await storage.initialize()
    run = make_run("r1", pass_ids=["a", "b"], fail_ids=["c"])
    await storage.save(run)

    loaded = await storage.get("r1")
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.summary.accuracy == run.summary.accuracy


async def test_latest_baseline_returns_most_recent(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "runs.db")
    await storage.initialize()
    await storage.save(make_run("r1", pass_ids=["a"], fail_ids=["b"]))
    await storage.save(make_run("r2", pass_ids=["a", "b"], fail_ids=[]))
    latest = await storage.latest_baseline("support-email-classifier")
    assert latest is not None
    assert latest.summary.accuracy == 1.0


async def test_recent_returns_in_descending_order(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "runs.db")
    await storage.initialize()
    for i in range(5):
        await storage.save(make_run(f"r{i}", pass_ids=["a"], fail_ids=[]))
    runs = await storage.recent("support-email-classifier", limit=3)
    assert len(runs) == 3
