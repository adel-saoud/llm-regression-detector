"""Tests that old schema-v0 rows are forward-migrated on load."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from llm_regression_detector.storage.sqlite import SQLiteStorage


async def test_legacy_row_is_loaded_with_default_fields(tmp_path: Path) -> None:
    """A row written by an older schema (no CI / percentiles / per-category) loads."""
    db_path = tmp_path / "runs.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    legacy_payload = {
        "run_id": "legacy-1",
        "prompt_name": "support-email-classifier",
        "prompt_version": "v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "results": [
            {
                "case_id": "a",
                "predicted_category": "billing",
                "predicted_summary": "x",
                "judge": {"category_match": True, "summary_score": 4, "rationale": ""},
                "latency_ms": 12.0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "raw_output": "{}",
                "error": None,
            }
        ],
        # ↓ summary as it would have looked in schema v0 — no CI, no percentiles
        "summary": {
            "accuracy": 1.0,
            "avg_summary_score": 4.0,
            "avg_latency_ms": 12.0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "cases_total": 1,
            "cases_passed": 1,
        },
    }

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO eval_runs
                (run_id, prompt_name, prompt_version, timestamp, accuracy,
                 schema_version, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-1",
                "support-email-classifier",
                "v1",
                legacy_payload["timestamp"],
                1.0,
                0,  # legacy schema version
                json.dumps(legacy_payload),
            ),
        )
        await db.commit()

    loaded = await storage.get("legacy-1")
    assert loaded is not None
    # New fields populated with sensible defaults
    assert loaded.summary.latency_p50_ms == 12.0
    assert loaded.summary.accuracy_ci_low == loaded.summary.accuracy
    assert loaded.summary.per_category == []
    assert loaded.summary.estimated_cost_usd == 0.0


async def test_schema_version_is_recorded(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "runs.db")
    await storage.initialize()
    version = await storage.schema_version()
    assert version >= 1
