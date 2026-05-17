"""SQLite-backed run history with explicit schema versioning.

Eval runs are persisted as JSON blobs alongside indexed columns. A
``schema_meta`` table tracks the on-disk schema version; ``initialize`` runs
forward-only migrations on attach. Old rows that pre-date a non-additive change
are loaded leniently — pydantic models in this project use ``extra="forbid"``,
so we add new fields with sensible defaults rather than evolving them in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import aiosqlite

from llm_regression_detector.eval.dataset import EvalRun

CURRENT_SCHEMA_VERSION = 1

_SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id          TEXT PRIMARY KEY,
    prompt_name     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    accuracy        REAL NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    payload_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_prompt
    ON eval_runs (prompt_name, timestamp DESC);
"""


class StorageError(Exception):
    """Raised on persistence failures (corrupt rows, schema mismatch, etc)."""


class SQLiteStorage:
    """Append-only repository over the ``eval_runs`` table.

    Each ``EvalRun`` is serialised to JSON with a ``schema_version`` tag so a
    future model evolution can detect old rows and apply field defaults at
    load time without losing history.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA_V1)
            await db.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            await db.commit()

    async def schema_version(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def save(self, run: EvalRun) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                    (run_id, prompt_name, prompt_version, timestamp, accuracy,
                     schema_version, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.prompt_name,
                    run.prompt_version,
                    run.timestamp.isoformat(),
                    run.summary.accuracy,
                    CURRENT_SCHEMA_VERSION,
                    run.model_dump_json(),
                ),
            )
            await db.commit()

    async def get(self, run_id: str) -> EvalRun | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT payload_json, schema_version FROM eval_runs WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _load_run(row[0], int(row[1]))

    async def latest_baseline(self, prompt_name: str) -> EvalRun | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT payload_json, schema_version FROM eval_runs
                WHERE prompt_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (prompt_name,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _load_run(row[0], int(row[1]))

    async def recent(self, prompt_name: str, limit: int = 7) -> list[EvalRun]:
        """Return the N most recent runs for a prompt — newest first."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT payload_json, schema_version FROM eval_runs
                WHERE prompt_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (prompt_name, limit),
            )
            rows = await cursor.fetchall()
        return [_load_run(r[0], int(r[1])) for r in rows]

    async def recent_any(self, limit: int = 2) -> list[EvalRun]:
        """Return the N most recent runs across all prompts — newest first."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT payload_json, schema_version FROM eval_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [_load_run(r[0], int(r[1])) for r in rows]


def _load_run(payload_json: str, row_schema_version: int) -> EvalRun:
    """Load an ``EvalRun`` row, applying forward-compat defaults for older schemas."""
    raw: Any = json.loads(payload_json)
    if not isinstance(raw, dict):
        raise StorageError("eval_runs row payload is not a JSON object")
    payload = cast(dict[str, Any], raw)
    if row_schema_version < CURRENT_SCHEMA_VERSION:
        _migrate_payload(payload, from_version=row_schema_version)
    return EvalRun.model_validate(payload)


def _migrate_payload(payload: dict[str, Any], *, from_version: int) -> None:
    """Mutate an old-schema payload into the current shape. Forward-only."""
    if from_version < 1:
        # Earlier schemas didn't have CI / percentiles / per-category — fill defaults.
        summary_raw = payload.get("summary")
        if not isinstance(summary_raw, dict):
            return
        summary = cast(dict[str, Any], summary_raw)
        accuracy = float(summary.get("accuracy", 0.0))
        legacy_latency = float(summary.get("avg_latency_ms", 0.0))
        summary.setdefault("accuracy_ci_low", accuracy)
        summary.setdefault("accuracy_ci_high", accuracy)
        summary.setdefault("latency_p50_ms", legacy_latency)
        summary.setdefault("latency_p95_ms", legacy_latency)
        summary.setdefault("latency_p99_ms", legacy_latency)
        summary.setdefault("estimated_cost_usd", 0.0)
        summary.setdefault("per_category", [])
        summary.pop("avg_latency_ms", None)
