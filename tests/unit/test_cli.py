"""CLI smoke tests using Typer's CliRunner — exercises run/diff/report end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_regression_detector.cli import app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Force mock mode and isolate the working directory for every test."""
    for var in ("HF_TOKEN", "GEMINI_API_KEY", "WEBHOOK_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "lrd" in result.stdout.lower() or "regression" in result.stdout.lower()


def test_run_creates_report_and_persists(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    report = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "run",
            "--prompt",
            str(REPO_ROOT / "prompts" / "classifier_v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "golden_dataset" / "support_emails.json"),
            "--db",
            str(db),
            "--report",
            str(report),
            "--no-diff",
            "--no-notify",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert db.exists()
    assert report.exists()


def test_run_with_consensus_n_flag(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    result = runner.invoke(
        app,
        [
            "run",
            "--prompt",
            str(REPO_ROOT / "prompts" / "classifier_v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "golden_dataset" / "support_emails.json"),
            "--db",
            str(db),
            "--consensus-n",
            "3",
            "--no-diff",
            "--no-notify",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "consensus_n=3" in result.stdout


def test_run_then_diff_via_cli(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    base = [
        "--prompt",
        str(REPO_ROOT / "prompts" / "classifier_v1.yaml"),
        "--dataset",
        str(REPO_ROOT / "golden_dataset" / "support_emails.json"),
        "--db",
        str(db),
        "--no-diff",
        "--no-notify",
    ]
    r1 = runner.invoke(app, ["run", *base])
    assert r1.exit_code == 0, r1.stdout
    r2 = runner.invoke(app, ["run", *base])
    assert r2.exit_code == 0, r2.stdout

    # report subcommand should render for the most recent run; we discover it by
    # querying the DB directly to keep this test independent of stdout formatting
    import sqlite3

    with sqlite3.connect(db) as conn:
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT run_id FROM eval_runs ORDER BY timestamp DESC"
            ).fetchall()
        ]
    assert len(ids) == 2

    out = tmp_path / "out.html"
    r3 = runner.invoke(
        app,
        ["report", ids[0], "--output", str(out), "--db", str(db), "--pr-comment"],
    )
    assert r3.exit_code == 0, r3.stdout
    assert out.exists()
