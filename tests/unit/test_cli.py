"""CLI smoke tests using Typer's CliRunner: exercises run/diff/report end-to-end."""

from __future__ import annotations

import json
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
    for var in ("HF_TOKEN", "GEMINI_API_KEY", "LRD_WEBHOOK_URL"):
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
            str(REPO_ROOT / "prompts" / "incident_triage_v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "golden_dataset" / "incidents.json"),
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
            str(REPO_ROOT / "prompts" / "incident_triage_v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "golden_dataset" / "incidents.json"),
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
        str(REPO_ROOT / "prompts" / "incident_triage_v1.yaml"),
        "--dataset",
        str(REPO_ROOT / "golden_dataset" / "incidents.json"),
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


def test_schema_command_writes_dataset_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    result = runner.invoke(app, ["schema", "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    committed = (REPO_ROOT / "schemas" / "golden-dataset.v2.schema.json").read_text()
    assert out.read_text() == committed


def test_schema_command_prints_to_stdout() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert '"schema_version"' in result.stdout


def test_init_scaffolds_prompt_with_input_text_placeholder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["init", "--name", "demo", "--categories", "a,b", "--description", "Sorts tickets"],
    )
    assert result.exit_code == 0, result.stdout
    prompt_yaml = (tmp_path / "prompts" / "demo_v1.yaml").read_text()
    assert "{input_text}" in prompt_yaml
    assert "{email}" not in prompt_yaml
    assert "dataset: ../golden_dataset/demo.json" in prompt_yaml
    scaffolded_run = runner.invoke(
        app, ["run", "--prompt", "prompts/demo_v1.yaml", "--no-diff", "--no-notify"]
    )
    assert scaffolded_run.exit_code == 0, scaffolded_run.stdout


def _write_offline_suite(tmp_path: Path, *, correct: bool) -> tuple[Path, Path]:
    """40 classification cases plus predictions that are all right or all wrong."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = [f"c{i:02d}" for i in range(40)]
    dataset = tmp_path / "coordinator.dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cases": [
                    {"id": i, "input_text": f"question {i}", "expected_category": "ANSWER"}
                    for i in ids
                ],
            }
        )
    )
    predictions = tmp_path / "coordinator.predictions.jsonl"
    category = "ANSWER" if correct else "DECLINE"
    predictions.write_text(
        "\n".join(json.dumps({"id": i, "category": category, "answer": "Done."}) for i in ids)
    )
    return dataset, predictions


def test_score_grades_predictions_file_and_names_run_after_dataset(tmp_path: Path) -> None:
    dataset, predictions = _write_offline_suite(tmp_path, correct=True)
    db = tmp_path / "runs.db"
    report = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "score",
            "--dataset", str(dataset),
            "--predictions", str(predictions),
            "--version", "abc123",
            "--db", str(db),
            "--report", str(report),
            "--no-notify",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.stdout
    assert "coordinator@abc123" in result.stdout
    assert report.exists()


def test_score_exits_2_on_critical_regression(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    good_dataset, good_predictions = _write_offline_suite(tmp_path / "good", correct=True)
    bad_dataset, bad_predictions = _write_offline_suite(tmp_path / "bad", correct=False)
    common = ["--db", str(db), "--no-notify", "--name", "coordinator"]

    first = runner.invoke(
        app,
        ["score", "-d", str(good_dataset), "--predictions", str(good_predictions), *common],
    )
    assert first.exit_code == 0, first.stdout
    second = runner.invoke(
        app,
        ["score", "-d", str(bad_dataset), "--predictions", str(bad_predictions), *common],
    )
    assert second.exit_code == 2, second.stdout
    assert "CRITICAL" in second.stdout


def test_schema_command_writes_predictions_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "predictions.json"
    result = runner.invoke(app, ["schema", "--kind", "predictions", "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.read_text() == (REPO_ROOT / "schemas" / "predictions.v1.schema.json").read_text()


def test_run_reads_dataset_from_prompt_key(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--prompt", str(REPO_ROOT / "prompts" / "store_support_v1.yaml"),
            "--db", str(tmp_path / "runs.db"),
            "--no-diff",
            "--no-notify",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.stdout
    assert "8 cases" in result.stdout


def test_run_without_dataset_flag_or_key_fails_clearly(tmp_path: Path) -> None:
    prompt = tmp_path / "p.yaml"
    prompt.write_text("name: p\nversion: v1\nsystem: s\nuser_template: '{input_text}'\n")
    result = runner.invoke(app, ["run", "--prompt", str(prompt), "--no-diff", "--no-notify"])
    assert result.exit_code == 1
    assert "no dataset" in result.stdout.lower()


def test_affected_prompts_command_prints_one_path_per_line(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "x.yaml").write_text(
        "name: x\nversion: v1\nsystem: s\nuser_template: '{input_text}'\n"
        "dataset: ../golden_dataset/x.json\n"
    )
    result = runner.invoke(app, ["affected-prompts", "golden_dataset/x.json", "README.md"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout == "prompts/x.yaml\n"


def test_run_exits_1_on_misspelled_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LRD_CONCURENCY", "4")
    result = runner.invoke(app, ["run", "--prompt", "missing.yaml", "--no-diff", "--no-notify"])
    assert result.exit_code == 1
    assert "LRD_CONCURRENCY" in result.stdout
