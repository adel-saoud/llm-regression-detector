"""Typer CLI: entry point for ``lrd``."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from enum import StrEnum
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import structlog
import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from llm_regression_detector.ci import affected_prompts
from llm_regression_detector.config import Settings
from llm_regression_detector.diff.analyzer import Analyzer, DiffReport, to_alert_payload
from llm_regression_detector.diff.drift import DriftReport, analyze_drift
from llm_regression_detector.eval.dataset import (
    EvalRun,
    GoldenDataset,
    PredictionRecord,
    load_dataset,
    load_predictions,
    load_prompt,
    prompt_dataset_path,
)
from llm_regression_detector.eval.runner import Runner
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.eval.stats import MIN_DRIFT_HISTORY
from llm_regression_detector.llm.client import build_client, build_judge_client
from llm_regression_detector.notify import AlertSeverity, build_notifier
from llm_regression_detector.report.html import write_html
from llm_regression_detector.report.pr_comment import render_pr_comment
from llm_regression_detector.storage.sqlite import SQLiteStorage

app = typer.Typer(
    name="lrd",
    help="LLM Regression Detector: catch quality drops in LLM systems before they ship.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_DRIFT_WINDOW_RUNS = 8
"""Past runs the slow-drift band is computed over: above ``MIN_DRIFT_HISTORY``, still recent."""


def _configure_logging(settings: Settings) -> None:
    """Human-readable structlog output on stderr, coloured only on a terminal (not in CI)."""
    level = getattr(logging, settings.log_level.value)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


# ── `lrd run` ─────────────────────────────────────────────────────────────


@app.command()
def run(
    prompt: Annotated[Path, typer.Option("--prompt", "-p", help="Path to prompt YAML.")],
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to golden dataset JSON. Default: the prompt's `dataset` key.",
        ),
    ] = None,
    db: Annotated[Path, typer.Option(help="SQLite path for run history.")] = Path("evals/runs.db"),
    save: Annotated[bool, typer.Option(help="Persist this run to the database.")] = True,
    report: Annotated[
        Path | None,
        typer.Option(help="If set, write an HTML report to this path."),
    ] = None,
    diff_baseline: Annotated[
        bool,
        typer.Option("--diff/--no-diff", help="Compare to the latest stored baseline."),
    ] = True,
    drift: Annotated[
        bool,
        typer.Option(help="Also check slow drift over the last 8 stored runs."),
    ] = True,
    notify: Annotated[
        bool,
        typer.Option(help="Send webhook alert if a webhook URL is configured."),
    ] = True,
    consensus_n: Annotated[
        int | None,
        typer.Option(
            "--consensus-n",
            help="Override LRD_JUDGE_CONSENSUS_N for this run (1=single call, 3=majority vote).",
            min=1,
            max=7,
        ),
    ] = None,
    fail_on_critical: Annotated[
        bool,
        typer.Option(help="Exit non-zero on CRITICAL regressions (CI gating)."),
    ] = True,
) -> None:
    """Run a prompt against its golden dataset and (optionally) diff vs baseline."""
    asyncio.run(
        _run_async(
            prompt_path=prompt,
            dataset_path=dataset,
            db_path=db,
            save=save,
            report_path=report,
            diff_baseline=diff_baseline,
            drift_check=drift,
            notify=notify,
            consensus_n_override=consensus_n,
            fail_on_critical=fail_on_critical,
        )
    )


async def _run_async(
    *,
    prompt_path: Path,
    dataset_path: Path | None,
    db_path: Path,
    save: bool,
    report_path: Path | None,
    diff_baseline: bool,
    drift_check: bool,
    notify: bool,
    consensus_n_override: int | None,
    fail_on_critical: bool,
) -> None:
    settings = _load_settings()
    _configure_logging(settings)
    if settings.is_mock_mode:
        console.print("[yellow]No LLM credentials detected: running in MOCK mode.[/yellow]")

    prompt_spec = load_prompt(prompt_path)
    dataset_path = dataset_path or prompt_dataset_path(prompt_path, prompt_spec)
    if dataset_path is None:
        console.print(
            f"[red]No dataset for {prompt_path}: pass --dataset or add a `dataset` key "
            "to the prompt YAML.[/red]"
        )
        raise typer.Exit(code=1)
    cases = load_dataset(dataset_path)
    consensus_n = consensus_n_override or settings.judge_consensus_n
    console.print(
        f"[bold]{prompt_spec.name}@{prompt_spec.version}[/bold]: "
        f"{len(cases)} cases · judge consensus_n={consensus_n}"
    )

    runner = _build_runner(settings, consensus_n)
    candidate = await runner.run(prompt=prompt_spec, cases=cases)
    await _report_and_gate(
        settings,
        candidate,
        db_path=db_path,
        save=save,
        report_path=report_path,
        diff_baseline=diff_baseline,
        drift_check=drift_check,
        notify=notify,
        fail_on_critical=fail_on_critical,
    )


def _build_runner(settings: Settings, consensus_n: int) -> Runner:
    """Runner wired from settings, shared by ``lrd run`` and ``lrd score`` so both
    judge with the same judge model, concurrency and cost rates."""
    return Runner(
        client=build_client(settings),
        judge=Judge(build_judge_client(settings), consensus_n=consensus_n),
        concurrency=settings.concurrency,
        cost_per_million_input_usd=settings.cost_input_usd,
        cost_per_million_output_usd=settings.cost_output_usd,
    )


async def _report_and_gate(
    settings: Settings,
    candidate: EvalRun,
    *,
    db_path: Path,
    save: bool,
    report_path: Path | None,
    diff_baseline: bool,
    drift_check: bool,
    notify: bool,
    fail_on_critical: bool,
) -> None:
    """Shared tail of ``lrd run`` and ``lrd score``: diff, drift, persist, report, alert, gate.

    The baseline is looked up by ``candidate.prompt_name``, so prompt runs and offline
    runs each diff against earlier runs with the same name.
    """
    _print_summary(candidate)

    storage = SQLiteStorage(db_path)
    await storage.initialize()

    diff = None
    baseline = None
    if diff_baseline:
        baseline = await storage.latest_baseline(candidate.prompt_name)
        if baseline is not None and baseline.run_id != candidate.run_id:
            diff = Analyzer().diff(baseline, candidate)
            _print_diff(diff)

    drift_report: DriftReport | None = None
    if drift_check:
        history = await storage.recent(candidate.prompt_name, limit=_DRIFT_WINDOW_RUNS)
        drift_report = analyze_drift(history, candidate)
        _print_drift(drift_report)

    if save:
        await storage.save(candidate)
        console.print(f"[green]✓[/green] saved run [bold]{candidate.run_id}[/bold]")

    if report_path is not None:
        write_html(report_path, candidate=candidate, diff=diff, baseline=baseline)
        console.print(f"[green]✓[/green] wrote report → {report_path}")

    if notify and diff is not None:
        notifier = build_notifier(settings)
        await notifier.send(to_alert_payload(diff))

    if fail_on_critical and diff is not None and diff.severity is AlertSeverity.CRITICAL:
        console.print("[bold red]CRITICAL regression: exiting non-zero.[/bold red]")
        raise typer.Exit(code=2)


# ── `lrd score` ───────────────────────────────────────────────────────────


@app.command()
def score(
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions", help="Predictions JSONL: one {id, category?, answer?} per line."
        ),
    ],
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to golden dataset JSON."),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            help="Run name; keys the stored baseline. Default: dataset file name up to its "
            "first dot.",
        ),
    ] = None,
    version: Annotated[
        str,
        typer.Option(help="Label for this run, such as the SHA of the code under test."),
    ] = "offline",
    db: Annotated[Path, typer.Option(help="SQLite path for run history.")] = Path("evals/runs.db"),
    save: Annotated[bool, typer.Option(help="Persist this run to the database.")] = True,
    report: Annotated[
        Path | None,
        typer.Option(help="If set, write an HTML report to this path."),
    ] = None,
    diff_baseline: Annotated[
        bool,
        typer.Option("--diff/--no-diff", help="Compare to the latest stored baseline."),
    ] = True,
    drift: Annotated[
        bool,
        typer.Option(help="Also check slow drift over the last 8 stored runs."),
    ] = True,
    notify: Annotated[
        bool,
        typer.Option(help="Send webhook alert if a webhook URL is configured."),
    ] = True,
    consensus_n: Annotated[
        int | None,
        typer.Option(
            "--consensus-n",
            help="Override LRD_JUDGE_CONSENSUS_N for this run (1=single call, 3=majority vote).",
            min=1,
            max=7,
        ),
    ] = None,
    fail_on_critical: Annotated[
        bool,
        typer.Option(help="Exit non-zero on CRITICAL regressions (CI gating)."),
    ] = True,
) -> None:
    """Score answers a system already produced against the golden dataset (no prompt run).

    For agents and services that generate their own answers. Same judge, diff and
    exit codes as ``lrd run``.
    """
    asyncio.run(
        _score_async(
            predictions_path=predictions,
            dataset_path=dataset,
            name=name or dataset.name.split(".")[0],
            version=version,
            db_path=db,
            save=save,
            report_path=report,
            diff_baseline=diff_baseline,
            drift_check=drift,
            notify=notify,
            consensus_n_override=consensus_n,
            fail_on_critical=fail_on_critical,
        )
    )


async def _score_async(
    *,
    predictions_path: Path,
    dataset_path: Path,
    name: str,
    version: str,
    db_path: Path,
    save: bool,
    report_path: Path | None,
    diff_baseline: bool,
    drift_check: bool,
    notify: bool,
    consensus_n_override: int | None,
    fail_on_critical: bool,
) -> None:
    settings = _load_settings()
    _configure_logging(settings)
    if settings.is_mock_mode:
        console.print("[yellow]No LLM credentials detected: judging in MOCK mode.[/yellow]")

    cases = load_dataset(dataset_path)
    records = load_predictions(predictions_path)
    consensus_n = consensus_n_override or settings.judge_consensus_n
    console.print(
        f"[bold]{name}@{version}[/bold] (offline) · "
        f"{len(records)} predictions for {len(cases)} cases · judge consensus_n={consensus_n}"
    )

    runner = _build_runner(settings, consensus_n)
    candidate = await runner.score(name=name, version=version, cases=cases, predictions=records)
    await _report_and_gate(
        settings,
        candidate,
        db_path=db_path,
        save=save,
        report_path=report_path,
        diff_baseline=diff_baseline,
        drift_check=drift_check,
        notify=notify,
        fail_on_critical=fail_on_critical,
    )


# ── `lrd diff` ────────────────────────────────────────────────────────────


@app.command("diff")
def diff_cmd(
    baseline_id: Annotated[str, typer.Argument(help="Baseline run id.")],
    candidate_id: Annotated[str, typer.Argument(help="Candidate run id.")],
    db: Annotated[Path, typer.Option(help="SQLite path.")] = Path("evals/runs.db"),
) -> None:
    """Compare two stored runs by id."""
    asyncio.run(_diff_async(baseline_id, candidate_id, db))


async def _diff_async(baseline_id: str, candidate_id: str, db: Path) -> None:
    storage = SQLiteStorage(db)
    await storage.initialize()
    baseline = await storage.get(baseline_id)
    candidate = await storage.get(candidate_id)
    if baseline is None or candidate is None:
        console.print("[red]Run not found.[/red]")
        raise typer.Exit(code=1)
    diff = Analyzer().diff(baseline, candidate)
    _print_diff(diff)


# ── `lrd report` ──────────────────────────────────────────────────────────


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Candidate run id.")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("evals/report.html"),
    pr_comment: Annotated[
        bool, typer.Option(help="Print a markdown PR comment to stdout.")
    ] = False,
    db: Annotated[Path, typer.Option(help="SQLite path.")] = Path("evals/runs.db"),
) -> None:
    """Render an HTML (and optional PR-comment markdown) report for a stored run."""
    asyncio.run(_report_async(run_id, output, pr_comment, db))


async def _report_async(run_id: str, output: Path, pr_comment: bool, db: Path) -> None:
    storage = SQLiteStorage(db)
    await storage.initialize()
    candidate = await storage.get(run_id)
    if candidate is None:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(code=1)
    baseline = await storage.latest_baseline(candidate.prompt_name)
    diff = None
    if baseline is not None and baseline.run_id != candidate.run_id:
        diff = Analyzer().diff(baseline, candidate)
    write_html(output, candidate=candidate, diff=diff, baseline=baseline)
    console.print(f"[green]✓[/green] wrote report → {output}")
    if pr_comment:
        console.print("\n--- PR comment markdown ---\n")
        console.print(render_pr_comment(candidate, diff))


# ── `lrd pr-comment` ─────────────────────────────────────────────────────


@app.command("pr-comment")
def pr_comment_cmd(
    prompt_name: Annotated[
        str | None,
        typer.Option(
            "--prompt-name",
            "-n",
            help="Prompt name to look up. Defaults to the most recent run.",
        ),
    ] = None,
    db: Annotated[Path, typer.Option(help="SQLite path.")] = Path("evals/runs.db"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write markdown to this file instead of stdout."),
    ] = None,
) -> None:
    """Render a GitHub PR comment for the latest run (and its baseline).

    Reads the two most recent runs for the given prompt from the database and
    renders a diff summary as GitHub-flavoured markdown.  Designed to be called
    from CI after ``lrd run`` so no run-id bookkeeping is needed.
    """
    asyncio.run(_pr_comment_async(prompt_name=prompt_name, db_path=db, output_path=output))


async def _pr_comment_async(
    *,
    prompt_name: str | None,
    db_path: Path,
    output_path: Path | None,
) -> None:
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    if prompt_name is not None:
        runs = await storage.recent(prompt_name, limit=2)
    else:
        # No prompt name: grab the two most recent runs across all prompts.
        runs = await storage.recent_any(limit=2)

    if not runs:
        console.print("[yellow]No runs found in database.[/yellow]")
        raise typer.Exit(code=1)

    candidate = runs[0]
    baseline = runs[1] if len(runs) > 1 else None
    diff = Analyzer().diff(baseline, candidate) if baseline is not None else None
    markdown = render_pr_comment(candidate, diff)

    if output_path is not None:
        output_path.write_text(markdown)
        console.print(f"[green]✓[/green] wrote PR comment → {output_path}")
    else:
        print(markdown)


# ── `lrd init` ───────────────────────────────────────────────────────────


@app.command("init")
def init_cmd(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            prompt="Task name (slug, e.g. customer-support)",
            help="Slug used for file names and the prompt name.",
        ),
    ],
    categories: Annotated[
        str,
        typer.Option(
            "--categories",
            "-c",
            prompt="Output categories (comma-separated, e.g. billing,technical,account,general)",
            help="Comma-separated list of valid output categories.",
        ),
    ],
    description: Annotated[
        str,
        typer.Option(
            "--description",
            prompt="One-line description of what your LLM does (e.g. Classifies customer emails)",
            help="Used in the generated YAML description field.",
        ),
    ],
    prompts_dir: Annotated[
        Path, typer.Option("--prompts-dir", help="Directory for prompt YAMLs.")
    ] = Path("prompts"),
    dataset_dir: Annotated[
        Path, typer.Option("--dataset-dir", help="Directory for golden dataset JSON.")
    ] = Path("golden_dataset"),
) -> None:
    """Scaffold a prompt YAML and golden dataset template for a new LLM task.

    The prompt gets few-shots for the first three categories and the dataset one
    placeholder case for each of the first five: enough to show the shape.
    """
    category_names = [c.strip() for c in categories.split(",") if c.strip()]
    if not category_names:
        console.print("[red]No categories provided.[/red]")
        raise typer.Exit(code=1)

    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{name}_v1.yaml"
    categories_quoted = ", ".join(f'"{c}"' for c in category_names)
    few_shot_lines: list[str] = []
    for category in category_names[:3]:
        summary = "One-sentence summary of the input."
        few_shot_lines.append(
            f'  - input: "Example input for category {category}"\n'
            f'    output: \'{{"category": "{category}", "summary": "{summary}"}}\''
        )
    few_shots_yaml = "\n".join(few_shot_lines)
    dataset_path = dataset_dir / f"{name}.json"
    dataset_key = Path(os.path.relpath(dataset_path, prompts_dir)).as_posix()
    prompt_yaml = f"""name: {name}
version: v1
dataset: {dataset_key}
description: |
  {description}: v1 baseline.

system: |
  {description}.

  For each input, output a JSON object with exactly two fields:
    - "category": one of {categories_quoted}
    - "summary":  a single-sentence description of the input

  Be precise. Do not include any prose outside the JSON.

few_shots:
{few_shots_yaml}

user_template: |
  {{input_text}}

  Output JSON only.
"""
    prompt_path.write_text(prompt_yaml)
    console.print(f"[green]✓[/green] wrote prompt  → {prompt_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, str]] = []
    for case_number, category in enumerate(category_names[:5], start=1):
        cases.append(
            {
                "id": f"{category[0]}{case_number:03d}",
                "topic": f"Example {category} case {case_number}",
                "input_text": (
                    f"Replace this with a real example input for the '{category}' category."
                ),
                "expected_category": category,
            }
        )
    dataset_path.write_text(json.dumps(cases, indent=2) + "\n")
    console.print(f"[green]✓[/green] wrote dataset → {dataset_path}")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  1. Edit {dataset_path}: replace the placeholder cases with real examples")
    console.print("     (aim for 30–50 labelled cases; see docs/golden-dataset-guide.md)")
    console.print(f"  2. Edit {prompt_path}: refine the system prompt and few-shot examples")
    console.print("  3. Run your first baseline:")
    console.print(f"     [bold]uv run lrd run -p {prompt_path} --no-diff[/bold]")
    console.print("  4. Open the dashboard:")
    console.print("     [bold]uv run lrd dashboard[/bold]")


# ── `lrd affected-prompts` ───────────────────────────────────────────────


@app.command("affected-prompts")
def affected_prompts_cmd(
    changed: Annotated[
        list[Path],
        typer.Argument(help="Changed file paths, e.g. from `git diff --name-only`."),
    ],
    prompts_dir: Annotated[
        Path,
        typer.Option("--prompts-dir", help="Directory of prompt YAMLs (top level only)."),
    ] = Path("prompts"),
) -> None:
    """Print the prompts to re-evaluate for a change set, one path per line.

    A prompt is affected when its YAML changed or the dataset its `dataset` key
    points to changed. Meant for CI: loop over the output and `lrd run` each one.
    """
    for prompt_path in affected_prompts(changed, prompts_dir):
        typer.echo(prompt_path.as_posix())


# ── `lrd schema` ─────────────────────────────────────────────────────────


class SchemaKind(StrEnum):
    """File contract ``lrd schema`` prints: a whole v2 dataset file, or one predictions line."""

    DATASET = "dataset"
    PREDICTIONS = "predictions"


_SCHEMA_MODELS: dict[SchemaKind, type[BaseModel]] = {
    SchemaKind.DATASET: GoldenDataset,
    SchemaKind.PREDICTIONS: PredictionRecord,
}


@app.command()
def schema(
    kind: Annotated[
        SchemaKind,
        typer.Option(help="dataset: a v2 dataset file. predictions: one predictions JSONL line."),
    ] = SchemaKind.DATASET,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the JSON Schema here instead of stdout."),
    ] = None,
) -> None:
    """Print the JSON Schema of a v2 golden-dataset file or of one predictions line.

    Lets producers in other languages validate their exports without importing
    this package.
    """
    model = _SCHEMA_MODELS[kind]
    text = json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    console.print(f"[green]✓[/green] wrote JSON Schema → {output}")


# ── `lrd dashboard` ──────────────────────────────────────────────────────


@app.command()
def dashboard(
    db: Annotated[Path, typer.Option(help="SQLite path.")] = Path("evals/runs.db"),
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8501,
) -> None:
    """Launch the Streamlit dashboard.

    The dashboard ships inside the installed package: works whether you run
    from a source checkout or a ``pip install``ed copy.
    """
    import subprocess

    env = {**os.environ, "LRD_DB_PATH": str(db)}
    dashboard_resource = files("llm_regression_detector.dashboard") / "app.py"
    with as_file(dashboard_resource) as dashboard_path:
        subprocess.run(  # noqa: S603  # args are controlled
            ["streamlit", "run", str(dashboard_path), "--server.port", str(port)],  # noqa: S607
            env=env,
            check=False,
        )


# ── pretty printing ───────────────────────────────────────────────────────


def _print_summary(run: EvalRun) -> None:
    """Headline metrics, then the per-category table when the run has categories."""
    summary = run.summary
    table = Table(show_header=False, box=None, padding=(0, 2))
    ci_text = (
        f"[dim](95% CI {summary.accuracy_ci_low * 100:.1f}"
        f"–{summary.accuracy_ci_high * 100:.1f}%)[/dim]"
    )
    table.add_row("Accuracy", f"[bold]{summary.accuracy * 100:.1f}%[/bold]  {ci_text}")
    table.add_row("Passed", f"{summary.cases_passed}/{summary.cases_total}")
    table.add_row("Avg summary score", f"{summary.avg_summary_score:.2f} / 5")
    table.add_row(
        "Latency p50/p95/p99",
        f"{summary.latency_p50_ms:.0f} / {summary.latency_p95_ms:.0f} / "
        f"{summary.latency_p99_ms:.0f} ms",
    )
    table.add_row(
        "Tokens (in/out)",
        f"{summary.total_prompt_tokens} / {summary.total_completion_tokens}",
    )
    if summary.estimated_cost_usd > 0:
        table.add_row("Estimated cost", f"${summary.estimated_cost_usd:.4f}")
    console.print(table)
    if summary.per_category:
        category_table = Table(title="Per-category accuracy", title_style="dim")
        category_table.add_column("Category")
        category_table.add_column("Accuracy", justify="right")
        category_table.add_column("Passed", justify="right")
        for category in summary.per_category:
            category_table.add_row(
                category.category,
                f"{category.accuracy * 100:.1f}%",
                f"{category.cases_passed}/{category.cases_total}",
            )
        console.print(category_table)


def _load_settings() -> Settings:
    """Settings from the environment, or exit 1 with the validation message.

    Covers misspelled ``LRD_*`` variables and unknown ``.env`` keys, which a
    traceback would bury.
    """
    try:
        return Settings()
    except ValidationError as exc:
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_diff(diff: DiffReport) -> None:
    """One line: severity, accuracy delta, significance and flipped-case counts."""
    severity = diff.severity.value
    delta_pp = diff.accuracy_delta * 100
    colour = {"info": "green", "warning": "yellow", "critical": "red"}[severity]
    significance = "significant" if diff.is_significant else "[dim](within noise)[/dim]"
    console.print(
        f"\n[bold {colour}]{severity.upper()}[/bold {colour}] · "
        f"accuracy {delta_pp:+.2f} pp {significance} · "
        f"regressions={len(diff.regressions)} · "
        f"improvements={len(diff.improvements)}\n"
    )
    if diff.large_gain:
        console.print("[bold cyan]Large gain: check the judge or dataset.[/bold cyan]\n")


def _print_drift(drift: DriftReport) -> None:
    """Drift verdict, or nothing while history is too short for the check to run."""
    if drift.window_size < MIN_DRIFT_HISTORY:
        return
    if drift.has_drift:
        console.print(
            f"[bold yellow]⚠ Slow drift detected[/bold yellow]: latest "
            f"{drift.latest_accuracy * 100:.1f}% < threshold "
            f"{drift.threshold * 100:.1f}% (MA {drift.moving_average * 100:.1f}%, "
            f"σ {drift.std_dev * 100:.2f}%)"
        )
    else:
        console.print(
            f"[dim]No slow drift over {drift.window_size} runs · "
            f"MA {drift.moving_average * 100:.1f}% ± {drift.std_dev * 100:.2f}%[/dim]"
        )


if __name__ == "__main__":
    app()
