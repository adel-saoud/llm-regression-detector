"""Typer CLI — entry point for ``lrd``."""

from __future__ import annotations

import asyncio
import logging
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table

from llm_regression_detector.config import Settings
from llm_regression_detector.diff.analyzer import Analyzer, DiffReport, to_alert_payload
from llm_regression_detector.diff.drift import DriftReport, analyze_drift
from llm_regression_detector.eval.dataset import EvalRun, load_dataset, load_prompt
from llm_regression_detector.eval.runner import Runner
from llm_regression_detector.eval.scorer import Judge
from llm_regression_detector.llm.client import build_client
from llm_regression_detector.notify import build_notifier
from llm_regression_detector.report.html import write_html
from llm_regression_detector.report.pr_comment import render_pr_comment
from llm_regression_detector.storage.sqlite import SQLiteStorage

app = typer.Typer(
    name="lrd",
    help="LLM Regression Detector — catch quality drops in LLM systems before they ship.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _configure_logging(settings: Settings) -> None:
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
        Path,
        typer.Option("--dataset", "-d", help="Path to golden dataset JSON."),
    ] = Path("golden_dataset/support_emails.json"),
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
        typer.Option(help="Also compute slow-drift over the last N stored runs."),
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
    """Run a prompt against the golden dataset and (optionally) diff vs baseline."""
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
    dataset_path: Path,
    db_path: Path,
    save: bool,
    report_path: Path | None,
    diff_baseline: bool,
    drift_check: bool,
    notify: bool,
    consensus_n_override: int | None,
    fail_on_critical: bool,
) -> None:
    settings = Settings()
    _configure_logging(settings)
    if settings.is_mock_mode:
        console.print("[yellow]No LLM credentials detected — running in MOCK mode.[/yellow]")

    prompt_spec = load_prompt(prompt_path)
    cases = load_dataset(dataset_path)
    consensus_n = consensus_n_override or settings.judge_consensus_n
    console.print(
        f"[bold]{prompt_spec.name}@{prompt_spec.version}[/bold] — "
        f"{len(cases)} cases · judge consensus_n={consensus_n}"
    )

    client = build_client(settings)
    judge = Judge(client, consensus_n=consensus_n)
    runner = Runner(
        client=client,
        judge=judge,
        concurrency=settings.concurrency,
        cost_per_million_input_usd=settings.cost_per_million_input_usd,
        cost_per_million_output_usd=settings.cost_per_million_output_usd,
    )
    candidate = await runner.run(prompt=prompt_spec, cases=cases)

    _print_summary(candidate)

    storage = SQLiteStorage(db_path)
    await storage.initialize()

    diff = None
    baseline = None
    if diff_baseline:
        baseline = await storage.latest_baseline(prompt_spec.name)
        if baseline is not None and baseline.run_id != candidate.run_id:
            diff = Analyzer().diff(baseline, candidate)
            _print_diff(diff)

    drift_report: DriftReport | None = None
    if drift_check:
        history = await storage.recent(prompt_spec.name, limit=8)
        # Exclude the candidate itself if it's already saved (it isn't yet).
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

    if fail_on_critical and diff is not None and diff.severity.value == "critical":
        console.print("[bold red]CRITICAL regression — exiting non-zero.[/bold red]")
        raise typer.Exit(code=2)


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


# ── `lrd dashboard` ──────────────────────────────────────────────────────


@app.command()
def dashboard(
    db: Annotated[Path, typer.Option(help="SQLite path.")] = Path("evals/runs.db"),
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8501,
) -> None:
    """Launch the Streamlit dashboard.

    The dashboard ships inside the installed package — works whether you run
    from a source checkout or a ``pip install``ed copy.
    """
    import os
    import subprocess

    env = {**os.environ, "LRD_DB_PATH": str(db)}
    package_root = files("llm_regression_detector.dashboard") / "app.py"
    with as_file(package_root) as dash_path:
        subprocess.run(  # noqa: S603 — args are controlled
            ["streamlit", "run", str(dash_path), "--server.port", str(port)],  # noqa: S607
            env=env,
            check=False,
        )


# ── pretty printing ───────────────────────────────────────────────────────


def _print_summary(run: EvalRun) -> None:
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
        cat_table = Table(title="Per-category accuracy", title_style="dim")
        cat_table.add_column("Category")
        cat_table.add_column("Accuracy", justify="right")
        cat_table.add_column("Passed", justify="right")
        for cat in summary.per_category:
            cat_table.add_row(
                cat.category,
                f"{cat.accuracy * 100:.1f}%",
                f"{cat.cases_passed}/{cat.cases_total}",
            )
        console.print(cat_table)


def _print_diff(diff: DiffReport) -> None:
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


def _print_drift(drift: DriftReport) -> None:
    if drift.window_size < 5:
        # Not enough history yet — silent.
        return
    if drift.has_drift:
        console.print(
            f"[bold yellow]⚠ Slow drift detected[/bold yellow] — latest "
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
