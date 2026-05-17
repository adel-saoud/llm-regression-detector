from __future__ import annotations

from pathlib import Path

from llm_regression_detector.diff.analyzer import Analyzer
from llm_regression_detector.diff.drift import DriftReport
from llm_regression_detector.eval.dataset import CategoryAccuracy
from llm_regression_detector.report.html import render_html, write_html
from llm_regression_detector.report.pr_comment import render_pr_comment
from tests.conftest import make_run


def test_render_html_without_diff_includes_summary_metrics() -> None:
    run = make_run("r1", pass_ids=["a", "b", "c"], fail_ids=["d"])
    html = render_html(run)
    assert "Eval report" in html
    assert "75.0%" in html  # 3/4 pass
    assert run.run_id in html


def test_render_html_with_diff_shows_regressions() -> None:
    pass_ids = [f"p{i}" for i in range(40)]
    base = make_run("base", pass_ids=pass_ids, fail_ids=[])
    cand = make_run("cand", pass_ids=pass_ids[:28], fail_ids=pass_ids[28:])
    diff = Analyzer().diff(base, cand)
    html = render_html(cand, diff=diff, baseline=base)
    assert "Got worse" in html
    assert "CRITICAL" in html


def test_write_html_creates_file(tmp_path: Path) -> None:
    run = make_run("r1", pass_ids=["a"], fail_ids=[])
    out = write_html(tmp_path / "nested" / "out.html", run)
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_pr_comment_no_baseline() -> None:
    run = make_run("r1", pass_ids=["a", "b"], fail_ids=[])
    body = render_pr_comment(run, diff=None)
    assert "No baseline" in body
    assert run.run_id in body


def test_pr_comment_with_critical_regression() -> None:
    pass_ids = [f"p{i}" for i in range(30)]
    base = make_run("base", pass_ids=pass_ids, fail_ids=[])
    cand = make_run("cand", pass_ids=pass_ids[:21], fail_ids=pass_ids[21:])
    diff = Analyzer().diff(base, cand)
    body = render_pr_comment(cand, diff=diff, report_url="https://example.com/r")
    assert "Critical regression" in body
    assert "merge blocked" in body
    assert "View full report" in body


def test_pr_comment_includes_improvements() -> None:
    base = make_run("base", pass_ids=["a"], fail_ids=["b", "c"])
    cand = make_run("cand", pass_ids=["a", "b", "c"], fail_ids=[])
    diff = Analyzer().diff(base, cand)
    body = render_pr_comment(cand, diff=diff)
    assert "Improvements" in body


def test_pr_comment_with_per_category_breakdown() -> None:
    per_cat = [
        CategoryAccuracy(category="billing", accuracy=0.9, cases_total=10, cases_passed=9),
        CategoryAccuracy(category="technical", accuracy=0.6, cases_total=10, cases_passed=6),
    ]
    base = make_run("base", pass_ids=[f"p{i}" for i in range(20)], fail_ids=[])
    cand = make_run(
        "cand",
        pass_ids=[f"p{i}" for i in range(15)],
        fail_ids=[f"p{i}" for i in range(15, 20)],
        per_category=per_cat,
    )
    diff = Analyzer().diff(base, cand)
    body = render_pr_comment(cand, diff=diff)
    assert "Per-category accuracy" in body
    assert "billing" in body
    assert "technical" in body


def test_pr_comment_with_drift_no_drift() -> None:
    run = make_run("r", pass_ids=["a", "b"], fail_ids=[])
    drift = DriftReport(
        has_drift=False,
        moving_average=0.95,
        std_dev=0.01,
        threshold=0.93,
        latest_accuracy=0.95,
        window_size=7,
    )
    body = render_pr_comment(run, diff=None, drift=drift)
    assert "Slow-drift check" in body
    assert "No drift" in body


def test_pr_comment_with_drift_detected() -> None:
    pass_ids = [f"p{i}" for i in range(30)]
    base = make_run("base", pass_ids=pass_ids, fail_ids=[])
    cand = make_run("cand", pass_ids=pass_ids[:21], fail_ids=pass_ids[21:])
    diff = Analyzer().diff(base, cand)
    drift = DriftReport(
        has_drift=True,
        moving_average=0.95,
        std_dev=0.02,
        threshold=0.91,
        latest_accuracy=0.70,
        window_size=7,
    )
    body = render_pr_comment(cand, diff=diff, drift=drift)
    assert "below the" in body
    assert "MA − 2σ" in body
