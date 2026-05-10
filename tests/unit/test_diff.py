from __future__ import annotations

from llm_regression_detector.diff.analyzer import Analyzer, Thresholds, to_alert_payload
from llm_regression_detector.notify.base import AlertSeverity
from tests.conftest import make_run


def test_no_change_is_info() -> None:
    base = make_run("base", pass_ids=["a", "b", "c"], fail_ids=["d"])
    cand = make_run("cand", pass_ids=["a", "b", "c"], fail_ids=["d"])
    diff = Analyzer().diff(base, cand)
    assert diff.severity is AlertSeverity.INFO
    assert diff.regressions == []


def test_small_sample_critical_delta_downgrades_to_warning() -> None:
    """A 25pp drop on N=4 has overlapping CIs — must NOT be CRITICAL."""
    base = make_run("base", pass_ids=["a", "b", "c", "d"], fail_ids=[])
    cand = make_run("cand", pass_ids=["a", "b", "c"], fail_ids=["d"])
    diff = Analyzer().diff(base, cand)
    # Wilson 95% CI on 4/4 vs 3/4 overlaps; the analyser must downgrade to WARNING
    # so we don't fire false CRITICAL alarms on tiny datasets.
    assert diff.severity is AlertSeverity.WARNING
    assert diff.is_significant is False
    assert {r.case_id for r in diff.regressions} == {"d"}


def test_large_sample_significant_drop_is_critical() -> None:
    """100% → 80% on 50 cases — CIs do not overlap → CRITICAL."""
    pass_ids = [f"p{i:03d}" for i in range(50)]
    fail_in_cand = pass_ids[:10]
    base = make_run("base", pass_ids=pass_ids, fail_ids=[])
    cand = make_run(
        "cand",
        pass_ids=[i for i in pass_ids if i not in fail_in_cand],
        fail_ids=fail_in_cand,
    )
    diff = Analyzer(Thresholds()).diff(base, cand)
    assert diff.severity is AlertSeverity.CRITICAL
    assert diff.is_significant is True
    assert len(diff.regressions) == 10


def test_improvement_only_is_info() -> None:
    base = make_run("base", pass_ids=["a"], fail_ids=["b", "c"])
    cand = make_run("cand", pass_ids=["a", "b"], fail_ids=["c"])
    diff = Analyzer().diff(base, cand)
    assert diff.severity is AlertSeverity.INFO
    assert {i.case_id for i in diff.improvements} == {"b"}


def test_to_alert_payload_includes_significance_marker() -> None:
    pass_ids = [f"p{i}" for i in range(20)]
    base = make_run("base", pass_ids=pass_ids, fail_ids=[])
    cand = make_run("cand", pass_ids=pass_ids[:15], fail_ids=pass_ids[15:])
    diff = Analyzer().diff(base, cand)
    payload = to_alert_payload(diff, report_url="https://example.com/r")
    assert "(within noise)" in payload.title or diff.is_significant
    assert payload.report_url == "https://example.com/r"
    assert payload.run_id == "cand"
