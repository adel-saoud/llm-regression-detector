"""Tests for the slow-drift wrapper over EvalRun history."""

from __future__ import annotations

from llm_regression_detector.diff.drift import analyse_drift
from tests.conftest import make_run


def test_no_drift_when_history_too_short() -> None:
    history = [
        make_run(f"r{i}", pass_ids=[f"p{j}" for j in range(8)], fail_ids=[]) for i in range(2)
    ]
    latest = make_run("latest", pass_ids=[f"p{j}" for j in range(8)], fail_ids=[])
    drift = analyse_drift(history, latest)
    assert drift.has_drift is False
    assert drift.window_size == 2


def test_drift_detected_when_latest_collapses() -> None:
    # 7 strong runs, then a weak one
    history = [
        make_run(f"r{i}", pass_ids=[f"p{j}" for j in range(20)], fail_ids=[]) for i in range(7)
    ]
    latest = make_run(
        "latest", pass_ids=[f"p{j}" for j in range(8)], fail_ids=[f"f{j}" for j in range(12)]
    )
    drift = analyse_drift(history, latest)
    assert drift.has_drift is True
    assert drift.window_size == 7


def test_no_drift_when_latest_within_band() -> None:
    # Mixed accuracies create a non-zero σ; latest within MA - 2σ is fine.
    pass_counts = [18, 17, 19, 18, 17, 19, 18]
    history = [
        make_run(
            f"r{i}",
            pass_ids=[f"p{j}" for j in range(pc)],
            fail_ids=[f"f{j}" for j in range(20 - pc)],
        )
        for i, pc in enumerate(pass_counts)
    ]
    latest = make_run(
        "latest",
        pass_ids=[f"p{j}" for j in range(17)],
        fail_ids=[f"f{j}" for j in range(3)],
    )
    drift = analyse_drift(history, latest)
    assert drift.has_drift is False
