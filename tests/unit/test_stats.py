"""Tests for the statistical helpers — Wilson, percentiles, drift."""

from __future__ import annotations

import pytest

from llm_regression_detector.eval.stats import (
    detect_slow_drift,
    intervals_overlap,
    percentile,
    wilson_interval,
)

# ── Wilson interval ─────────────────────────────────────────────────────────


def test_wilson_zero_total_is_degenerate() -> None:
    ci = wilson_interval(0, 0)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high == 0.0


def test_wilson_perfect_score_has_finite_low() -> None:
    """50/50 should have an upper bound at 1.0 but a non-trivial lower bound."""
    ci = wilson_interval(50, 50)
    assert ci.point == 1.0
    assert ci.high == 1.0
    assert 0.92 < ci.low < 1.0


def test_wilson_zero_score_has_finite_high() -> None:
    ci = wilson_interval(0, 50)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert 0.0 < ci.high < 0.08


def test_wilson_centred_interval_brackets_proportion() -> None:
    ci = wilson_interval(40, 50)
    assert ci.low < ci.point < ci.high
    # Half-width on N=50, p=0.8 should be roughly ~0.10
    assert 0.05 < ci.half_width < 0.15


def test_intervals_overlap_detection() -> None:
    a = wilson_interval(45, 50)  # ~0.83 - 0.97
    b = wilson_interval(40, 50)  # ~0.67 - 0.89
    c = wilson_interval(20, 50)  # ~0.30 - 0.55
    assert intervals_overlap(a, b)  # close — overlap expected
    assert not intervals_overlap(a, c)  # far apart — no overlap


# ── Percentiles ─────────────────────────────────────────────────────────────


def test_percentile_empty_returns_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    assert percentile([42.0], 95) == 42.0


def test_percentile_basic_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 50) == 3.0


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0]
    assert percentile(values, 50) == 15.0  # midpoint


def test_percentile_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0, 2.0], 101)


# ── Slow drift ──────────────────────────────────────────────────────────────


def test_drift_skipped_below_min_history() -> None:
    signal = detect_slow_drift([0.95, 0.94], 0.50)
    assert signal.has_drift is False
    assert signal.std_dev == 0.0


def test_drift_detects_low_outlier() -> None:
    history = [0.95, 0.94, 0.96, 0.95, 0.93, 0.94, 0.95]
    signal = detect_slow_drift(history, latest_accuracy=0.70)
    assert signal.has_drift is True
    assert signal.latest == 0.70
    assert signal.threshold > 0.70


def test_drift_clean_run_within_band() -> None:
    history = [0.95, 0.94, 0.96, 0.95, 0.93, 0.94, 0.95]
    signal = detect_slow_drift(history, latest_accuracy=0.94)
    assert signal.has_drift is False
