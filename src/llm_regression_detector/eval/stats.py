"""Statistical helpers for eval analysis.

Three things live here:

1. **Wilson score interval** — 95% CI for a binary-pass-rate metric. Better than
   the normal-approximation interval at small N, where eval datasets live.
2. **Percentiles** — p50/p95/p99 latency from a sample list.
3. **Slow drift** — does the latest accuracy fall below the moving-average
   minus k·σ band of recent runs?

These are intentionally tiny, dependency-free, and pure — easy to test, easy
to reason about. ``scipy`` is overkill for this scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """95% Wilson score interval for a Bernoulli proportion."""

    point: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0


def wilson_interval(passed: int, total: int, *, z: float = 1.96) -> WilsonInterval:
    """Wilson score interval for ``passed`` successes out of ``total`` trials.

    ``z=1.96`` corresponds to a 95% confidence level (two-sided). Falls back
    to a degenerate interval at ``[0, 0]`` when total == 0.
    """
    if total <= 0:
        return WilsonInterval(point=0.0, low=0.0, high=0.0)

    p = passed / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)

    low = max(0.0, (centre - margin) / denom)
    high = min(1.0, (centre + margin) / denom)
    return WilsonInterval(point=p, low=low, high=high)


def intervals_overlap(a: WilsonInterval, b: WilsonInterval) -> bool:
    """True iff two intervals share any point — ie. the difference is *not* significant."""
    return not (a.high < b.low or b.high < a.low)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy default).

    ``p`` is in ``[0, 100]``. Empty input returns ``0.0`` to keep callers simple.
    """
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")

    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    rank = (p / 100.0) * (n - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    weight = rank - lower_idx
    return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight


@dataclass(frozen=True, slots=True)
class DriftSignal:
    """Output of the slow-drift detector."""

    has_drift: bool
    moving_average: float
    std_dev: float
    threshold: float
    latest: float


def detect_slow_drift(
    historical_accuracies: Sequence[float],
    latest_accuracy: float,
    *,
    k_sigma: float = 2.0,
    min_history: int = 5,
) -> DriftSignal:
    """Flag drift when ``latest_accuracy`` falls below ``MA - k·σ`` of history.

    Skips detection (returns ``has_drift=False``) until at least ``min_history``
    points exist — small samples have unreliable σ.
    """
    if len(historical_accuracies) < min_history:
        return DriftSignal(
            has_drift=False,
            moving_average=0.0,
            std_dev=0.0,
            threshold=0.0,
            latest=latest_accuracy,
        )
    n = len(historical_accuracies)
    ma = sum(historical_accuracies) / n
    variance = sum((x - ma) ** 2 for x in historical_accuracies) / n
    sigma = math.sqrt(variance)
    threshold = ma - k_sigma * sigma
    return DriftSignal(
        has_drift=latest_accuracy < threshold,
        moving_average=ma,
        std_dev=sigma,
        threshold=threshold,
        latest=latest_accuracy,
    )
