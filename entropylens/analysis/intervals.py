"""Interval estimates used by the evaluation harness, benchmarks and reports.

Conventions follow ``docs/methodology.md`` section 7. Proportions use Wilson score intervals, or
Clopper-Pearson exact intervals when the count is 0 or n. Differences of proportions use Newcombe's
hybrid score interval. Medians and ratios use a percentile bootstrap with a recorded seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    method: str
    confidence: float = 0.95


def _check_counts(successes: int, trials: int) -> None:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(
            f"require trials >= 1 and 0 <= successes <= trials, got {successes}/{trials}"
        )


def _check_confidence(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must satisfy 0 < confidence < 1")
    return confidence


def wilson(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    _check_counts(successes, trials)
    z = float(stats.norm.ppf(0.5 + _check_confidence(confidence) / 2))
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), "wilson", confidence)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Clopper-Pearson exact interval for a binomial proportion."""
    _check_counts(successes, trials)
    tail = (1 - _check_confidence(confidence)) / 2
    low = 0.0 if successes == 0 else float(stats.beta.ppf(tail, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1 - tail, successes + 1, trials - successes))
    )
    return Interval(successes / trials, low, high, "clopper-pearson", confidence)


def proportion_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """The methodology's default: Clopper-Pearson at 0 or n, Wilson otherwise."""
    if successes in (0, trials):
        return clopper_pearson(successes, trials, confidence)
    return wilson(successes, trials, confidence)


def zero_event_upper_bound(trials: int, confidence: float = 0.95) -> float:
    """One-sided upper confidence bound on a rate after observing zero events (about 3/n at 95%)."""
    if trials < 1:
        raise ValueError("trials must be positive")
    return float(1 - (1 - _check_confidence(confidence)) ** (1 / trials))


def newcombe_difference(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int, confidence: float = 0.95
) -> Interval:
    """Newcombe's hybrid score interval for ``p_a - p_b`` (independent samples)."""
    a = wilson(successes_a, trials_a, confidence)
    b = wilson(successes_b, trials_b, confidence)
    difference = a.estimate - b.estimate
    low = difference - math.sqrt((a.estimate - a.low) ** 2 + (b.high - b.estimate) ** 2)
    high = difference + math.sqrt((a.high - a.estimate) ** 2 + (b.estimate - b.low) ** 2)
    return Interval(difference, low, high, "newcombe-hybrid-score", confidence)


def bootstrap(
    values: Sequence[float] | npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.median,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for ``statistic`` of ``values``.

    The resampling seed is a parameter so that the interval is reproducible and can be recorded.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    replicates = np.apply_along_axis(statistic, 1, data[indices])
    tail = (1 - _check_confidence(confidence)) / 2
    low, high = np.quantile(replicates, [tail, 1 - tail])
    return Interval(
        float(statistic(data)), float(low), float(high), "percentile-bootstrap", confidence
    )
