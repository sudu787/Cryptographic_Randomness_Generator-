"""Tests for interval estimation against published or independently derived values."""

from __future__ import annotations

import numpy as np
import pytest

from entropylens.analysis.intervals import (
    bootstrap,
    clopper_pearson,
    newcombe_difference,
    proportion_interval,
    wilson,
    zero_event_upper_bound,
)


def test_wilson_known_value() -> None:
    # 5/10 at 95%: centre 0.5, half-width 1.96*sqrt(2.5 + 0.9604)/13.8416.
    interval = wilson(5, 10)
    assert (interval.low, interval.high) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_clopper_pearson_boundary_cases_have_closed_forms() -> None:
    all_successes = clopper_pearson(200, 200)
    assert all_successes.low == pytest.approx(0.025 ** (1 / 200))  # about 0.9817
    assert all_successes.high == 1.0
    no_successes = clopper_pearson(0, 50)
    assert no_successes.low == 0.0
    assert no_successes.high == pytest.approx(1 - 0.025 ** (1 / 50))


def test_clopper_pearson_contains_wilson_for_moderate_counts() -> None:
    exact, score = clopper_pearson(30, 100), wilson(30, 100)
    assert exact.low <= score.low
    assert score.high <= exact.high


def test_proportion_interval_switches_method_at_the_boundaries() -> None:
    assert proportion_interval(0, 10).method == "clopper-pearson"
    assert proportion_interval(10, 10).method == "clopper-pearson"
    assert proportion_interval(3, 10).method == "wilson"


def test_zero_event_bound_approximates_the_rule_of_three() -> None:
    assert zero_event_upper_bound(1000) == pytest.approx(3 / 1000, rel=0.01)


def test_newcombe_matches_the_published_worked_example() -> None:
    # Newcombe (1998), method 10, example 56/70 vs 48/80: difference 0.2, interval 0.0524 to 0.3339.
    interval = newcombe_difference(56, 70, 48, 80)
    assert interval.estimate == pytest.approx(0.2)
    assert (interval.low, interval.high) == pytest.approx((0.0524, 0.3339), abs=1e-4)


def test_bootstrap_is_reproducible_and_brackets_the_estimate() -> None:
    values = np.random.default_rng(1).exponential(100.0, size=200)
    first = bootstrap(values, seed=7, resamples=2000)
    second = bootstrap(values, seed=7, resamples=2000)
    assert first == second
    assert first.low <= first.estimate <= first.high
    assert first.estimate == pytest.approx(float(np.median(values)))


def test_bootstrap_of_a_constant_sample_is_degenerate() -> None:
    interval = bootstrap([5.0] * 20, resamples=100)
    assert (interval.estimate, interval.low, interval.high) == (5.0, 5.0, 5.0)


@pytest.mark.parametrize(("successes", "trials"), [(-1, 10), (11, 10), (0, 0)])
def test_invalid_counts_are_rejected(successes: int, trials: int) -> None:
    with pytest.raises(ValueError, match="require"):
        wilson(successes, trials)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        clopper_pearson(1, 10, confidence=1.0)
    with pytest.raises(ValueError, match="empty"):
        bootstrap([])
    with pytest.raises(ValueError, match="positive"):
        zero_event_upper_bound(0)
