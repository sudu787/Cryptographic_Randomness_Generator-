"""Tests for threshold derivations, checked against SP 800-90B and against exact distributions."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from entropylens.monitoring.thresholds import (
    apt_cutoff,
    binomial_region,
    collision_block_samples,
    rct_cutoff,
    transition_count_pmf,
    transition_region,
    two_sided_region,
)

H_S1 = -math.log2(0.55)

# NIST SP 800-90B (January 2018), section 4.4.2, Table 2: APT cutoffs at alpha = 2^-20.
TABLE_2_BINARY = {0.2: 941, 0.4: 840, 0.6: 748, 0.8: 664, 1.0: 589}  # W = 1024
TABLE_2_NON_BINARY = {0.5: 410, 1.0: 311, 2.0: 177, 4.0: 62, 8.0: 13}  # W = 512


# --- Repetition Count Test ---


def test_rct_matches_the_worked_examples_in_sp800_90b() -> None:
    # Section 4.4.1: alpha = 2^-20, H = 2 gives C = 11; H = 8 with C = 6 is "about once per 10^12"
    # samples, i.e. alpha = 2^-40.
    assert rct_cutoff(2**-20, 2.0).value == 11
    assert rct_cutoff(2**-40, 8.0).value == 6


@pytest.mark.parametrize(
    ("alpha", "h", "expected"),
    [(2**-30, 1.0, 31), (2**-20, 1.0, 21), (2**-30, H_S1, 36), (2**-40, H_S1, 48)],
)
def test_rct_cutoff_values(alpha: float, h: float, expected: int) -> None:
    assert rct_cutoff(alpha, h).value == expected


@pytest.mark.parametrize("alpha", [2**-20, 2**-30, 2**-40, 1e-6])
@pytest.mark.parametrize("h", [0.3, H_S1, 1.0, 2.5, 8.0])
def test_rct_cutoff_is_the_smallest_integer_meeting_the_bound(alpha: float, h: float) -> None:
    cutoff = rct_cutoff(alpha, h)
    assert 2 ** (-h * (cutoff.value - 1)) <= alpha * (1 + 1e-12)
    assert 2 ** (-h * (cutoff.value - 2)) > alpha
    assert cutoff.false_alarm_probability <= alpha * (1 + 1e-12)
    assert str(cutoff.value) in cutoff.derivation


# --- Adaptive Proportion Test ---


@pytest.mark.parametrize(("h", "expected"), sorted(TABLE_2_BINARY.items()))
def test_apt_reproduces_table_2_binary(h: float, expected: int) -> None:
    assert apt_cutoff(2**-20, h, 1024).value == expected


@pytest.mark.parametrize(("h", "expected"), sorted(TABLE_2_NON_BINARY.items()))
def test_apt_reproduces_table_2_non_binary(h: float, expected: int) -> None:
    assert apt_cutoff(2**-20, h, 512).value == expected


@pytest.mark.parametrize("alpha", [2**-10, 2**-20, 2**-30])
@pytest.mark.parametrize(("h", "window"), [(1.0, 1024), (H_S1, 1024), (8.0, 512)])
def test_apt_cutoff_is_the_smallest_meeting_the_binomial_bound(
    alpha: float, h: float, window: int
) -> None:
    value = apt_cutoff(alpha, h, window).value
    assert stats.binom.sf(value - 1, window, 2**-h) <= alpha
    assert stats.binom.sf(value - 2, window, 2**-h) > alpha


def test_apt_reports_the_exact_probability_with_the_first_sample_counted() -> None:
    cutoff = apt_cutoff(2**-30, 8.0, 512)
    exact = stats.binom.sf(cutoff.value - 2, 511, 1 / 256)
    assert cutoff.false_alarm_probability == pytest.approx(exact)
    # The standard's formula understates this case by more than a factor of three.
    assert cutoff.false_alarm_probability > 3 * 2**-30
    assert "x alpha" in cutoff.derivation


def test_apt_rejects_impossible_configurations() -> None:
    with pytest.raises(ValueError, match="no APT cutoff"):
        apt_cutoff(2**-40, 0.001, 16)
    with pytest.raises(ValueError, match="window"):
        apt_cutoff(2**-20, 1.0, 1)


# --- Repeated-block length ---


def test_collision_block_length_meets_union_bound_minimally() -> None:
    block = collision_block_samples(2**-30, 1.0, 1 << 16)
    assert block.value == 46  # 16 + 30 bits
    assert (1 << 16) * 2.0**-block.value <= 2**-30
    assert (1 << 16) * 2.0 ** -(block.value - 1) > 2**-30
    assert collision_block_samples(2**-30, H_S1, 1 << 16).value == math.ceil(46 / H_S1)


# --- Exact two-sided regions ---


@pytest.mark.parametrize(("n", "p"), [(4096, 0.5), (4096, 0.55), (2048, 0.505), (100, 0.3)])
@pytest.mark.parametrize("alpha", [2**-6, 2**-20, 2**-30])
def test_binomial_region_is_exact_and_tight(n: int, p: float, alpha: float) -> None:
    region = binomial_region(n, p, alpha)
    lower = stats.binom.cdf(region.low - 1, n, p) if region.low > 0 else 0.0
    upper = stats.binom.sf(region.high, n, p)
    assert lower <= alpha / 2 * (1 + 1e-9)
    assert upper <= alpha / 2 * (1 + 1e-9)
    assert region.false_alarm_probability == pytest.approx(lower + upper, rel=1e-6, abs=1e-300)
    # Tightness: widening either tail by one value would exceed alpha / 2.
    if region.low <= n:
        assert stats.binom.cdf(region.low, n, p) > alpha / 2
    assert stats.binom.sf(region.high - 1, n, p) > alpha / 2


def test_two_sided_region_on_a_small_known_distribution() -> None:
    pmf = np.array([0.01, 0.04, 0.9, 0.04, 0.01])
    region = two_sided_region(pmf, 0.04, "toy")  # at most 0.02 per tail: only 0 and 4 are rejected
    assert (region.low, region.high) == (1, 3)
    assert region.false_alarm_probability == pytest.approx(0.02)


def test_transition_pmf_matches_binomial_for_a_fair_source() -> None:
    pmf = transition_count_pmf(1000, 0.5)
    assert pmf == pytest.approx(stats.binom.pmf(np.arange(1000), 999, 0.5), abs=1e-15)


@pytest.mark.parametrize("p", [0.3, 0.55])
def test_transition_pmf_matches_brute_force_enumeration(p: float) -> None:
    n = 10
    expected = np.zeros(n)
    for sequence in itertools.product((0, 1), repeat=n):
        ones = sum(sequence)
        changes = sum(a != b for a, b in itertools.pairwise(sequence))
        expected[changes] += p**ones * (1 - p) ** (n - ones)
    assert transition_count_pmf(n, p) == pytest.approx(expected, rel=1e-12)
    assert transition_count_pmf(n, p).sum() == pytest.approx(1.0)


def test_transition_region_is_cached_and_read_only() -> None:
    assert transition_count_pmf(512, 0.55) is transition_count_pmf(512, 0.55)
    with pytest.raises(ValueError, match="read-only"):
        transition_count_pmf(512, 0.55)[0] = 1.0
    region = transition_region(4096, 0.5, 2**-30)
    assert region.low < 2048 < region.high


# --- Validation ---


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, True, "0.01"])
def test_invalid_alpha_is_rejected(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        rct_cutoff(alpha, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(("h", "bits"), [(0, 8), (-1, 8), (1.5, 1), (9, 8)])
def test_invalid_entropy_is_rejected(h: float, bits: int) -> None:
    with pytest.raises(ValueError, match="min-entropy"):
        rct_cutoff(2**-20, h, bits)


def test_invalid_region_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="require"):
        binomial_region(0, 0.5, 0.01)
    with pytest.raises(ValueError, match="require"):
        transition_count_pmf(1, 0.5)
    with pytest.raises(ValueError, match="history"):
        collision_block_samples(2**-20, 1.0, 0)
