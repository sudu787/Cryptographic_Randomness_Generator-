"""Tests for the SP 800-22 subset.

Reference values are computed three independent ways where possible:

1. a direct transcription of each SP 800-22 formula in this file, using only the ``math`` module
   and plain loops (no code shared with the package);
2. exact distributions (enumeration, the David-Barton formula with integer arithmetic, a random-walk
   dynamic program) where the published constants or approximations are in doubt;
3. the values printed in SP 800-22r1a, compared only where they agree with (1) and (2). Where they
   disagree, the test records the published value and the reason.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import special, stats

from entropylens.analysis import statistical_tests as st

# The first 100 binary digits of the expansion used in SP 800-22r1a sections 2.1.8-2.13.8.
E100 = (
    "11001001000011111101101010100010001000010110100011"
    "00001000110100110001001100011001100010100010111000"
)
# The 128-bit input of SP 800-22r1a section 2.4.8.
E128 = (
    "11001100000101010110110001001100111000000000001001"
    "00110101010001000100111101011010000000110101111100"
    "1100111001101101100010110010"
)


def _bits(text: str) -> np.ndarray:
    return np.array([int(c) for c in text], dtype=np.uint8)


# --- Reference transcriptions (independent of the package) ---


def ref_frequency(bits: str) -> float:
    s = sum(1 if c == "1" else -1 for c in bits)
    return math.erfc(abs(s) / math.sqrt(len(bits)) / math.sqrt(2))


def ref_block_frequency(bits: str, m: int) -> float:
    blocks = [bits[i : i + m] for i in range(0, len(bits) - m + 1, m)]
    chi2 = 4 * m * sum((b.count("1") / m - 0.5) ** 2 for b in blocks)
    return float(special.gammaincc(len(blocks) / 2, chi2 / 2))


def ref_runs(bits: str) -> float:
    n = len(bits)
    pi = bits.count("1") / n
    v = 1 + sum(a != b for a, b in itertools.pairwise(bits))
    return math.erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))


def ref_cusum(bits: str, reverse: bool = False) -> float:
    steps = [1 if c == "1" else -1 for c in (reversed(bits) if reverse else bits)]
    partial, z = 0, 0
    for step in steps:
        partial += step
        z = max(z, abs(partial))
    n = len(bits)

    def phi(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2))

    total = 1.0
    for k in range(math.ceil((-n / z + 1) / 4), math.floor((n / z - 1) / 4) + 1):
        total -= phi((4 * k + 1) * z / math.sqrt(n)) - phi((4 * k - 1) * z / math.sqrt(n))
    for k in range(math.ceil((-n / z - 3) / 4), math.floor((n / z - 1) / 4) + 1):
        total += phi((4 * k + 3) * z / math.sqrt(n)) - phi((4 * k + 1) * z / math.sqrt(n))
    return total


def longest_run_cdf_by_formula(block: int, m: int) -> Fraction:
    """David and Barton's formula quoted in SP 800-22 section 3.4, in exact integer arithmetic."""
    total = 0
    for r in range(block + 1):
        for j in range(min(block - r + 1, r // (m + 1)) + 1):
            total += (
                (-1) ** j * math.comb(block - r + 1, j) * math.comb(block - j * (m + 1), block - r)
            )
    return Fraction(total, 2**block)


# --- Worked examples: implementation = reference = published (where the publication is right) ---


@pytest.mark.parametrize(("bits", "published"), [("1011010101", 0.527089), (E100, 0.109599)])
def test_frequency_examples(bits: str, published: float) -> None:
    result = st.frequency(_bits(bits))
    assert result.p_value == pytest.approx(ref_frequency(bits), rel=1e-12)
    assert result.p_value == pytest.approx(published, abs=5e-7)


@pytest.mark.parametrize(
    ("bits", "block", "published"), [("0110011010", 3, 0.801252), (E100, 10, 0.706438)]
)
def test_block_frequency_examples(bits: str, block: int, published: float) -> None:
    result = st.block_frequency(_bits(bits), block)
    assert result.p_value == pytest.approx(ref_block_frequency(bits, block), rel=1e-12)
    assert result.p_value == pytest.approx(published, abs=5e-7)


def test_block_frequency_example_statistic() -> None:
    assert st.block_frequency(_bits(E100), 10).statistic == pytest.approx(7.2)


@pytest.mark.parametrize(("bits", "published"), [("1001101011", 0.147232), (E100, 0.500798)])
def test_runs_examples(bits: str, published: float) -> None:
    # SP 800-22 2.3.4 step (2) says both "the test is not run" and "the runs test is applicable"
    # for the 10-bit example, and 2.3.8 lists tau = 0.02 although 2/sqrt(100) = 0.2. The
    # prerequisite as defined (|pi - 1/2| < 2/sqrt(n)) holds in both examples, so the test runs.
    result = st.runs(_bits(bits))
    assert result.p_value == pytest.approx(ref_runs(bits), rel=1e-12)
    assert result.p_value == pytest.approx(published, abs=5e-7)


def test_runs_example_statistic() -> None:
    assert st.runs(_bits(E100)).statistic == 52


@pytest.mark.parametrize(
    ("bits", "reverse", "published"),
    [("1011010111", False, 0.4116588), (E100, False, 0.219194), (E100, True, 0.114866)],
)
def test_cusum_examples(bits: str, reverse: bool, published: float) -> None:
    # 2.13.4 prints 0.4116588; 2.13.6 prints 0.411658, a truncation. The formula gives 0.41165862.
    # 2.13.8 prints "z = 1.6 / 1.9", which is z/sqrt(n); z = max|S_k| is 16 / 19.
    result = st.cumulative_sums(_bits(bits), reverse=reverse)
    assert result.p_value == pytest.approx(ref_cusum(bits, reverse), rel=1e-12)
    assert result.p_value == pytest.approx(published, abs=5e-7)


def test_cusum_example_statistics_are_integers() -> None:
    assert st.cumulative_sums(_bits(E100)).statistic == 16
    assert st.cumulative_sums(_bits(E100), reverse=True).statistic == 19


def test_longest_run_example_with_exact_probabilities() -> None:
    # SP 800-22 prints two answers for this input: chi2 = 4.882605, p = 0.180598 (2.4.4, computed
    # from the rounded four-digit table) and chi2 = 4.882457, p = 0.180609 (2.4.8). The exact class
    # probabilities give the second.
    result = st.longest_run(_bits(E128))
    assert result.statistic == pytest.approx(4.882457, abs=5e-7)
    assert result.p_value == pytest.approx(0.180609, abs=5e-7)


def test_longest_run_example_with_the_rounded_table_gives_the_other_published_value() -> None:
    rows = _bits(E128).reshape(16, 8)
    classes = np.clip(st._longest_runs(rows), 1, 4) - 1
    observed = np.bincount(classes, minlength=4)
    assert observed.tolist() == [4, 9, 3, 0]  # as published
    rounded = np.array([0.2148, 0.3672, 0.2305, 0.1875])
    chi2 = float(np.sum((observed - 16 * rounded) ** 2 / (16 * rounded)))
    assert chi2 == pytest.approx(4.882605, abs=5e-7)
    assert special.gammaincc(1.5, chi2 / 2) == pytest.approx(0.180598, abs=5e-7)


# --- Longest-run class probabilities: exact, and compared with the printed tables ---


@pytest.mark.parametrize("block", [8, 16])
def test_longest_run_cdf_matches_enumeration(block: int) -> None:
    counts = np.zeros(block + 1)
    for sequence in itertools.product((0, 1), repeat=block):
        runs = [len(list(g)) for value, g in itertools.groupby(sequence) if value == 1]
        counts[max(runs, default=0)] += 1
    assert st.longest_run_cdf(block) == pytest.approx(np.cumsum(counts) / 2**block, abs=1e-15)


@pytest.mark.parametrize("block", [128, 512])
def test_longest_run_cdf_matches_the_exact_combinatorial_formula(block: int) -> None:
    cdf = st.longest_run_cdf(block)
    for m in range(3, 12):
        assert cdf[m] == pytest.approx(float(longest_run_cdf_by_formula(block, m)), abs=1e-15)


@pytest.mark.parametrize(
    ("block", "low", "high", "printed", "max_difference"),
    [
        (8, 1, 4, [0.2148, 0.3672, 0.2305, 0.1875], 5e-5),
        (128, 4, 9, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124], 1.5e-4),
        # The remaining printed tables are imprecise beyond rounding; the documented differences:
        (512, 6, 11, [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124], 0.0130),
        (1000, 7, 12, [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088], 0.0083),
        (10_000, 10, 16, [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727], 0.0017),
    ],
)
def test_exact_class_probabilities_versus_printed_tables(
    block: int, low: int, high: int, printed: list[float], max_difference: float
) -> None:
    exact = st.longest_run_class_probabilities(block, low, high)
    assert exact.sum() == pytest.approx(1.0)
    assert np.abs(exact - np.array(printed)).max() <= max_difference


def test_printed_m512_table_does_not_sum_to_one() -> None:
    assert sum([0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124]) == pytest.approx(1.0059)


# --- Cumulative sums: asymptotic formula against the exact random-walk distribution ---


def _exact_max_excursion_tail(n: int, z: int) -> float:
    """P(max_k |S_k| >= z) for a fair +/-1 walk of n steps, by dynamic programming."""
    probabilities = np.zeros(2 * z + 1)  # positions -z..z; reaching +/- z is absorbed
    probabilities[z] = 1.0
    absorbed = 0.0
    for _ in range(n):
        step = np.zeros_like(probabilities)
        step[1:] += 0.5 * probabilities[:-1]
        step[:-1] += 0.5 * probabilities[1:]
        absorbed += step[0] + step[-1]
        step[0] = step[-1] = 0.0
        probabilities = step
    return absorbed


@pytest.mark.parametrize(("n", "z"), [(2000, 60), (2000, 90), (5000, 150)])
def test_cusum_formula_approximates_the_exact_tail(n: int, z: int) -> None:
    assert st.cusum_p_value(z, n) == pytest.approx(_exact_max_excursion_tail(n, z), abs=0.02)


# --- Behaviour under the null hypothesis (fixed seeds, so deterministic) ---


@pytest.mark.parametrize(
    "name", ["frequency", "block-frequency", "runs", "longest-run", "cusum-forward"]
)
def test_p_values_are_roughly_uniform_for_random_input(name: str) -> None:
    test = {
        "frequency": st.frequency,
        "block-frequency": st.block_frequency,
        "runs": st.runs,
        "longest-run": st.longest_run,
        "cusum-forward": st.cumulative_sums,
    }[name]
    rng = np.random.default_rng(31)
    p_values = [test(rng.integers(0, 2, 10_000, dtype=np.uint8)).p_value for _ in range(400)]
    # SP 800-22 section 4.2 checks the proportion passing and the uniformity of p-values.
    assert sum(p < 0.01 for p in p_values) <= 12  # about 4 expected
    assert stats.kstest(p_values, "uniform").pvalue > 1e-4


def test_obviously_non_random_inputs_are_rejected() -> None:
    biased = (np.random.default_rng(1).random(10_000) < 0.6).astype(np.uint8)
    alternating = np.tile(np.array([0, 1], dtype=np.uint8), 5000)
    assert st.frequency(biased).p_value < 1e-6
    assert st.runs(alternating).p_value < 1e-6
    assert st.cumulative_sums(np.repeat(np.array([1, 0], np.uint8), 5000)).p_value < 1e-6


def test_runs_prerequisite_failure_is_reported_not_interpreted() -> None:
    result = st.runs(np.ones(1000, dtype=np.uint8))
    assert result.p_value == 0.0
    assert not result.applicable
    assert "prerequisite" in result.note


def test_short_inputs_and_unrecommended_block_sizes_are_marked_inapplicable() -> None:
    result = st.frequency(_bits("1011010101"))
    assert not result.applicable
    assert "recommended" in result.note
    # M = 10 is below the recommended minimum block size of 20.
    assert st.block_frequency(_bits(E100), 10).applicable is False


def test_recommended_block_size_meets_all_three_conditions() -> None:
    for n in (100, 1000, 65_536, 1_000_000):
        m = st.recommended_block_size(n)
        assert m >= 20
        assert m > 0.01 * n
        assert n // m < 100


# --- Multiple testing ---


def test_holm_adjustment_known_values() -> None:
    assert st.holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_benjamini_hochberg_known_values() -> None:
    adjusted = st.benjamini_hochberg_adjust([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_battery_rejects_with_holm_and_excludes_inapplicable_tests() -> None:
    random_bits = np.random.default_rng(3).integers(0, 2, 65_536, dtype=np.uint8)
    clean = st.run_battery(random_bits)
    assert len(clean.results) == 6
    assert all(p is not None for p in clean.adjusted_p_values)

    constant = st.run_battery(np.ones(65_536, dtype=np.uint8))
    assert constant.any_rejected
    runs_index = [r.name for r in constant.results].index("runs")
    assert constant.adjusted_p_values[runs_index] is None  # prerequisite failed: excluded


def test_input_conversion() -> None:
    assert st.as_bits(b"\x80\x01").tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="only 0 and 1"):
        st.as_bits(np.array([0, 2], dtype=np.uint8))
    with pytest.raises(TypeError, match="expected"):
        st.as_bits("0101")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="128"):
        st.longest_run(_bits("1" * 100))
