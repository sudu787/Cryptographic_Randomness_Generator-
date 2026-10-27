"""Output-level statistical tests: a documented subset of NIST SP 800-22 Rev. 1a.

Every test has the same null hypothesis, H0: *the bits are independent and each is 1 with
probability 1/2*. A small p-value is evidence against H0 for the specific property the test
examines. A large p-value is **not** evidence of randomness, and no number of passed tests shows
unpredictability: a
deterministic generator with a published seed passes all of them (``docs/threat_model.md`` section
5). NIST's 2022 decision to revise SP 800-22 includes rejecting its use for assessing cryptographic
generators.

Implementation notes:

* Formulas follow SP 800-22r1a sections 2.1-2.4 and 2.13. The published worked examples contain
  inconsistencies, so the reference values in the tests are recomputed independently, not copied.
* The longest-run class probabilities are **computed exactly** (dynamic programming over the
  current run length) rather than taken from the four-digit table in SP 800-22 section 3.4.
* When many tests are run, :func:`run_battery` applies Holm's step-down correction so that the
  family-wise error rate is at most ``family_alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy import special, stats

Bits = npt.NDArray[np.uint8]

NULL_HYPOTHESIS = "the bits are independent and each equals 1 with probability 1/2"


@dataclass(frozen=True, slots=True)
class TestInfo:
    """Documentation of one test, shown in the dashboard next to its result."""

    name: str
    title: str
    statistic: str
    reference_distribution: str
    minimum_bits: int
    detects: str


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of one test on one bit sequence.

    ``applicable`` is False when the input-size recommendation or a prerequisite is not met. The
    p-value is then reported but must not be interpreted; :func:`run_battery` excludes it.
    """

    name: str
    statistic: float
    p_value: float
    n_bits: int
    applicable: bool
    note: str = ""


TESTS: dict[str, TestInfo] = {
    "frequency": TestInfo(
        "frequency",
        "Frequency (monobit), SP 800-22 2.1",
        "|S_n| / sqrt(n), where S_n is the sum of +1/-1 values",
        "half-normal (large n)",
        100,
        "an excess of ones or zeros overall",
    ),
    "block-frequency": TestInfo(
        "block-frequency",
        "Frequency within a block, SP 800-22 2.2",
        "4M * sum((pi_i - 1/2)^2) over N blocks of M bits",
        "chi-square with N degrees of freedom",
        100,
        "local imbalance of ones and zeros",
    ),
    "runs": TestInfo(
        "runs",
        "Runs, SP 800-22 2.3",
        "total number of runs V_n",
        "normal (large n), conditional on the observed proportion of ones",
        100,
        "oscillation between ones and zeros that is too fast or too slow",
    ),
    "longest-run": TestInfo(
        "longest-run",
        "Longest run of ones in a block, SP 800-22 2.4",
        "chi-square of longest-run class counts over N blocks of M bits",
        "chi-square with K degrees of freedom (exact class probabilities)",
        128,
        "clusters of ones (runs that are too long or too short)",
    ),
    "cusum-forward": TestInfo(
        "cusum-forward",
        "Cumulative sums (forward), SP 800-22 2.13",
        "maximum |partial sum| of +1/-1 values from the start",
        "random-walk maximum (large n)",
        100,
        "too many ones or zeros early in the sequence",
    ),
    "cusum-reverse": TestInfo(
        "cusum-reverse",
        "Cumulative sums (reverse), SP 800-22 2.13",
        "maximum |partial sum| of +1/-1 values from the end",
        "random-walk maximum (large n)",
        100,
        "too many ones or zeros late in the sequence",
    ),
}


def as_bits(data: bytes | bytearray | Bits) -> Bits:
    """Bits of ``data``: bytes are expanded most significant bit first; 0/1 arrays pass through.

    Raises:
        TypeError: For unsupported input types.
        ValueError: If an array contains values other than 0 and 1.
    """
    if isinstance(data, bytes | bytearray):
        return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
    if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
        if data.size and int(data.max()) > 1:
            raise ValueError("bit arrays must contain only 0 and 1")
        return data
    raise TypeError("expected bytes or a one-dimensional uint8 array of bits")


def _result(name: str, statistic: float, p_value: float, n: int, note: str = "") -> TestResult:
    applicable = n >= TESTS[name].minimum_bits and not note
    if n < TESTS[name].minimum_bits:
        note = note or f"fewer than the recommended {TESTS[name].minimum_bits} bits"
    return TestResult(
        name, float(statistic), float(min(max(p_value, 0.0), 1.0)), n, applicable, note
    )


def frequency(data: bytes | Bits) -> TestResult:
    """SP 800-22 2.1: ``p = erfc(|S_n| / sqrt(2n))``."""
    bits = as_bits(data)
    n = bits.size
    if n == 0:
        raise ValueError("cannot test an empty sequence")
    s_obs = abs(2 * int(bits.sum()) - n) / math.sqrt(n)
    return _result("frequency", s_obs, special.erfc(s_obs / math.sqrt(2)), n)


def recommended_block_size(n: int) -> int:
    """Smallest block size meeting SP 800-22 2.2.7: ``M >= 20``, ``M > 0.01 n`` and ``N < 100``."""
    return max(20, n // 99 + 1)


def block_frequency(data: bytes | Bits, block: int | None = None) -> TestResult:
    """SP 800-22 2.2: ``p = igamc(N/2, chi2/2)`` with ``chi2 = 4M sum((pi_i - 1/2)^2)``.

    Unused trailing bits are discarded, as the standard specifies. By default the block size is the
    smallest one meeting the standard's recommendation; an explicit block size outside it is
    reported as not applicable.
    """
    bits = as_bits(data)
    n = bits.size
    block = recommended_block_size(n) if block is None else block
    if block < 1:
        raise ValueError("block must be positive")
    blocks = n // block
    if blocks == 0:
        raise ValueError("the sequence is shorter than one block")
    proportions = bits[: blocks * block].reshape(blocks, block).mean(axis=1)
    chi2 = 4 * block * float(np.sum((proportions - 0.5) ** 2))
    within = block >= 20 and block > 0.01 * n and blocks < 100
    note = "" if within else "block size outside the SP 800-22 recommendation"
    return _result("block-frequency", chi2, special.gammaincc(blocks / 2, chi2 / 2), n, note)


def runs(data: bytes | Bits) -> TestResult:
    """SP 800-22 2.3, including its frequency prerequisite ``|pi - 1/2| < 2/sqrt(n)``.

    If the prerequisite fails, the p-value is 0.0 and the result is marked not applicable.
    """
    bits = as_bits(data)
    n = bits.size
    if n < 2:
        raise ValueError("the runs test needs at least two bits")
    pi = float(bits.mean())
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return _result("runs", v_obs, 0.0, n, "frequency prerequisite failed: test not applicable")
    expected = 2 * n * pi * (1 - pi)
    p_value = special.erfc(abs(v_obs - expected) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    return _result("runs", v_obs, p_value, n)


# --- Longest run of ones ---

#: (block length M, upper bound of the lowest class, lower bound of the highest class), selected
#: by sequence length as in SP 800-22 section 2.4.2.
LONGEST_RUN_DESIGNS: tuple[tuple[int, int, int, int], ...] = (
    # (minimum n, M, lowest class "<= low", highest class ">= high")
    (750_000, 10_000, 10, 16),
    (6_272, 128, 4, 9),
    (128, 8, 1, 4),
)


@lru_cache(maxsize=16)
def longest_run_cdf(block: int) -> npt.NDArray[np.float64]:
    """Exact ``P(longest run of ones <= m)`` for ``m = 0..block`` in ``block`` fair random bits.

    For each cap ``m`` this is a Markov chain over the length of the current run of ones (0..m):
    a 0 resets the run, a 1 extends it, and extending beyond ``m`` leaves the chain. The
    probability of never leaving in ``block`` steps is the sum of ``T^block`` applied to the start
    state, computed by repeated squaring.
    """
    if block < 1:
        raise ValueError("block must be positive")
    cdf = np.ones(block + 1)
    for cap in range(block + 1):
        transition = np.zeros((cap + 1, cap + 1))
        transition[0, :] = 0.5  # next bit 0: the run resets
        transition[np.arange(1, cap + 1), np.arange(cap)] = 0.5  # next bit 1: the run grows
        cdf[cap] = np.linalg.matrix_power(transition, block)[:, 0].sum()
        if cdf[cap] >= 1.0 - 1e-17:
            cdf[cap:] = 1.0
            break
    cdf.setflags(write=False)
    return cdf


@lru_cache(maxsize=16)
def longest_run_class_probabilities(block: int, low: int, high: int) -> npt.NDArray[np.float64]:
    """Exact class probabilities ``{<= low}, {low+1}, ..., {high-1}, {>= high}``."""
    cdf = longest_run_cdf(block)
    middle = [cdf[m] - cdf[m - 1] for m in range(low + 1, high)]
    probabilities = np.array([cdf[low], *middle, 1.0 - cdf[high - 1]])
    probabilities.setflags(write=False)
    return probabilities


def _longest_runs(rows: Bits) -> npt.NDArray[np.int64]:
    """Longest run of ones in each row of a 2-D 0/1 array (vectorised)."""
    m, width = rows.shape
    padded = np.zeros((m, width + 2), dtype=np.int8)
    padded[:, 1:-1] = rows
    edges = np.diff(padded, axis=1)  # +1 where a run starts, -1 just after it ends
    row_starts, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    longest = np.zeros(m, dtype=np.int64)
    np.maximum.at(longest, row_starts, ends - starts)
    return longest


def longest_run(data: bytes | Bits) -> TestResult:
    """SP 800-22 2.4 with exact class probabilities; M chosen by sequence length."""
    bits = as_bits(data)
    n = bits.size
    design = next((d for d in LONGEST_RUN_DESIGNS if n >= d[0]), None)
    if design is None:
        raise ValueError("the longest-run test needs at least 128 bits")
    _, block, low, high = design
    blocks = n // block
    longest = _longest_runs(bits[: blocks * block].reshape(blocks, block))
    classes = np.clip(longest, low, high) - low
    observed = np.bincount(classes, minlength=high - low + 1)
    expected = blocks * longest_run_class_probabilities(block, low, high)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    k = high - low
    return _result("longest-run", chi2, special.gammaincc(k / 2, chi2 / 2), n)


# --- Cumulative sums ---


def cusum_p_value(z: float, n: int) -> float:
    """SP 800-22 2.13 step 4: ``P(max_k |S_k| >= z)`` for a fair random walk of ``n`` steps.

    The series is the standard reflection-principle expression. Summation limits use real-valued
    bounds; terms outside them are negligible (their arguments are beyond +/- sqrt(n)).
    """
    if z <= 0:
        return 1.0
    root = math.sqrt(n)
    ratio = n / z
    first = range(math.ceil((-ratio + 1) / 4), math.floor((ratio - 1) / 4) + 1)
    second = range(math.ceil((-ratio - 3) / 4), math.floor((ratio - 1) / 4) + 1)
    phi = stats.norm.cdf
    sum1 = sum(phi((4 * k + 1) * z / root) - phi((4 * k - 1) * z / root) for k in first)
    sum2 = sum(phi((4 * k + 3) * z / root) - phi((4 * k + 1) * z / root) for k in second)
    return float(1.0 - sum1 + sum2)


def cumulative_sums(data: bytes | Bits, *, reverse: bool = False) -> TestResult:
    """SP 800-22 2.13, forward (mode 0) or reverse (mode 1)."""
    bits = as_bits(data)
    n = bits.size
    if n == 0:
        raise ValueError("cannot test an empty sequence")
    steps = 2 * bits.astype(np.int64) - 1
    if reverse:
        steps = steps[::-1]
    z = int(np.abs(np.cumsum(steps)).max())
    name = "cusum-reverse" if reverse else "cusum-forward"
    return _result(name, z, cusum_p_value(z, n), n)


# --- Multiple testing ---


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control under any dependence)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control; exploratory use)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for position, index in enumerate(order):
        rank = m - position
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


@dataclass(frozen=True, slots=True)
class BatteryResult:
    """All tests on one sequence, with Holm-adjusted p-values over the applicable tests."""

    results: tuple[TestResult, ...]
    adjusted_p_values: tuple[float | None, ...]
    family_alpha: float

    @property
    def rejected(self) -> tuple[str, ...]:
        """Tests whose Holm-adjusted p-value is below ``family_alpha``."""
        return tuple(
            r.name
            for r, p in zip(self.results, self.adjusted_p_values, strict=True)
            if p is not None and p < self.family_alpha
        )

    @property
    def any_rejected(self) -> bool:
        """Whether the battery rejects H0 at the family level."""
        return bool(self.rejected)


BATTERY: tuple[Callable[[Bits], TestResult], ...] = (
    frequency,
    block_frequency,
    runs,
    longest_run,
    cumulative_sums,
    lambda bits: cumulative_sums(bits, reverse=True),
)


def run_battery(data: bytes | Bits, family_alpha: float = 0.01) -> BatteryResult:
    """Run the SP 800-22 subset and control the family-wise error rate with Holm's method.

    Inapplicable results (input too short, or a failed prerequisite) are reported but excluded from
    the family. A failed runs prerequisite means the sequence already failed the frequency test.
    """
    if not 0 < family_alpha < 1:
        raise ValueError("family_alpha must satisfy 0 < alpha < 1")
    bits = as_bits(data)
    results = tuple(test(bits) for test in BATTERY)
    applicable = [i for i, r in enumerate(results) if r.applicable]
    adjusted_subset = holm_adjust([results[i].p_value for i in applicable])
    adjusted: list[float | None] = [None] * len(results)
    for i, value in zip(applicable, adjusted_subset, strict=True):
        adjusted[i] = value
    return BatteryResult(results, tuple(adjusted), family_alpha)
