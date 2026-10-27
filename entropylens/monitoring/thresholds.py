"""Threshold derivations for the health detectors.

Every critical threshold depends only on the false-alarm budget ``alpha`` and on the nominal source
model, and never on fault data (``docs/methodology.md`` sections 1 and 4). Each function also
returns a human-readable derivation, which the dashboard shows next to every alert.

All critical thresholds use exact distributions or rigorous bounds:

* RCT and APT cutoffs follow NIST SP 800-90B section 4.4 (checked against its Table 2 in the tests).
* Two-sided binomial and transition-count acceptance regions come from exact probability mass
  functions, so the false-alarm probability per evaluation is at most ``alpha``.
* The repeated-block length comes from a union bound on collision probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy import stats

# Guards math.ceil against floating-point error when -log2(alpha)/H is mathematically an integer.
_CEIL_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Cutoff:
    """A one-sided cutoff: the detector alarms when its statistic reaches ``value``.

    ``false_alarm_probability`` is the worst-case probability of an alarm per evaluation under the
    nominal model: exact for the RCT and APT, a union bound for the repeated-block length.
    """

    value: int
    derivation: str
    false_alarm_probability: float


@dataclass(frozen=True, slots=True)
class AcceptanceRegion:
    """Two-sided acceptance region ``[low, high]``: the detector alarms outside it.

    ``false_alarm_probability`` is the exact probability of an alarm under the null model, which is
    at most the configured ``alpha``.
    """

    low: int
    high: int
    false_alarm_probability: float
    derivation: str


def validate_alpha(alpha: float) -> float:
    """Return ``alpha`` if ``0 < alpha < 1``, else raise ``ValueError``."""
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not 0 < alpha < 1:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha!r}")
    return float(alpha)


def validate_entropy(h: float, bits_per_sample: int) -> float:
    """Return ``h`` if ``0 < h <= bits_per_sample``, else raise ``ValueError``."""
    if isinstance(h, bool) or not isinstance(h, int | float) or not 0 < h <= bits_per_sample:
        raise ValueError(f"min-entropy H must satisfy 0 < H <= {bits_per_sample}, got {h!r}")
    return float(h)


def rct_cutoff(alpha: float, h: float, bits_per_sample: int = 8) -> Cutoff:
    """Repetition Count Test cutoff ``C = 1 + ceil(-log2(alpha) / H)`` (SP 800-90B section 4.4.1).

    ``C`` is the smallest integer with ``2^(-H(C-1)) <= alpha``. Under the nominal model, a run that
    starts at a given sample reaches length ``C`` with probability at most ``alpha``.
    """
    alpha = validate_alpha(alpha)
    h = validate_entropy(h, bits_per_sample)
    ratio = -math.log2(alpha) / h
    value = 1 + math.ceil(ratio - _CEIL_TOLERANCE)
    return Cutoff(
        value=value,
        derivation=(
            f"C = 1 + ceil(-log2(alpha) / H) = 1 + ceil({-math.log2(alpha):.4g} / {h:.4f}) "
            f"= {value}"
        ),
        false_alarm_probability=float(2.0 ** (-h * (value - 1))),
    )


def apt_cutoff(alpha: float, h: float, window: int, bits_per_sample: int = 8) -> Cutoff:
    """Adaptive Proportion Test cutoff (SP 800-90B section 4.4.2).

    ``C = 1 + CRITBINOM(W, 2^-H, 1 - alpha)``, which is the smallest ``c`` with
    ``P(Binomial(W, 2^-H) >= c) <= alpha``. This reproduces the standard's Table 2.

    The test as specified always counts the window's first sample, so its statistic is
    ``B = 1 + Binomial(W - 1, P(A))``. The returned ``false_alarm_probability`` is the exact
    worst case ``P(1 + Binomial(W - 1, 2^-H) >= C)``, which can exceed ``alpha``: for uniform bytes
    (H = 8, W = 512) at alpha = 2^-30 it is about 3.5 alpha. The cutoff itself follows the standard.

    Raises:
        ValueError: If no cutoff within the window satisfies the bound, so the test could never
            fire within its false-alarm budget.
    """
    alpha = validate_alpha(alpha)
    h = validate_entropy(h, bits_per_sample)
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError(f"window must be an int >= 2, got {window!r}")
    p = 2.0**-h
    counts = np.arange(1, window + 1)
    tail = stats.binom.sf(counts - 1, window, p)  # P(B >= c) for c = 1..W
    feasible = np.flatnonzero(tail <= alpha)
    if feasible.size == 0:
        raise ValueError(f"no APT cutoff <= W={window} achieves alpha={alpha:.3g} with H={h}")
    value = int(counts[feasible[0]])
    exact = float(stats.binom.sf(value - 2, window - 1, p))  # P(1 + Bin(W-1, p) >= C)
    return Cutoff(
        value=value,
        derivation=(
            f"C = smallest c with P(Binomial(W={window}, p=2^-H={p:.4f}) >= c) <= alpha="
            f"{alpha:.3g}; C = {value}. Exact false-alarm probability with the first sample "
            f"counted: {exact:.3g} ({exact / alpha:.2f} x alpha)"
        ),
        false_alarm_probability=exact,
    )


def collision_block_samples(alpha: float, h: float, history_samples: int) -> Cutoff:
    """Smallest block length ``k`` with ``history * 2^(-k H) <= alpha``.

    The repeated-block detector compares each new block of ``k`` samples with every ``k``-sample
    window in a history of ``history_samples`` positions. Two non-overlapping blocks of independent
    samples, each with min-entropy at least ``H`` per sample, are equal with probability at most
    ``2^(-k H)``. The union bound over the history then caps the false-alarm probability per
    evaluated block at ``alpha``.
    """
    alpha = validate_alpha(alpha)
    if isinstance(history_samples, bool) or not isinstance(history_samples, int):
        raise ValueError("history_samples must be an int")
    if history_samples < 1:
        raise ValueError("history_samples must be positive")
    if not h > 0:
        raise ValueError("min-entropy H must be positive")
    bits_needed = math.log2(history_samples) - math.log2(alpha)
    value = math.ceil(bits_needed / h - _CEIL_TOLERANCE)
    return Cutoff(
        value=value,
        derivation=(
            f"k = ceil((log2(L) - log2(alpha)) / H) = ceil(({math.log2(history_samples):.4g} + "
            f"{-math.log2(alpha):.4g}) / {h:.4f}) = {value} samples"
        ),
        false_alarm_probability=min(1.0, history_samples * 2.0 ** (-value * h)),
    )


def two_sided_region(pmf: npt.NDArray[np.float64], alpha: float, label: str) -> AcceptanceRegion:
    """Exact two-sided acceptance region with at most ``alpha / 2`` probability in each tail.

    Args:
        pmf: Probability mass function over the statistic's values ``0 .. len(pmf) - 1``.
        alpha: Total false-alarm budget per evaluation.
        label: Description of the null distribution for the derivation text.
    """
    alpha = validate_alpha(alpha)
    lower_tail = np.cumsum(pmf)  # P(X <= x)
    upper_tail = np.cumsum(pmf[::-1])[::-1]  # P(X >= x)
    low = int(np.count_nonzero(lower_tail <= alpha / 2))  # P(X < low) <= alpha/2
    above = np.flatnonzero(upper_tail <= alpha / 2)  # candidates for high + 1
    high = int(above[0]) - 1 if above.size else len(pmf) - 1
    p_low = float(lower_tail[low - 1]) if low > 0 else 0.0
    p_high = float(upper_tail[high + 1]) if high + 1 < len(pmf) else 0.0
    return AcceptanceRegion(
        low=low,
        high=high,
        false_alarm_probability=p_low + p_high,
        derivation=(
            f"exact two-sided region under {label}: alarm if statistic < {low} or > {high}; "
            f"P(alarm) = {p_low + p_high:.3g} <= alpha = {alpha:.3g}"
        ),
    )


def binomial_region(n: int, p: float, alpha: float, label: str = "") -> AcceptanceRegion:
    """Exact two-sided region for a ``Binomial(n, p)`` statistic."""
    if n < 1 or not 0 < p < 1:
        raise ValueError("require n >= 1 and 0 < p < 1")
    pmf = stats.binom.pmf(np.arange(n + 1), n, p)
    return two_sided_region(pmf, alpha, label or f"Binomial(n={n}, p={p:.4f})")


@lru_cache(maxsize=32)
def transition_count_pmf(n: int, p: float) -> npt.NDArray[np.float64]:
    """Exact distribution of the number of value changes in ``n`` IID Bernoulli(``p``) samples.

    Dynamic programming over (last value, transitions so far): ``O(n^2)`` work, computed once per
    ``(n, p)`` and cached. For ``p = 0.5`` the result equals ``Binomial(n - 1, 1/2)``.
    """
    if n < 2 or not 0 < p < 1:
        raise ValueError("require n >= 2 and 0 < p < 1")
    q = 1.0 - p
    ends_zero = np.zeros(n)  # index t: P(last sample is 0 and t transitions so far)
    ends_one = np.zeros(n)
    ends_zero[0], ends_one[0] = q, p
    for _ in range(n - 1):
        new_zero = q * ends_zero
        new_zero[1:] += q * ends_one[:-1]
        new_one = p * ends_one
        new_one[1:] += p * ends_zero[:-1]
        ends_zero, ends_one = new_zero, new_one
    pmf = ends_zero + ends_one
    pmf.setflags(write=False)
    return pmf


def transition_region(n: int, p: float, alpha: float) -> AcceptanceRegion:
    """Exact two-sided region for the transition count of ``n`` IID Bernoulli(``p``) samples."""
    return two_sided_region(
        transition_count_pmf(n, p), alpha, f"transition count of {n} IID Bernoulli({p:.4f})"
    )
