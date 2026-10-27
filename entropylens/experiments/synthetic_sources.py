"""Simulated raw noise sources (docs/methodology.md section 2).

These models describe the *statistical behaviour* of a noise source. Their randomness comes from
NumPy's PCG64 generator, which is not a cryptographic generator and is never used for anything
security-relevant. The host operating system's RNG is never touched.

Binary models are two-state Markov chains with ``P(1 | previous 0) = p01`` and
``P(1 | previous 1) = p11``. IID Bernoulli(p) is the special case ``p01 = p11 = p``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from entropylens.monitoring.samples import BITS_PER_SAMPLE, SampleKind, Samples


@dataclass(frozen=True, slots=True)
class SourceModel:
    """A nominal noise-source model with its claimed min-entropy per sample.

    In a real system the claimed min-entropy ``H`` comes from an SP 800-90B entropy assessment. Here
    it is known by construction, which is a major simplification (docs/methodology.md section 11).

    Attributes:
        name: Model identifier (``S0``, ``S1``, ``S2``, ``S0-byte``).
        kind: ``"binary"`` (samples are 0/1) or ``"byte"`` (samples are 0..255).
        claimed_entropy: Min-entropy ``H`` per sample used to derive thresholds.
        p01: Binary only: ``P(1 | previous sample 0)``.
        p11: Binary only: ``P(1 | previous sample 1)``.
        description: Human-readable summary.
    """

    name: str
    kind: SampleKind
    claimed_entropy: float
    p01: float = 0.5
    p11: float = 0.5
    description: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.claimed_entropy <= BITS_PER_SAMPLE[self.kind]:
            raise ValueError("claimed_entropy must lie in (0, bits per sample]")
        if not (0 < self.p01 < 1 and 0 < self.p11 < 1):
            raise ValueError("transition probabilities must lie strictly between 0 and 1")
        if self.kind == "byte" and (self.p01, self.p11) != (0.5, 0.5):
            raise ValueError("byte models are uniform IID; transition probabilities do not apply")

    @property
    def is_iid(self) -> bool:
        """True when successive samples are independent."""
        return self.kind == "byte" or self.p01 == self.p11

    @property
    def p_one(self) -> float:
        """Stationary probability of a 1 (binary models)."""
        return self.p01 / (self.p01 + 1 - self.p11)

    def generate(self, rng: np.random.Generator, n: int, previous: int | None = None) -> Samples:
        """Draw ``n`` samples, continuing a stream whose last sample was ``previous``."""
        if self.kind == "byte":
            return rng.integers(0, 256, size=n, dtype=np.uint8)
        return binary_chain(rng, n, self.p01, self.p11, previous)


def binary_chain(
    rng: np.random.Generator, n: int, p01: float, p11: float, previous: int | None = None
) -> Samples:
    """Sample ``n`` values of a binary Markov chain, vectorised via geometric run lengths.

    A run of zeros ends when a 1 follows, which happens with probability ``p01`` per step, so zero
    runs are Geometric(``p01``) and one runs are Geometric(``1 - p11``). The geometric distribution
    is memoryless, so starting a new call from ``previous`` continues the chain exactly.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0 < p01 < 1 and 0 < p11 < 1):
        raise ValueError("transition probabilities must lie strictly between 0 and 1")
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    if p01 == p11:
        return (rng.random(n) < p01).astype(np.uint8)

    stationary = p01 / (p01 + 1 - p11)
    p_first = stationary if previous is None else (p11 if previous else p01)
    value = int(rng.random() < p_first)
    pieces: list[Samples] = []
    produced = 0
    while produced < n:
        # Enough run pairs to cover the remainder with high probability; loop if not.
        pairs = max(16, math.ceil((n - produced) * max(p01, 1 - p11)) + 16)
        zero_runs = rng.geometric(p01, size=pairs)
        one_runs = rng.geometric(1 - p11, size=pairs)
        first, second = (one_runs, zero_runs) if value else (zero_runs, one_runs)
        lengths = np.column_stack((first, second)).ravel()
        values = np.tile(np.array([value, 1 - value], dtype=np.uint8), pairs)
        piece = np.repeat(values, lengths)
        pieces.append(piece)
        produced += piece.size
    return np.concatenate(pieces)[:n]


S0 = SourceModel("S0", "binary", 1.0, 0.5, 0.5, "Ideal IID Bernoulli(0.5)")
S1 = SourceModel(
    "S1", "binary", -math.log2(0.55), 0.55, 0.55, "Healthy but biased IID Bernoulli(0.55)"
)
S2 = SourceModel(
    "S2",
    "binary",
    -math.log2(0.55),
    0.45,
    0.55,
    "Healthy but correlated Markov chain, P(next = current) = 0.55",
)
S0_BYTE = SourceModel("S0-byte", "byte", 8.0, description="Ideal uniform IID bytes")

MODELS: dict[str, SourceModel] = {m.name: m for m in (S0, S1, S2, S0_BYTE)}
ModelName = Literal["S0", "S1", "S2", "S0-byte"]
