"""Layer B: an educational Linear Congruential Generator (LCG). INSECURE BY DESIGN.

Recurrence::

    x[n+1] = (a * x[n] + c) mod m

with modulus ``m``, multiplier ``a``, increment ``c`` and seed ``x[0]``. The entire internal state
is one integer in ``[0, m)``, so the generator has at most ``m`` states and its output is periodic
with period at most ``m``. :func:`has_full_period` checks the Hull-Dobell conditions for period
exactly ``m``.

Why an LCG must never be used for security:

1. **The seed determines everything.** A guessable seed, such as a timestamp, gives guessable
   output. :func:`recover_seed` enumerates candidate seeds.
2. **Output reveals state.** When each output is the full state, one output predicts all later ones
   (:func:`predict_outputs`). Three consecutive outputs usually reveal an unknown multiplier and
   increment (:func:`recover_parameters`).
3. **Truncation does not help.** Hiding the low 16 bits of a 48-bit state leaves only 2^16
   candidates, and a second output identifies the right one (:func:`recover_truncated_state`).
4. **Low-order bits are weak.** For ``m = 2^k``, bit ``j`` of the state has period at most
   ``2^(j+1)``. With odd ``a`` and ``c``, bit 0 simply alternates.

These functions demonstrate the attacks on a deliberately weak generator in the dashboard and the
professor-facing demo. They do not attack any real system.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from entropylens.security.base import GeneratorInfo, validate_request_size

MAX_BRUTE_FORCE_TRIALS = 1 << 24
"""Upper bound on candidates any recovery function will enumerate, so a UI call cannot hang."""


@dataclass(frozen=True, slots=True)
class LCGParameters:
    """Parameters of an LCG and of how its output is taken from the state.

    Each output is ``(state >> output_shift) mod 2^output_bits``. ``output_bits`` must be a multiple
    of 8 so that outputs serialise to whole bytes.

    Attributes:
        name: Short identifier used in generator IDs.
        multiplier: ``a``, with ``0 < a < m``.
        increment: ``c``, with ``0 <= c < m``.
        modulus: ``m``, with ``m >= 2``.
        output_shift: Number of low state bits discarded from each output.
        output_bits: Number of state bits kept in each output.
        reference: Where the parameter values come from.
    """

    name: str
    multiplier: int
    increment: int
    modulus: int
    output_shift: int
    output_bits: int
    reference: str

    def __post_init__(self) -> None:
        if self.modulus < 2:
            raise ValueError("modulus must be at least 2")
        if not 0 < self.multiplier < self.modulus:
            raise ValueError("multiplier must satisfy 0 < a < m")
        if not 0 <= self.increment < self.modulus:
            raise ValueError("increment must satisfy 0 <= c < m")
        if self.output_shift < 0 or self.output_bits <= 0 or self.output_bits % 8:
            raise ValueError("output_shift must be >= 0 and output_bits a positive multiple of 8")
        if self.output_shift + self.output_bits > self.state_bits:
            raise ValueError("the output window extends beyond the state bits")

    @property
    def state_bits(self) -> int:
        """Number of bits needed to represent any state in ``[0, m)``."""
        return (self.modulus - 1).bit_length()

    @property
    def outputs_full_state(self) -> bool:
        """True when each output is the entire state."""
        return self.output_shift == 0 and self.output_bits == self.state_bits

    def step(self, state: int) -> int:
        """Apply the recurrence once."""
        return (self.multiplier * state + self.increment) % self.modulus

    def output_of(self, state: int) -> int:
        """Extract the output value from a state."""
        return (state >> self.output_shift) & ((1 << self.output_bits) - 1)


NUMERICAL_RECIPES_32 = LCGParameters(
    name="numerical-recipes-32",
    multiplier=1664525,
    increment=1013904223,
    modulus=1 << 32,
    output_shift=0,
    output_bits=32,
    reference="Press et al., Numerical Recipes in C, 2nd ed. (1992), ch. 7 'quick and dirty' LCG",
)
"""Full-state 32-bit LCG: every output is the complete internal state."""

DRAND48_HIGH32 = LCGParameters(
    name="drand48-high32",
    multiplier=0x5DEECE66D,
    increment=0xB,
    modulus=1 << 48,
    output_shift=16,
    output_bits=32,
    reference="POSIX drand48 (IEEE Std 1003.1-2024): a, c, m; output floor(X/2^16) as in mrand48",
)
"""48-bit LCG whose outputs hide the 16 low-order state bits, as POSIX ``mrand48()`` does."""

PRESETS: dict[str, LCGParameters] = {p.name: p for p in (NUMERICAL_RECIPES_32, DRAND48_HIGH32)}


def _validate_state(value: object, params: LCGParameters, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, not {type(value).__name__}")
    if not 0 <= value < params.modulus:
        raise ValueError(f"{what} must satisfy 0 <= {what} < m = {params.modulus}")
    return value


class EducationalLCG:
    """Deterministic LCG byte generator for teaching. **Never use it for security.**

    The output is a continuous byte stream: each output is serialised little-endian, and bytes left
    over from one ``generate`` call start the next one. ``generate(3)`` followed by
    ``generate(5)`` therefore equals ``generate(8)`` on a fresh instance with the same seed.

    The seed and state are deliberately exposed, because this generator exists to show why such
    exposure (or guessability) is fatal.
    """

    __slots__ = ("_info", "_leftover", "_params", "_seed", "_state")

    def __init__(self, seed: int, params: LCGParameters = NUMERICAL_RECIPES_32) -> None:
        self._params = params
        self._seed = _validate_state(seed, params, "seed")
        self._state = self._seed
        self._leftover = b""
        is_power_of_two = params.modulus == 1 << params.state_bits
        modulus_text = f"2^{params.state_bits}" if is_power_of_two else str(params.modulus)
        last_bit = params.output_shift + params.output_bits - 1
        self._info = GeneratorInfo(
            identifier=f"lcg-{params.name}",
            display_name=f"Educational LCG ({params.name}), INSECURE",
            security_status="educational-insecure",
            deterministic=True,
            backend=(
                f"x[n+1] = ({params.multiplier}*x[n] + {params.increment}) mod {modulus_text}; "
                f"output = state bits {params.output_shift}..{last_bit}"
            ),
        )

    @classmethod
    def seeded_from_clock(
        cls,
        params: LCGParameters = NUMERICAL_RECIPES_32,
        *,
        clock: Callable[[], float] = time.time,
    ) -> EducationalLCG:
        """Seed from the current Unix time in whole seconds.

        This is a **deliberately bad practice**, kept only for the predictable-seed demonstration.
        Anyone who knows roughly when the generator was seeded can recover the seed with
        :func:`recover_seed`.
        """
        return cls(int(clock()) % params.modulus, params)

    @property
    def info(self) -> GeneratorInfo:
        """Non-secret description of this generator."""
        return self._info

    @property
    def params(self) -> LCGParameters:
        """The recurrence parameters."""
        return self._params

    @property
    def seed(self) -> int:
        """The initial state. Not secret: this generator has no secrets worth keeping."""
        return self._seed

    @property
    def state(self) -> int:
        """The current internal state, exposed on purpose for teaching."""
        return self._state

    def next_output(self) -> int:
        """Advance the state once and return the resulting output value."""
        self._state = self._params.step(self._state)
        return self._params.output_of(self._state)

    def outputs(self, count: int) -> list[int]:
        """Return the next ``count`` output values."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"count must be an int, not {type(count).__name__}")
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.next_output() for _ in range(count)]

    def generate(self, n_bytes: int) -> bytes:
        """Return the next ``n_bytes`` bytes of the output stream."""
        size = validate_request_size(n_bytes)
        width = self._params.output_bits // 8
        needed = size - len(self._leftover)
        chunks = [self._leftover]
        if needed > 0:
            count = -(-needed // width)  # ceiling division
            chunks.extend(value.to_bytes(width, "little") for value in self.outputs(count))
        stream = b"".join(chunks)
        self._leftover = stream[size:]
        return stream[:size]

    def __repr__(self) -> str:
        return f"EducationalLCG(seed={self._seed}, params={self._params.name!r})"


class VectorizedLCG:
    """NumPy implementation emitting exactly the same byte stream as :class:`EducationalLCG`.

    Used only to separate algorithm cost from interpreter cost in benchmarks. It precomputes the
    jump coefficients ``x[n+j] = A_j x[n] + C_j (mod m)`` for ``j = 1..block`` and evaluates a
    whole block of states with one vectorised multiply-add. Arithmetic is ``uint64`` with
    wrap-around, which is exact modulo ``m`` because ``m`` is a power of two not above ``2^64``.
    **Insecure by design**, exactly like the scalar version.
    """

    __slots__ = ("_add", "_dtype", "_info", "_leftover", "_mask", "_mul", "_params", "_state")

    _WIDTHS: ClassVar[dict[int, str]] = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}

    def __init__(
        self, seed: int, params: LCGParameters = NUMERICAL_RECIPES_32, block: int = 4096
    ) -> None:
        if params.modulus != 1 << params.state_bits or params.state_bits > 64:
            raise ValueError("VectorizedLCG needs a power-of-two modulus of at most 2^64")
        if params.output_bits not in self._WIDTHS:
            raise ValueError("VectorizedLCG supports output widths of 8, 16, 32 or 64 bits")
        if block < 1:
            raise ValueError("block must be positive")
        self._params = params
        self._state = _validate_state(seed, params, "seed")
        self._leftover = b""
        self._mask = np.uint64(params.modulus - 1)
        self._dtype = np.dtype(self._WIDTHS[params.output_bits])
        multipliers, increments = [], []
        a_j, c_j = 1, 0
        for _ in range(block):
            a_j = (a_j * params.multiplier) % params.modulus
            c_j = (c_j * params.multiplier + params.increment) % params.modulus
            multipliers.append(a_j)
            increments.append(c_j)
        self._mul = np.array(multipliers, dtype=np.uint64)
        self._add = np.array(increments, dtype=np.uint64)
        scalar = EducationalLCG(self._state, params).info
        self._info = GeneratorInfo(
            identifier=f"{scalar.identifier}-numpy",
            display_name=f"Educational LCG ({params.name}, NumPy), INSECURE",
            security_status="educational-insecure",
            deterministic=True,
            backend=f"{scalar.backend}; vectorised with NumPy",
        )

    @property
    def info(self) -> GeneratorInfo:
        """Non-secret description of this generator."""
        return self._info

    def _outputs(self, count: int) -> npt.NDArray[np.uint64]:
        chunks = []
        block = self._mul.size
        for start in range(0, count, block):
            size = min(block, count - start)
            states = (self._mul[:size] * np.uint64(self._state) + self._add[:size]) & self._mask
            self._state = int(states[-1])
            chunks.append(states)
        states = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint64)
        out_mask = np.uint64((1 << self._params.output_bits) - 1)
        return (states >> np.uint64(self._params.output_shift)) & out_mask

    def generate(self, n_bytes: int) -> bytes:
        """Return the next ``n_bytes`` bytes of the output stream."""
        size = validate_request_size(n_bytes)
        width = self._params.output_bits // 8
        needed = size - len(self._leftover)
        stream = self._leftover
        if needed > 0:
            count = -(-needed // width)
            stream += self._outputs(count).astype(self._dtype).tobytes()
        self._leftover = stream[size:]
        return stream[:size]


# --- Demonstration attacks ------------------------------------------------------------------------


def predict_outputs(params: LCGParameters, state: int, count: int) -> list[int]:
    """Predict the ``count`` outputs that follow internal state ``state``.

    For a full-state LCG (:attr:`LCGParameters.outputs_full_state`), the last observed output *is*
    the state, so observing one output is enough to predict everything that follows.
    """
    current = _validate_state(state, params, "state")
    predictions = []
    for _ in range(count):
        current = params.step(current)
        predictions.append(params.output_of(current))
    return predictions


def recover_parameters(
    observed: Sequence[int], modulus: int, *, max_candidates: int = 1 << 16
) -> list[tuple[int, int]]:
    """Recover unknown ``(a, c)`` of a full-state LCG with known modulus from consecutive outputs.

    Consecutive differences satisfy ``d[i+1] = a * d[i] (mod m)``. With ``g = gcd(d[i], m)``, a
    solution exists only if ``g`` divides ``d[i+1]``, and then there are exactly ``g`` candidates
    for ``a`` modulo ``m``. Each candidate fixes ``c = x[1] - a*x[0] (mod m)``. Candidates that
    fail to reproduce every observed transition are discarded.

    Args:
        observed: At least three consecutive outputs (each output being the full state).
        modulus: The known modulus ``m``.
        max_candidates: Refuse to enumerate more candidates than this.

    Returns:
        All ``(a, c)`` pairs consistent with every observation, sorted. More than one pair means
        more observations are needed. An empty list means the outputs did not come from such an
        LCG.

    Raises:
        ValueError: With fewer than three outputs, outputs outside ``[0, m)``, or when the
            observations are too uninformative (more than ``max_candidates`` candidates).
    """
    if len(observed) < 3:
        raise ValueError("at least three consecutive outputs are needed")
    if modulus < 2 or any(not 0 <= x < modulus for x in observed):
        raise ValueError("outputs must lie in [0, modulus) with modulus >= 2")

    diffs = [(y - x) % modulus for x, y in pairwise(observed)]
    # Use the equation whose coefficient shares the fewest factors with m (fewest candidates).
    i = min(range(len(diffs) - 1), key=lambda j: math.gcd(diffs[j], modulus))
    g = math.gcd(diffs[i], modulus)
    if diffs[i + 1] % g:
        return []
    if g > max_candidates:
        raise ValueError(f"{g} candidate multipliers; provide more varied observations")

    reduced_modulus = modulus // g
    base = (diffs[i + 1] // g) * pow(diffs[i] // g, -1, reduced_modulus) % reduced_modulus
    consistent = set()
    for k in range(g):
        a = base + k * reduced_modulus
        if a == 0:
            continue
        c = (observed[1] - a * observed[0]) % modulus
        if all((a * x + c) % modulus == y for x, y in pairwise(observed)):
            consistent.add((a, c))
    return sorted(consistent)


def recover_truncated_state(params: LCGParameters, observed: Sequence[int]) -> list[int]:
    """Recover the internal state of an LCG whose outputs are the top bits of the state.

    The first output fixes the high bits of the corresponding state, leaving ``2^output_shift``
    candidates for the hidden low bits. Each candidate is stepped forward and checked against the
    remaining outputs.

    Args:
        params: Parameters whose output window is the top ``output_bits`` of the state.
        observed: At least two consecutive outputs.

    Returns:
        Every candidate for the state *after* the last observed output; usually exactly one.
        Pass it to :func:`predict_outputs` to predict what comes next.

    Raises:
        ValueError: If the parameters do not output the top state bits, fewer than two outputs are
            given, or the search space exceeds :data:`MAX_BRUTE_FORCE_TRIALS`.
    """
    if params.output_shift + params.output_bits != params.state_bits:
        raise ValueError("outputs must be the top bits of the state")
    if len(observed) < 2:
        raise ValueError("at least two consecutive outputs are needed")
    if any(not 0 <= value < 1 << params.output_bits for value in observed):
        raise ValueError(f"outputs must lie in [0, 2^{params.output_bits})")
    hidden = 1 << params.output_shift
    if hidden > MAX_BRUTE_FORCE_TRIALS:
        raise ValueError(f"{hidden} candidates exceeds the brute-force limit")

    high = observed[0] << params.output_shift
    states = []
    for low in range(hidden):
        state = high | low
        if state >= params.modulus:
            break
        for expected in observed[1:]:
            state = params.step(state)
            if params.output_of(state) != expected:
                break
        else:
            states.append(state)
    return states


def recover_seed(
    params: LCGParameters,
    observed: Sequence[int],
    candidates: Iterable[int],
    *,
    max_trials: int = MAX_BRUTE_FORCE_TRIALS,
) -> list[int]:
    """Find every candidate seed whose first outputs equal ``observed``.

    This models an attacker who knows *how* a seed was chosen, for example "Unix time in seconds
    around 14:00 today", and simply tries every possibility.

    Raises:
        ValueError: If ``observed`` is empty or more than ``max_trials`` candidates are supplied.
    """
    if not observed:
        raise ValueError("at least one observed output is needed")
    matches = []
    for trials, seed in enumerate(candidates, start=1):
        if trials > max_trials:
            raise ValueError(f"more than {max_trials} candidate seeds supplied")
        if 0 <= seed < params.modulus and predict_outputs(params, seed, len(observed)) == list(
            observed
        ):
            matches.append(seed)
    return matches


def has_full_period(multiplier: int, increment: int, modulus: int) -> bool:
    """Hull-Dobell theorem: does the LCG visit all ``m`` states from every seed?

    For ``c != 0`` the period is ``m`` for every seed if and only if:

    1. ``gcd(c, m) = 1``;
    2. ``a - 1`` is divisible by every prime factor of ``m``;
    3. ``a - 1`` is divisible by 4 if ``m`` is divisible by 4.

    With ``c = 0`` the state 0 maps to itself, so the period is always less than ``m``.
    Prime factors are found by trial division, which is fast for the power-of-two moduli used here
    and slower for large prime moduli.
    """
    if modulus < 2 or not 0 < multiplier < modulus or not 0 <= increment < modulus:
        raise ValueError("require m >= 2, 0 < a < m and 0 <= c < m")
    if increment == 0 or math.gcd(increment, modulus) != 1:
        return False
    if modulus % 4 == 0 and (multiplier - 1) % 4:
        return False
    return all((multiplier - 1) % p == 0 for p in _prime_factors(modulus))


def _prime_factors(n: int) -> set[int]:
    factors = set()
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.add(p)
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        factors.add(n)
    return factors
