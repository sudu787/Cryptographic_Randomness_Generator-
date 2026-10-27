"""Labelled fault injection into simulated noise-source traces (docs/methodology.md section 3).

A healthy trace is generated first. :func:`inject` then rewrites it from the onset sample onwards
according to a fault specification and returns per-sample ground-truth labels. Faults only ever
affect simulated data.

Fault specifications are strict, frozen Pydantic models, so they can be embedded directly in JSON
experiment configurations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from entropylens.experiments.synthetic_sources import binary_chain
from entropylens.monitoring.samples import SampleKind, Samples
from entropylens.security.crypto_reference import ChaCha20Reference
from entropylens.security.educational_lcg import DRAND48_HIGH32, EducationalLCG

Probability = Annotated[float, Field(gt=0, lt=1)]
BoolArray = npt.NDArray[np.bool_]
HealthyDraw = Callable[[int], Samples]
"""Draws ``n`` fresh samples from the healthy model (used to draw the F4 pattern)."""


class FaultBase(BaseModel):
    """Common configuration for fault specifications: strict, immutable, no unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ConstantFault(FaultBase):
    """F1: the source is stuck at one value."""

    fault: Literal["F1"] = "F1"
    value: Annotated[int, Field(ge=0, le=255)] = 0


class StuckBitFault(FaultBase):
    """F2 (byte sources only): one bit position of every sample is stuck."""

    fault: Literal["F2"] = "F2"
    bit: Annotated[int, Field(ge=0, le=7)] = 0
    value: Annotated[int, Field(ge=0, le=1)] = 0


class BiasFault(FaultBase):
    """F3 (binary sources only): samples become IID Bernoulli(``p_one``)."""

    fault: Literal["F3"] = "F3"
    p_one: Probability = 0.65


class PatternFault(FaultBase):
    """F4: the source repeats a pattern of ``period`` samples, drawn once from the healthy model."""

    fault: Literal["F4"] = "F4"
    period: Annotated[int, Field(ge=2, le=1 << 20)] = 16


class BlockReplayFault(FaultBase):
    """F5: ``block_samples`` samples are re-emitted from ``distance`` samples earlier.

    With ``repeat_every`` set, the replay recurs at that interval; otherwise it happens once.
    """

    fault: Literal["F5"] = "F5"
    block_samples: Annotated[int, Field(ge=1, le=1 << 22)] = 512
    distance: Annotated[int, Field(ge=1, le=1 << 22)] = 4096
    repeat_every: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _repeat_after_block(self) -> BlockReplayFault:
        if self.repeat_every is not None and self.repeat_every < self.block_samples:
            raise ValueError("repeat_every must be at least block_samples")
        return self


class CorrelationFault(FaultBase):
    """F6 (binary only): symmetric Markov chain with ``P(next = current) = stay_probability``."""

    fault: Literal["F6"] = "F6"
    stay_probability: Probability = 0.70


class ChangePointFault(FaultBase):
    """F7 (binary only): abrupt change to a Markov chain with the given transition probabilities."""

    fault: Literal["F7"] = "F7"
    p01: Probability = 0.60
    p11: Probability = 0.60


class ReplayFromStartFault(FaultBase):
    """F8a: from the onset the source replays its own output from ``replay_from`` onwards.

    This is an analogue of a virtual-machine snapshot restore or a reseed with a repeated seed.
    """

    fault: Literal["F8a"] = "F8a"
    replay_from: Annotated[int, Field(ge=0)] = 0


class LowEntropySubstitutionFault(FaultBase):
    """F8b: output replaced by an LCG seeded from a ``seed_bits``-bit seed space (no replay).

    Expected to be **undetectable** by statistical monitoring.
    """

    fault: Literal["F8b"] = "F8b"
    seed_bits: Annotated[int, Field(ge=1, le=32)] = 16


class OutageFault(FaultBase):
    """F9: the source fails at the onset, by raising an exception or returning a short read."""

    fault: Literal["F9"] = "F9"
    mode: Literal["exception", "short-read"] = "exception"


class NegativeControlFault(FaultBase):
    """N1: ChaCha20 keystream under a published key. Zero secrecy, statistically excellent."""

    fault: Literal["N1"] = "N1"


BurstBase = Annotated[
    ConstantFault | BiasFault | PatternFault | CorrelationFault | ChangePointFault,
    Field(discriminator="fault"),
]


class IntermittentFault(FaultBase):
    """F10: ``base`` is active in bursts with geometric on/off durations of the given means."""

    fault: Literal["F10"] = "F10"
    base: BurstBase = Field(default_factory=ConstantFault)
    mean_on: Annotated[float, Field(ge=1)] = 64.0
    mean_off: Annotated[float, Field(ge=1)] = 4096.0


FaultSpec = Annotated[
    ConstantFault
    | StuckBitFault
    | BiasFault
    | PatternFault
    | BlockReplayFault
    | CorrelationFault
    | ChangePointFault
    | ReplayFromStartFault
    | LowEntropySubstitutionFault
    | OutageFault
    | IntermittentFault
    | NegativeControlFault,
    Field(discriminator="fault"),
]

_BINARY_ONLY = {"F3", "F6", "F7"}
_BYTE_ONLY = {"F2"}


@dataclass(frozen=True, slots=True)
class Outage:
    """A source-availability failure at ``index`` (no samples are delivered from there on)."""

    index: int
    mode: Literal["exception", "short-read"]


@dataclass(frozen=True, slots=True)
class InjectedTrace:
    """A trace after fault injection.

    Attributes:
        samples: Samples actually delivered by the source.
        faulty: Ground-truth label per delivered sample (True where the fault affects it).
        onset: Index at which the fault begins.
        outage: Set when the fault is a source outage (F9).
    """

    samples: Samples
    faulty: BoolArray
    onset: int
    outage: Outage | None = None


def check_compatible(fault: FaultBase, kind: SampleKind) -> None:
    """Raise ``ValueError`` if ``fault`` cannot be applied to a ``kind`` source."""
    name = getattr(fault, "fault", "")
    base = getattr(fault, "base", None)
    names = {name} | ({getattr(base, "fault", "")} if base is not None else set())
    if kind == "byte" and names & _BINARY_ONLY:
        raise ValueError(f"fault {sorted(names & _BINARY_ONLY)} applies only to binary sources")
    if kind == "binary" and names & _BYTE_ONLY:
        raise ValueError("fault F2 (stuck bit) applies only to byte sources")
    if kind == "binary" and isinstance(fault, ConstantFault) and fault.value > 1:
        raise ValueError("binary sources can only be stuck at 0 or 1")


def _copy_from_earlier(samples: Samples, start: int, length: int, distance: int) -> None:
    """Set ``samples[start + i] = samples[start + i - distance]`` in order, for ``i < length``.

    Copying in chunks of ``distance`` keeps the sequential semantics when the source and destination
    overlap (``distance < length``), which turns the replay into a periodic repetition.
    """
    end = min(start + length, samples.size)
    position = start
    while position < end:
        chunk = min(distance, end - position)
        samples[position : position + chunk] = samples[
            position - distance : position - distance + chunk
        ]
        position += chunk


def _to_kind(stream: bytes, kind: SampleKind, n: int) -> Samples:
    data = np.frombuffer(stream, dtype=np.uint8)
    return np.unpackbits(data)[:n] if kind == "binary" else data[:n].copy()


def _replacement(
    fault: FaultBase,
    kind: SampleKind,
    n: int,
    previous: int,
    healthy_draw: HealthyDraw,
    rng: np.random.Generator,
) -> Samples:
    """Faulty samples for ``n`` positions that simply replace the healthy ones."""
    if isinstance(fault, ConstantFault):
        return np.full(n, fault.value, dtype=np.uint8)
    if isinstance(fault, BiasFault):
        return binary_chain(rng, n, fault.p_one, fault.p_one, previous)
    if isinstance(fault, CorrelationFault):
        return binary_chain(rng, n, 1 - fault.stay_probability, fault.stay_probability, previous)
    if isinstance(fault, ChangePointFault):
        return binary_chain(rng, n, fault.p01, fault.p11, previous)
    if isinstance(fault, PatternFault):
        return np.resize(healthy_draw(fault.period), n)
    if isinstance(fault, LowEntropySubstitutionFault):
        seed = int(rng.integers(0, 1 << fault.seed_bits))
        width = n if kind == "byte" else -(-n // 8)
        return _to_kind(EducationalLCG(seed, DRAND48_HIGH32).generate(width), kind, n)
    if isinstance(fault, NegativeControlFault):
        width = n if kind == "byte" else -(-n // 8)
        return _to_kind(ChaCha20Reference.negative_control().generate(width), kind, n)
    raise TypeError(f"{type(fault).__name__} is not a replacement fault")


def burst_mask(rng: np.random.Generator, n: int, mean_on: float, mean_off: float) -> BoolArray:
    """Boolean mask of length ``n`` alternating on/off bursts, starting with an on burst.

    Burst lengths are geometric with the given means, so bursts start and stop without memory.
    """
    if mean_on < 1 or mean_off < 1:
        raise ValueError("mean burst lengths must be at least 1")
    pieces: list[BoolArray] = []
    produced = 0
    while produced < n:
        pairs = max(8, int((n - produced) / (mean_on + mean_off)) + 8)
        on = rng.geometric(1 / mean_on, size=pairs)
        off = rng.geometric(1 / mean_off, size=pairs)
        lengths = np.column_stack((on, off)).ravel()
        piece = np.repeat(np.tile(np.array([True, False]), pairs), lengths)
        pieces.append(piece)
        produced += piece.size
    return np.concatenate(pieces)[:n]


def inject(
    healthy: Samples,
    kind: SampleKind,
    fault: FaultBase,
    onset: int,
    rng: np.random.Generator,
    healthy_draw: HealthyDraw,
) -> InjectedTrace:
    """Apply ``fault`` to ``healthy`` from sample ``onset`` onwards.

    Args:
        healthy: The healthy trace. It is not modified.
        kind: Sample kind of the trace.
        fault: The fault specification.
        onset: First affected sample index (``0 < onset <= len(healthy)``).
        rng: Generator for any randomness the fault needs (consumed in a fixed order).
        healthy_draw: Draws fresh healthy samples (used to draw the F4 pattern).

    Returns:
        The delivered samples, their ground-truth labels, and any outage event.
    """
    check_compatible(fault, kind)
    if not 0 < onset <= healthy.size:
        raise ValueError("onset must satisfy 0 < onset <= len(healthy)")
    samples = healthy.copy()
    faulty = np.zeros(samples.size, dtype=bool)
    n_after = samples.size - onset
    previous = int(samples[onset - 1])

    if isinstance(fault, OutageFault):
        return InjectedTrace(samples[:onset], faulty[:onset], onset, Outage(onset, fault.mode))

    if isinstance(fault, StuckBitFault):
        after = samples[onset:]
        mask = np.uint8(1 << fault.bit)
        samples[onset:] = (after | mask) if fault.value else (after & ~mask)
        faulty[onset:] = True
    elif isinstance(fault, BlockReplayFault):
        if fault.distance > onset:
            raise ValueError("replay distance reaches before the start of the trace")
        starts = range(onset, samples.size, fault.repeat_every) if fault.repeat_every else (onset,)
        for start in starts:
            _copy_from_earlier(samples, start, fault.block_samples, fault.distance)
            faulty[start : start + fault.block_samples] = True
    elif isinstance(fault, ReplayFromStartFault):
        if fault.replay_from >= onset:
            raise ValueError("replay_from must lie before the onset")
        # A restored snapshot re-emits the original deterministic stream from replay_from onwards.
        samples[onset:] = healthy[fault.replay_from : fault.replay_from + n_after]
        faulty[onset:] = True
    elif isinstance(fault, IntermittentFault):
        active = burst_mask(rng, n_after, fault.mean_on, fault.mean_off)
        burst_samples = _replacement(fault.base, kind, n_after, previous, healthy_draw, rng)
        samples[onset:][active] = burst_samples[active]
        faulty[onset:] = active
    else:
        samples[onset:] = _replacement(fault, kind, n_after, previous, healthy_draw, rng)
        faulty[onset:] = True
    return InjectedTrace(samples, faulty, onset)
