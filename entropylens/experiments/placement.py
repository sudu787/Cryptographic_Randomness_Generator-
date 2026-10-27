"""Experiment E3 (placement): the same fault traces seen by raw monitoring and by output tests.

For every trial the identical labelled raw trace is examined in three ways
(docs/methodology.md section 5):

* **P2**: raw-sample health monitoring (the detectors of :mod:`experiments.evaluation`).
* **P1**: SHA-256-conditioned output (P0) split into blocks of ``block_bits`` bits, each tested by
  the SP 800-22 subset with Holm correction at ``family_alpha``.
* **P3**: the conditioned output passed through the stateful DRBG stand-in, then tested like P1.

Each output block maps back to a span of raw samples (512 raw samples give 256 output bits), so a
rejection is attributed to the raw index at the end of the block's span. Hypothesis H4 predicts
that P1 sees only faults that make the conditioner's input repeat exactly, and P3 sees none.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from entropylens.analysis.intervals import Interval, proportion_interval
from entropylens.analysis.statistical_tests import run_battery
from entropylens.experiments.conditioning import (
    CONDITIONING_INPUT_SAMPLES,
    DIGEST_BYTES,
    drbg_chain,
    sha256_condition,
)
from entropylens.experiments.evaluation import (
    AVAILABILITY,
    CellSpec,
    TrialRecord,
    score_trace,
    trial_trace,
)
from entropylens.experiments.synthetic_sources import MODELS
from entropylens.monitoring.alerts import DetectorLevel
from entropylens.monitoring.source_health import build_detectors

DEFAULT_BLOCK_BITS = 1 << 16
RAW_SAMPLES_PER_OUTPUT_BIT = CONDITIONING_INPUT_SAMPLES // (DIGEST_BYTES * 8)  # = 2
PIPELINES = ("P1", "P2", "P3")


@dataclass(frozen=True, slots=True)
class OutputTestOutcome:
    """Output-test results for one trial and one pipeline.

    ``first_rejection_after`` is the last raw-sample index of the first rejected block whose raw
    span extends past the onset. Testing stops there. Blocks wholly before the onset are all
    tested, and their rejections are false alarms.
    """

    blocks_tested: int
    rejections_before_onset: int
    first_rejection_after: int | None
    rejected_tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlacementTrial:
    """One trial seen by all three pipelines."""

    record: TrialRecord
    p1: OutputTestOutcome
    p3: OutputTestOutcome


def output_tests(
    output: bytes, onset: int, *, block_bits: int = DEFAULT_BLOCK_BITS, family_alpha: float = 0.01
) -> OutputTestOutcome:
    """Run the battery over consecutive output blocks, mapping each back to raw-sample indices."""
    if block_bits < 128 or block_bits % 8:
        raise ValueError("block_bits must be a multiple of 8 and at least 128")
    bits = np.unpackbits(np.frombuffer(output, dtype=np.uint8))
    raw_per_block = block_bits * RAW_SAMPLES_PER_OUTPUT_BIT
    tested = before = 0
    for block in range(bits.size // block_bits):
        battery = run_battery(bits[block * block_bits : (block + 1) * block_bits], family_alpha)
        tested += 1
        raw_end = (block + 1) * raw_per_block
        if battery.any_rejected:
            if raw_end <= onset:
                before += 1
            else:
                return OutputTestOutcome(tested, before, raw_end - 1, battery.rejected)
    return OutputTestOutcome(tested, before, None, ())


def run_placement_trial(
    spec: CellSpec,
    master_seed: int,
    cell_index: int,
    trial: int,
    *,
    block_bits: int = DEFAULT_BLOCK_BITS,
    family_alpha: float = 0.01,
) -> PlacementTrial:
    """Generate one trial's trace once and examine it with P1, P2 and P3."""
    trace, spawn_key = trial_trace(spec, master_seed, cell_index, trial)
    record = score_trace(spec, trace, trial, spawn_key)
    conditioned = sha256_condition(trace.samples, MODELS[spec.model].kind)
    return PlacementTrial(
        record=record,
        p1=output_tests(conditioned, trace.onset, block_bits=block_bits, family_alpha=family_alpha),
        p3=output_tests(
            drbg_chain(conditioned), trace.onset, block_bits=block_bits, family_alpha=family_alpha
        ),
    )


@dataclass(frozen=True, slots=True)
class PlacementResult:
    """All trials of one placement cell."""

    spec: CellSpec
    master_seed: int
    cell_index: int
    trials: tuple[PlacementTrial, ...]
    monitor_detectors: tuple[str, ...]
    block_bits: int
    family_alpha: float
    elapsed_seconds: float


def run_placement_cell(
    spec: CellSpec,
    master_seed: int,
    cell_index: int,
    progress: Callable[[int, int], None] | None = None,
    *,
    block_bits: int = DEFAULT_BLOCK_BITS,
    family_alpha: float = 0.01,
) -> PlacementResult:
    """Run every trial of ``spec`` through P1, P2 and P3."""
    started = time.perf_counter()
    detectors, _ = build_detectors(MODELS[spec.model], spec.monitor)
    critical = tuple(
        [d.spec.name for d in detectors if d.spec.level is DetectorLevel.CRITICAL] + [AVAILABILITY]
    )
    trials = []
    for trial in range(spec.trials):
        trials.append(
            run_placement_trial(
                spec,
                master_seed,
                cell_index,
                trial,
                block_bits=block_bits,
                family_alpha=family_alpha,
            )
        )
        if progress is not None:
            progress(trial + 1, spec.trials)
    return PlacementResult(
        spec,
        master_seed,
        cell_index,
        tuple(trials),
        critical,
        block_bits,
        family_alpha,
        time.perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class PlacementSummary:
    """Detection (post-onset alarm or rejection) per pipeline, with intervals.

    For a healthy cell these rates are the false-alarm controls for H4.
    """

    cell: str
    trials: int
    detected: dict[str, int]
    detection: dict[str, Interval]
    false_alarms_before_onset: dict[str, int]
    median_latency: dict[str, float | None]


def summarize_placement(result: PlacementResult) -> PlacementSummary:
    """Per-pipeline detection counts, rates, pre-onset false alarms and median latency."""
    outcomes: dict[str, list[int | None]] = {name: [] for name in PIPELINES}
    before = dict.fromkeys(PIPELINES, 0)
    for trial in result.trials:
        record = trial.record
        outcomes["P2"].append(record.first_alarm(result.monitor_detectors, after_onset=True))
        before["P2"] += record.first_alarm(result.monitor_detectors, after_onset=False) is not None
        for name, outcome in (("P1", trial.p1), ("P3", trial.p3)):
            outcomes[name].append(outcome.first_rejection_after)
            before[name] += outcome.rejections_before_onset > 0
    n = len(result.trials)
    detected = {name: sum(x is not None for x in values) for name, values in outcomes.items()}
    latency: dict[str, float | None] = {}
    for name, values in outcomes.items():
        delays = [
            x - t.record.onset for x, t in zip(values, result.trials, strict=True) if x is not None
        ]
        latency[name] = float(np.median(delays)) if delays else None
    return PlacementSummary(
        cell=result.spec.name,
        trials=n,
        detected=detected,
        detection={name: proportion_interval(k, n) for name, k in detected.items()},
        false_alarms_before_onset=before,
        median_latency=latency,
    )
