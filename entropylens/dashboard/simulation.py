"""Computation behind the interactive pages, kept free of Streamlit so it can be unit-tested.

Interactive runs reuse the experiment machinery (``trial_trace``, ``score_trace``, ``run_cell``,
``run_pipeline``), so what the dashboard shows is computed exactly as in the reported experiments.
They use the seed typed into the page, which is a development seed, never the pre-registered
evaluation seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from entropylens.experiments.conditioning import PipelineResult, run_pipeline, sha256_condition
from entropylens.experiments.evaluation import (
    CellSpec,
    TrialDesign,
    TrialRecord,
    score_trace,
    trial_trace,
)
from entropylens.experiments.fault_injection import FaultBase, InjectedTrace
from entropylens.experiments.placement import OutputTestOutcome, output_tests
from entropylens.experiments.synthetic_sources import MODELS, ModelName
from entropylens.monitoring.alerts import Alert, SourceStatus
from entropylens.monitoring.source_health import HealthMonitor, MonitorSettings

INTERACTIVE_BLOCK_BITS = 1 << 14


def lab_spec(
    model: ModelName,
    fault: FaultBase | None,
    *,
    warmup: int,
    post_onset: int,
    buffer: int,
    settings: MonitorSettings,
    trials: int = 1,
) -> CellSpec:
    """A cell whose onset is fixed at ``warmup`` (no random onset span), for interactive use."""
    return CellSpec.model_validate(
        {
            "name": "interactive",
            "model": model,
            "fault": fault,
            "trials": trials,
            "design": TrialDesign(
                warmup_samples=warmup,
                onset_span=0,
                post_onset_samples=post_onset,
                buffer_samples=buffer,
            ),
            "monitor": settings,
        }
    )


@dataclass(frozen=True, slots=True)
class MonitorRun:
    """A trace examined by the streaming monitor, buffer by buffer."""

    trace: InjectedTrace
    record: TrialRecord
    buffer_ends: list[int]
    statuses: list[SourceStatus]
    alerts: list[Alert]
    failure: Alert | None
    disabled: dict[str, str]


def run_lab_trace(spec: CellSpec, seed: int) -> MonitorRun:
    """Generate one labelled trace, score every detector, and stream it through the monitor."""
    trace, spawn_key = trial_trace(spec, seed, 0, 0)
    record = score_trace(spec, trace, 0, spawn_key)
    monitor = HealthMonitor(MODELS[spec.model], spec.monitor)
    ends: list[int] = []
    statuses: list[SourceStatus] = []
    alerts: list[Alert] = []
    buffer = spec.design.buffer_samples
    for start in range(0, trace.samples.size, buffer):
        report = monitor.process(trace.samples[start : start + buffer])
        ends.append(min(start + buffer, trace.samples.size))
        statuses.append(report.status)
        alerts.extend(report.alerts)
    if trace.outage is not None:
        report = monitor.report_source_error(trace.outage.mode, "simulated outage")
        ends.append(trace.samples.size)
        statuses.append(report.status)
        alerts.extend(report.alerts)
    return MonitorRun(
        trace, record, ends, statuses, alerts, monitor.failure, dict(monitor.disabled)
    )


@dataclass(frozen=True, slots=True)
class PipelineComparison:
    """The same trace without monitoring (P0, plus P1 output tests) and with it (P2)."""

    unmonitored: PipelineResult
    monitored: PipelineResult
    output_tests: OutputTestOutcome


def compare_pipelines(spec: CellSpec, trace: InjectedTrace) -> PipelineComparison:
    """Exposure with and without the monitor, and what output tests say about P0's output."""
    model = MODELS[spec.model]
    buffer = spec.design.buffer_samples
    unmonitored = run_pipeline(trace.samples, model.kind, None, buffer, trace.faulty)
    monitored = run_pipeline(
        trace.samples, model.kind, HealthMonitor(model, spec.monitor), buffer, trace.faulty
    )
    outcome = output_tests(
        sha256_condition(trace.samples, model.kind),
        trace.onset,
        block_bits=INTERACTIVE_BLOCK_BITS,
    )
    return PipelineComparison(unmonitored, monitored, outcome)


def windowed_fraction(
    samples: npt.NDArray[np.uint8], binary: bool, window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Fraction of one-bits per window: (last sample index of each window, fraction)."""
    count = samples.size // window
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    rows = samples[: count * window].reshape(count, window)
    if not binary:
        rows = np.unpackbits(rows, axis=1)
    ends = (np.arange(1, count + 1, dtype=np.int64) * window) - 1
    return ends, rows.mean(axis=1)


def windowed_longest_run(
    samples: npt.NDArray[np.uint8], window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Longest run of identical samples per window (the RCT's statistic)."""
    count = samples.size // window
    longest = np.zeros(count, dtype=np.int64)
    for i in range(count):
        row = samples[i * window : (i + 1) * window]
        boundaries = np.flatnonzero(row[1:] != row[:-1]) + 1
        edges = np.concatenate(([0], boundaries, [window]))
        longest[i] = int(np.diff(edges).max())
    ends = (np.arange(1, count + 1, dtype=np.int64) * window) - 1
    return ends, longest
