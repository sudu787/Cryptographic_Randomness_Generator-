"""Trial runner and detection metrics (docs/methodology.md sections 6 to 8).

A *cell* is one (source model, fault, monitor settings) combination run for many independent trials.
Each trial:

1. derives its own generator from ``SeedSequence(master_seed, spawn_key=(cell, trial))``, so any
   single trial can be re-run in isolation;
2. draws the onset ``t0`` uniformly from ``[warm-up, warm-up + onset_span]``;
3. generates a healthy trace of ``t0 + post_onset_samples`` samples and injects the fault at ``t0``;
4. runs every applicable detector *independently* over the trace in buffers, recording its alarms
   before ``t0``, its first alarm at or after ``t0``, and its evaluation counts.

Recording each detector separately means any detector subset (the full monitor, RCT + APT only, or
leave-one-out ablations) can be scored afterwards without re-running trials.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from entropylens.analysis.intervals import (
    Interval,
    bootstrap,
    clopper_pearson,
    newcombe_difference,
    proportion_interval,
)
from entropylens.experiments.conditioning import CONDITIONING_INPUT_SAMPLES, DIGEST_BYTES
from entropylens.experiments.fault_injection import FaultSpec, InjectedTrace, inject
from entropylens.experiments.synthetic_sources import MODELS, ModelName
from entropylens.monitoring.alerts import DetectorLevel
from entropylens.monitoring.detectors import AVAILABILITY_SPEC
from entropylens.monitoring.source_health import MonitorSettings, build_detectors

AVAILABILITY = AVAILABILITY_SPEC.name


class TrialDesign(BaseModel):
    """Timing of each trial. Defaults are the pre-declared values (methodology section 3)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    warmup_samples: Annotated[int, Field(ge=4096, le=1 << 24)] = 1 << 16
    onset_span: Annotated[int, Field(ge=0, le=1 << 24)] = 1 << 16
    post_onset_samples: Annotated[int, Field(ge=1, le=1 << 26)] = 1 << 20
    buffer_samples: Annotated[int, Field(ge=64, le=1 << 22)] = 1 << 16


class CellSpec(BaseModel):
    """One experimental cell: a source model, an optional fault, and monitor settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]
    model: ModelName
    fault: FaultSpec | None = None
    trials: Annotated[int, Field(ge=1, le=100_000)] = 200
    design: TrialDesign = Field(default_factory=TrialDesign)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)


@dataclass(frozen=True, slots=True)
class DetectorTrial:
    """One detector's behaviour in one trial.

    ``first_alarm_before`` and ``first_alarm_after`` are absolute sample indices. For faulty trials,
    processing after the onset stops at the first alarm, so ``evaluations_after`` and
    ``alarms_after`` are complete only for healthy cells.
    """

    evaluations_before: int
    alarms_before: int
    first_alarm_before: int | None
    evaluations_after: int
    alarms_after: int
    first_alarm_after: int | None


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Outcome of one trial.

    ``faulty_before_buffer[i]`` is the number of faulty samples before the start of buffer ``i``
    on the regular buffer grid. Exposure for any detector subset is derived from it.
    """

    trial: int
    spawn_key: tuple[int, ...]
    onset: int
    length: int
    faulty_samples: int
    detectors: Mapping[str, DetectorTrial]
    faulty_before_buffer: tuple[int, ...]

    def first_alarm(self, names: Iterable[str], *, after_onset: bool) -> int | None:
        """Earliest alarm among ``names``, before or after the onset."""
        indices = [
            index
            for name in names
            if (
                index := (
                    self.detectors[name].first_alarm_after
                    if after_onset
                    else self.detectors[name].first_alarm_before
                )
            )
            is not None
        ]
        return min(indices) if indices else None


@dataclass(frozen=True, slots=True)
class CellResult:
    """All trials of one cell, plus which detectors were active.

    Attributes:
        critical: Active critical detectors, including ``availability``.
        warning: Active exploratory detectors.
        disabled: Inapplicable or disabled detectors, with reasons.
        alphas: Per-evaluation false-alarm budget of each statistical detector.
        false_alarm_probabilities: Worst-case exact (or bounded) false-alarm probability per
            evaluation of each statistical detector under the null model.
    """

    spec: CellSpec
    master_seed: int
    cell_index: int
    trials: tuple[TrialRecord, ...]
    critical: tuple[str, ...]
    warning: tuple[str, ...]
    disabled: Mapping[str, str]
    alphas: Mapping[str, float]
    false_alarm_probabilities: Mapping[str, float]
    elapsed_seconds: float


def _healthy_trace(spec: CellSpec, rng: np.random.Generator) -> tuple[InjectedTrace, int]:
    model = MODELS[spec.model]
    design = spec.design
    onset = design.warmup_samples + int(rng.integers(0, design.onset_span + 1))
    healthy = model.generate(rng, onset + design.post_onset_samples)
    if spec.fault is None:
        return InjectedTrace(healthy, np.zeros(healthy.size, dtype=bool), onset), onset
    trace = inject(healthy, model.kind, spec.fault, onset, rng, lambda n: model.generate(rng, n))
    return trace, onset


def trial_trace(
    spec: CellSpec, master_seed: int, cell_index: int, trial: int
) -> tuple[InjectedTrace, tuple[int, ...]]:
    """The labelled trace of one trial and its spawn key; determined by the three integers."""
    seed = np.random.SeedSequence(master_seed, spawn_key=(cell_index, trial))
    trace, _ = _healthy_trace(spec, np.random.default_rng(seed))
    return trace, tuple(int(k) for k in seed.spawn_key)


def run_trial(spec: CellSpec, master_seed: int, cell_index: int, trial: int) -> TrialRecord:
    """Run one trial; fully determined by ``(master_seed, cell_index, trial)``."""
    trace, spawn_key = trial_trace(spec, master_seed, cell_index, trial)
    return score_trace(spec, trace, trial, spawn_key)


def score_trace(
    spec: CellSpec, trace: InjectedTrace, trial: int, spawn_key: tuple[int, ...]
) -> TrialRecord:
    """Run every applicable detector independently over ``trace`` and record the outcomes."""
    onset = trace.onset
    samples = trace.samples
    buffer = spec.design.buffer_samples
    stop_early = spec.fault is not None
    detectors, _ = build_detectors(MODELS[spec.model], spec.monitor)

    outcomes: dict[str, DetectorTrial] = {}
    for detector in detectors:
        counts = {"eb": 0, "ab": 0, "ea": 0, "aa": 0}
        first_before: int | None = None
        first_after: int | None = None
        for segment_start, segment_end, before in ((0, onset, True), (onset, samples.size, False)):
            for start in range(segment_start, segment_end, buffer):
                update = detector.update(samples[start : min(start + buffer, segment_end)])
                if before:
                    counts["eb"] += update.evaluations
                    counts["ab"] += update.alarms
                    if first_before is None and update.alarms:
                        first_before = int(update.alarm_indices[0])
                else:
                    counts["ea"] += update.evaluations
                    counts["aa"] += update.alarms
                    if first_after is None and update.alarms:
                        first_after = int(update.alarm_indices[0])
                        if stop_early:
                            break
        outcomes[detector.spec.name] = DetectorTrial(
            counts["eb"], counts["ab"], first_before, counts["ea"], counts["aa"], first_after
        )
    outage = trace.outage.index if trace.outage is not None else None
    outcomes[AVAILABILITY] = DetectorTrial(0, 0, None, 0, 1 if outage is not None else 0, outage)

    cumulative = np.concatenate(([0], np.cumsum(trace.faulty, dtype=np.int64)))
    return TrialRecord(
        trial=trial,
        spawn_key=spawn_key,
        onset=onset,
        length=int(samples.size),
        faulty_samples=int(cumulative[-1]),
        detectors=outcomes,
        faulty_before_buffer=tuple(int(c) for c in cumulative[0 : samples.size + 1 : buffer]),
    )


def run_cell(
    spec: CellSpec,
    master_seed: int,
    cell_index: int,
    progress: Callable[[int, int], None] | None = None,
) -> CellResult:
    """Run every trial of ``spec``. ``progress(done, total)`` is called after each trial."""
    started = time.perf_counter()
    model = MODELS[spec.model]
    detectors, disabled = build_detectors(model, spec.monitor)
    trials = []
    for trial in range(spec.trials):
        trials.append(run_trial(spec, master_seed, cell_index, trial))
        if progress is not None:
            progress(trial + 1, spec.trials)
    return CellResult(
        spec=spec,
        master_seed=master_seed,
        cell_index=cell_index,
        trials=tuple(trials),
        critical=tuple(
            [d.spec.name for d in detectors if d.spec.level is DetectorLevel.CRITICAL]
            + [AVAILABILITY]
        ),
        warning=tuple(d.spec.name for d in detectors if d.spec.level is DetectorLevel.WARNING),
        disabled=dict(disabled),
        alphas={d.spec.name: d.spec.alpha for d in detectors if d.spec.alpha is not None},
        false_alarm_probabilities={
            d.spec.name: d.spec.false_alarm_probability
            for d in detectors
            if d.spec.false_alarm_probability is not None
        },
        elapsed_seconds=time.perf_counter() - started,
    )


# --- Metrics ---


@dataclass(frozen=True, slots=True)
class EvaluationRate:
    """Observed false alarms per evaluation on healthy data.

    ``alpha`` is the configured budget; ``false_alarm_probability`` is the worst-case exact (or
    bounded) probability under the null model, which is what the observed rate estimates.
    """

    detector: str
    alarms: int
    evaluations: int
    alpha: float
    false_alarm_probability: float | None
    rate: Interval | None


@dataclass(frozen=True, slots=True)
class CellSummary:
    """Metrics of one cell for one detector subset.

    For a faulty cell, ``detection`` is the detection rate (recall). For a healthy cell, it is the
    rate of alarms within the same post-onset span, which is the control for *excess detection*.
    """

    cell: str
    detectors: tuple[str, ...]
    trials: int
    faulty: bool
    detected: int
    detection: Interval
    pre_onset_alarms: int
    pre_onset_false_alarm: Interval
    latency_median: float | None
    latency_p95: float | None
    latency_median_ci: Interval | None
    exposure_mean_samples: float
    exposure_max_samples: int
    exposure_mean_conditioned_bytes: float
    per_evaluation: tuple[EvaluationRate, ...]


def exposure_samples(record: TrialRecord, first_alarm: int | None, buffer: int) -> int:
    """Faulty samples released before the gate closed (buffer-level gating).

    The buffer containing the first alarm is withheld, and the gate stays latched afterwards.
    """
    if first_alarm is None:
        return record.faulty_samples
    return record.faulty_before_buffer[first_alarm // buffer]


def summarize(
    result: CellResult, detectors: Iterable[str] | None = None, *, bootstrap_seed: int = 0
) -> CellSummary:
    """Score ``result`` for a detector subset (default: every active critical detector)."""
    names = tuple(detectors) if detectors is not None else result.critical
    unknown = set(names) - set(result.critical) - set(result.warning)
    if unknown:
        raise ValueError(f"detectors not active in this cell: {sorted(unknown)}")
    faulty = result.spec.fault is not None
    buffer = result.spec.design.buffer_samples
    n = len(result.trials)

    detected = pre_alarms = 0
    latency_values: list[float] = []
    exposures: list[int] = []
    for record in result.trials:
        before = record.first_alarm(names, after_onset=False)
        after = record.first_alarm(names, after_onset=True)
        pre_alarms += before is not None
        if after is not None:
            detected += 1
            latency_values.append(float(after - record.onset))
        exposures.append(exposure_samples(record, before if before is not None else after, buffer))
    latencies = len(latency_values)

    rates = []
    for name in names:
        if name == AVAILABILITY:
            continue
        alarms = sum(
            r.detectors[name].alarms_before + (0 if faulty else r.detectors[name].alarms_after)
            for r in result.trials
        )
        evaluations = sum(
            r.detectors[name].evaluations_before
            + (0 if faulty else r.detectors[name].evaluations_after)
            for r in result.trials
        )
        interval = clopper_pearson(alarms, evaluations) if evaluations else None
        rates.append(
            EvaluationRate(
                name,
                alarms,
                evaluations,
                result.alphas[name],
                result.false_alarm_probabilities.get(name),
                interval,
            )
        )

    exposure_array = np.asarray(exposures, dtype=np.float64)
    return CellSummary(
        cell=result.spec.name,
        detectors=names,
        trials=n,
        faulty=faulty,
        detected=detected,
        detection=proportion_interval(detected, n),
        pre_onset_alarms=pre_alarms,
        pre_onset_false_alarm=proportion_interval(pre_alarms, n),
        latency_median=float(np.median(latency_values)) if latencies else None,
        latency_p95=float(np.percentile(latency_values, 95)) if latencies else None,
        latency_median_ci=bootstrap(latency_values, seed=bootstrap_seed) if latencies else None,
        exposure_mean_samples=float(exposure_array.mean()),
        exposure_max_samples=int(exposure_array.max()),
        exposure_mean_conditioned_bytes=float(
            (exposure_array // CONDITIONING_INPUT_SAMPLES).mean() * DIGEST_BYTES
        ),
        per_evaluation=tuple(rates),
    )


def excess_detection(faulty: CellSummary, control: CellSummary) -> Interval:
    """Detection rate minus the healthy control's alarm rate over the same span (Newcombe)."""
    if not faulty.faulty or control.faulty:
        raise ValueError("compare a faulty cell against a healthy control cell")
    return newcombe_difference(faulty.detected, faulty.trials, control.detected, control.trials)


def precision_at_prevalence(faulty: CellSummary, control: CellSummary, prevalence: float) -> float:
    """Precision TP / (TP + FP) for a stated fault prevalence.

    Precision depends on how common faults are, so it is only meaningful together with
    ``prevalence``; the experiment design fixes it, it is not a property of the detector.
    """
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must satisfy 0 < prevalence < 1")
    true_positive = faulty.detection.estimate * prevalence
    false_positive = control.detection.estimate * (1 - prevalence)
    total = true_positive + false_positive
    return true_positive / total if total else float("nan")
