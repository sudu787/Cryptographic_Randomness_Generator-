"""Tests for conditioning pipelines, the trial runner, metrics and experiment configuration."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from entropylens.config import ConfigError, DetectionExperimentConfig, parse_config
from entropylens.experiments import fault_injection as fi
from entropylens.experiments.conditioning import (
    drbg_chain,
    pack_samples,
    run_pipeline,
    sha256_condition,
)
from entropylens.experiments.evaluation import (
    CellSpec,
    TrialDesign,
    excess_detection,
    exposure_samples,
    precision_at_prevalence,
    run_cell,
    run_trial,
    summarize,
)
from entropylens.experiments.synthetic_sources import S0
from entropylens.monitoring.alerts import SourceStatus
from entropylens.monitoring.source_health import HealthMonitor, MonitorSettings
from entropylens.security.crypto_reference import ChaCha20Reference

SMALL = TrialDesign(
    warmup_samples=8192, onset_span=4096, post_onset_samples=16384, buffer_samples=4096
)
MASTER_SEED = 20261026


def _cell(name: str, fault: fi.FaultBase | None, trials: int = 5, **kwargs: object) -> CellSpec:
    return CellSpec(name=name, model="S0", fault=fault, trials=trials, design=SMALL, **kwargs)  # type: ignore[arg-type]


# --- Conditioning and pipelines ---


def test_pack_samples_is_msb_first() -> None:
    assert pack_samples(np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8), "binary") == b"\x81"
    assert pack_samples(np.array([7, 255], dtype=np.uint8), "byte") == b"\x07\xff"


def test_sha256_conditioning_hashes_complete_blocks() -> None:
    zeros = np.zeros(1100, dtype=np.uint8)
    output = sha256_condition(zeros, "binary")
    assert output == hashlib.sha256(bytes(64)).digest() * 2  # two 512-sample blocks; 76 left over
    with pytest.raises(ValueError, match="multiple of 8"):
        sha256_condition(zeros, "binary", input_samples=100)


def test_drbg_chain_carries_state_between_inputs() -> None:
    digest = hashlib.sha256(b"x").digest()
    key_1 = hashlib.sha256(bytes(32) + digest).digest()
    key_2 = hashlib.sha256(key_1 + digest).digest()
    expected = ChaCha20Reference.from_key(key_1, bytes(12)).generate(
        40
    ) + ChaCha20Reference.from_key(key_2, bytes(12)).generate(40)
    output = drbg_chain(digest * 2, bytes_per_input=40)
    assert output == expected
    assert output[:40] != output[40:]  # identical inputs still give fresh output
    with pytest.raises(ValueError, match="32-byte"):
        drbg_chain(b"short")


def test_pipeline_without_monitor_releases_everything() -> None:
    samples = S0.generate(np.random.default_rng(0), 32768)
    result = run_pipeline(samples, "binary", None, 4096, faulty=np.ones(32768, dtype=bool))
    assert result.released_samples == 32768
    assert result.released_faulty_samples == 32768
    assert len(result.output) == 32768 // 512 * 32
    assert result.final_status is None


def test_monitored_pipeline_withholds_faulty_buffers() -> None:
    rng = np.random.default_rng(1)
    healthy = S0.generate(rng, 65536)
    trace = fi.inject(
        healthy, "binary", fi.ConstantFault(), 40_000, rng, lambda n: S0.generate(rng, n)
    )
    monitor = HealthMonitor(S0, MonitorSettings())
    result = run_pipeline(trace.samples, "binary", monitor, 4096, trace.faulty)
    assert result.final_status is SourceStatus.FAILED
    assert result.failure is not None
    # The fault starts inside buffer 9 (36864..40959); that buffer and all later ones are withheld.
    assert result.released_samples == 9 * 4096
    assert result.released_faulty_samples == 0


# --- Trials ---


def test_trials_are_reproducible_and_seeded_by_cell_and_trial() -> None:
    spec = _cell("f3", fi.BiasFault(p_one=0.75))
    first = run_trial(spec, MASTER_SEED, 3, 7)
    assert first == run_trial(spec, MASTER_SEED, 3, 7)
    assert first.spawn_key == (3, 7)
    assert first.length == first.onset + SMALL.post_onset_samples
    assert SMALL.warmup_samples <= first.onset <= SMALL.warmup_samples + SMALL.onset_span
    assert run_trial(spec, MASTER_SEED, 3, 8) != first


def test_constant_fault_is_detected_quickly_by_the_rct() -> None:
    result = run_cell(_cell("f1", fi.ConstantFault()), MASTER_SEED, 0)
    summary = summarize(result)
    assert summary.detected == summary.trials == 5
    assert summary.pre_onset_alarms == 0
    assert summary.latency_median is not None
    assert summary.latency_median <= 31
    # At most the faulty samples before the next buffer boundary escape (alarm within 30 samples).
    assert summary.exposure_max_samples <= 30
    for record in result.trials:
        rct = record.detectors["rct"].first_alarm_after
        assert rct is not None
        assert 0 <= rct - record.onset <= 30


def test_ablation_shows_which_detectors_see_a_pattern() -> None:
    result = run_cell(_cell("f4", fi.PatternFault(period=16)), MASTER_SEED, 1)
    assert summarize(result).detected == 5
    assert summarize(result, ["rct"]).detected == 0
    assert summarize(result, ["repeated-block"]).detected == 5


def test_outage_is_detected_at_the_onset_with_no_exposure() -> None:
    summary = summarize(run_cell(_cell("f9", fi.OutageFault()), MASTER_SEED, 2))
    assert summary.detected == 5
    assert summary.latency_median == 0.0
    assert summary.exposure_max_samples == 0


def test_healthy_cell_reports_false_alarm_rates_per_evaluation() -> None:
    result = run_cell(_cell("healthy", None, trials=4), MASTER_SEED, 3)
    summary = summarize(result, list(result.critical) + list(result.warning))
    assert summary.faulty is False
    assert summary.detected == 0
    assert summary.pre_onset_alarms == 0
    rates = {rate.detector: rate for rate in summary.per_evaluation}
    assert set(rates) == {
        "rct",
        "apt",
        "repeated-block",
        "frequency",
        "runs",
        "lag-agreement",
        "autocorrelation",
    }
    assert rates["rct"].evaluations > 40_000  # every run is an evaluation
    assert rates["rct"].alarms == 0
    assert rates["rct"].rate is not None
    assert rates["rct"].rate.low == 0.0
    assert rates["apt"].alpha == 2**-30
    assert rates["apt"].false_alarm_probability is not None


def test_undetected_fault_exposes_every_faulty_sample() -> None:
    result = run_cell(_cell("n1", fi.NegativeControlFault(), trials=3), MASTER_SEED, 4)
    summary = summarize(result)
    assert summary.detected == 0
    assert summary.exposure_mean_samples == SMALL.post_onset_samples
    assert summary.exposure_mean_conditioned_bytes == SMALL.post_onset_samples // 512 * 32


def test_exposure_matches_the_monitored_pipeline() -> None:
    spec = _cell("f3", fi.BiasFault(p_one=0.7), trials=1)
    record = run_trial(spec, MASTER_SEED, 5, 0)
    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(5, 0)))
    onset = SMALL.warmup_samples + int(rng.integers(0, SMALL.onset_span + 1))
    healthy = S0.generate(rng, onset + SMALL.post_onset_samples)
    trace = fi.inject(healthy, "binary", spec.fault, onset, rng, lambda n: S0.generate(rng, n))  # type: ignore[arg-type]
    pipeline = run_pipeline(
        trace.samples,
        "binary",
        HealthMonitor(S0, MonitorSettings()),
        SMALL.buffer_samples,
        trace.faulty,
    )
    first = record.first_alarm(
        [n for n in record.detectors if n != "autocorrelation"], after_onset=True
    )
    assert pipeline.released_faulty_samples == exposure_samples(record, first, SMALL.buffer_samples)


def test_excess_detection_and_precision() -> None:
    faulty = summarize(run_cell(_cell("f1", fi.ConstantFault()), MASTER_SEED, 0))
    control = summarize(run_cell(_cell("healthy", None), MASTER_SEED, 3))
    excess = excess_detection(faulty, control)
    assert excess.estimate == pytest.approx(1.0)
    assert excess.low > 0
    assert precision_at_prevalence(faulty, control, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="healthy control"):
        excess_detection(control, faulty)
    with pytest.raises(ValueError, match="prevalence"):
        precision_at_prevalence(faulty, control, 0.0)


def test_summarize_rejects_unknown_detectors() -> None:
    result = run_cell(_cell("f1", fi.ConstantFault(), trials=1), MASTER_SEED, 0)
    with pytest.raises(ValueError, match="not active"):
        summarize(result, ["byte-chi-square"])


# --- Configuration ---


def _experiment(cells: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "name": "e1-detection",
            "purpose": "development",
            "master_seed": 1,
            "cells": cells,
        }
    )


def test_detection_experiment_config_parses_cells() -> None:
    config = parse_config(
        _experiment(
            [
                {"name": "s0-f1", "model": "S0", "fault": {"fault": "F1"}, "trials": 200},
                {
                    "name": "s1-f10",
                    "model": "S1",
                    "fault": {"fault": "F10", "base": {"fault": "F3", "p_one": 0.7}},
                    "monitor": {"alpha_log2": -40},
                    "design": {"buffer_samples": 4096},
                },
                {"name": "s0-healthy", "model": "S0", "trials": 1000},
            ]
        ),
        DetectionExperimentConfig,
    )
    assert [cell.name for cell in config.cells] == ["s0-f1", "s1-f10", "s0-healthy"]
    assert config.cells[1].monitor.alpha == 2**-40
    assert config.cells[2].fault is None


@pytest.mark.parametrize(
    "cells",
    [
        [],
        [{"name": "a", "model": "S0"}, {"name": "a", "model": "S1"}],
        [{"name": "a", "model": "S9"}],
        [{"name": "a", "model": "S0", "trials": 0}],
        [{"name": "a", "model": "S0", "design": {"warmup_samples": 10}}],
        [{"name": "A", "model": "S0"}],
    ],
)
def test_invalid_experiment_configs_are_rejected(cells: list[dict[str, object]]) -> None:
    with pytest.raises(ConfigError):
        parse_config(_experiment(cells), DetectionExperimentConfig)
