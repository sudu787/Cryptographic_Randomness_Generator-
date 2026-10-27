"""Tests for the streaming detectors: exact behaviour on crafted inputs, streaming invariance, no
false alarms on fixed healthy fixtures, and calibration of alarm rates against exact nulls."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from entropylens.monitoring.alerts import Outcome
from entropylens.monitoring.detectors import (
    AdaptiveProportionTest,
    AutocorrelationDetector,
    ByteChiSquareDetector,
    Detector,
    FrequencyDetector,
    LagAgreementDetector,
    RepeatedBlockDetector,
    RepetitionCountTest,
    RunsDetector,
    bit_windows,
    window_keys,
)

ALPHA = 2**-30
H_S1 = -math.log2(0.55)


def _bits(seed: int, n: int, p: float = 0.5) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < p).astype(np.uint8)


def _feed(detector: Detector, samples: np.ndarray, buffer: int) -> tuple[int, list[int]]:
    evaluations, alarms = 0, []
    for start in range(0, samples.size, buffer):
        update = detector.update(samples[start : start + buffer])
        evaluations += update.evaluations
        alarms.extend(update.alarm_indices.tolist())
    return evaluations, alarms


def _all_detectors() -> list[Detector]:
    return [
        RepetitionCountTest(ALPHA, 1.0, "binary"),
        AdaptiveProportionTest(ALPHA, 1.0, "binary"),
        RepeatedBlockDetector(ALPHA, 1.0, "binary", history_samples=4096),
        FrequencyDetector(ALPHA, 0.5, 1024),
        RunsDetector(ALPHA, 0.5, 1024),
        LagAgreementDetector(ALPHA, 0.5, 1024),
        AutocorrelationDetector(1e-3, 1024),
    ]


# --- Bit-window helpers ---


@pytest.mark.parametrize("width", [1, 7, 8, 9, 46, 63, 64])
def test_bit_windows_match_naive_computation(width: int) -> None:
    bits = _bits(1, 300)
    expected = [int("".join(map(str, bits[i : i + width])), 2) for i in range(300 - width + 1)]
    assert bit_windows(bits, width).tolist() == expected


def test_window_keys_support_multiword_and_byte_windows() -> None:
    bits = _bits(2, 400)
    keys = window_keys(bits, "binary", 100)  # 100 bits: two 64-bit words
    assert keys.shape == (301, 2)
    assert int(keys[5, 0]) == int("".join(map(str, bits[5:69])), 2)
    assert int(keys[5, 1]) == int("".join(map(str, bits[69:105])), 2)

    data = np.random.default_rng(3).integers(0, 256, 50, dtype=np.uint8)
    byte_keys = window_keys(data, "byte", 3)
    assert byte_keys.shape == (48, 1)
    assert int(byte_keys[10, 0]) == int.from_bytes(data[10:13].tobytes(), "big")


def test_bit_windows_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="width"):
        bit_windows(_bits(0, 100), 65)


# --- Repetition Count Test ---


def test_rct_alarms_exactly_when_a_run_reaches_the_cutoff() -> None:
    rct = RepetitionCountTest(2**-4, 1.0, "binary")  # C = 5
    samples = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
    update = rct.update(samples)
    assert rct.cutoff == 5
    assert update.alarm_indices.tolist() == [5]  # the fifth 1 (indices 1..5)
    assert update.evaluations == 4  # runs starting at 0, 1, 7 and 11
    assert update.first_alert is not None
    assert update.first_alert.outcome is Outcome.CRITICAL
    assert update.first_alert.evidence["cutoff"] == 5
    assert update.first_alert.limitation  # every alert states what it cannot establish


def test_rct_tracks_runs_across_buffer_boundaries() -> None:
    rct = RepetitionCountTest(2**-4, 1.0, "binary")
    assert rct.update(np.array([1, 0, 0], dtype=np.uint8)).alarms == 0
    update = rct.update(np.array([0, 0, 0, 0], dtype=np.uint8))  # zeros at 1..6; the fifth is at 5
    assert update.alarm_indices.tolist() == [5]
    assert update.evaluations == 0  # the run started in the previous buffer


# --- Adaptive Proportion Test ---


def test_apt_counts_the_first_value_in_each_window() -> None:
    apt = AdaptiveProportionTest(2**-20, 1.0, "binary")
    assert apt.cutoff == 589  # SP 800-90B Table 2
    window = np.zeros(1024, dtype=np.uint8)
    window[600:] = 1  # first value 0 occurs 600 times; reaches 589 at index 588
    update = apt.update(np.concatenate((window, _bits(4, 1024))))
    assert update.alarm_indices.tolist() == [588]
    assert update.evaluations == 2


def test_apt_window_spanning_buffers_alarms_at_the_same_index() -> None:
    window = np.zeros(1024, dtype=np.uint8)
    window[700:] = 1
    split = AdaptiveProportionTest(2**-20, 1.0, "binary")
    assert split.update(window[:300]).alarms == 0
    assert split.update(window[300:]).alarm_indices.tolist() == [588]


# --- Repeated-block detector ---


def test_repeated_block_detects_a_replay_at_an_unaligned_distance() -> None:
    detector = RepeatedBlockDetector(ALPHA, 1.0, "binary", history_samples=4096)
    k = detector.block
    samples = _bits(5, 8000)
    samples[5000:5600] = samples[5000 - 1234 : 5600 - 1234]  # 1234 is not a multiple of k
    _, alarms = _feed(detector, samples, 1000)
    first_full_block = -(-5000 // k) * k
    assert alarms[0] == first_full_block + k - 1
    assert all(5000 <= a < 5600 for a in alarms)


def test_repeated_block_ignores_replays_beyond_the_history() -> None:
    detector = RepeatedBlockDetector(ALPHA, 1.0, "binary", history_samples=1024)
    samples = _bits(6, 8000)
    samples[6000:6600] = samples[1000:1600]  # distance 5000 > history 1024
    assert _feed(detector, samples, 512)[1] == []


def test_repeated_block_works_on_byte_sources() -> None:
    detector = RepeatedBlockDetector(ALPHA, 8.0, "byte", history_samples=4096)
    data = np.random.default_rng(7).integers(0, 256, 6000, dtype=np.uint8)
    data[4000:4100] = data[3000:3100]
    alarms = _feed(detector, data, 777)[1]
    assert alarms
    assert 4000 <= alarms[0] < 4100


# --- Windowed IID detectors ---


def test_frequency_detector_flags_a_biased_window_at_its_last_sample() -> None:
    detector = FrequencyDetector(ALPHA, 0.5, 1024)
    biased = _bits(8, 1024, p=0.8)
    update = detector.update(np.concatenate((_bits(9, 1024), biased)))
    assert update.alarm_indices.tolist() == [2047]
    assert update.first_alert is not None
    assert update.first_alert.evidence["ones"] > detector.region.high


def test_runs_detector_flags_oscillation_and_stickiness() -> None:
    detector = RunsDetector(ALPHA, 0.5, 1024)
    oscillating = np.tile(np.array([0, 1], dtype=np.uint8), 512)
    sticky = np.repeat(_bits(10, 64), 16)
    assert detector.update(np.concatenate((oscillating, sticky))).alarms == 2


def test_lag_agreement_identifies_an_offending_lag() -> None:
    detector = LagAgreementDetector(ALPHA, 0.5, 1024, max_lag=8)
    pattern = np.tile(_bits(11, 3), 342)[:1024]  # period 3: lag-3 pairs always agree
    update = detector.update(pattern)
    assert update.alarms == 1
    assert update.first_alert is not None
    evidence = update.first_alert.evidence
    region = detector.regions[int(evidence["lag"])]
    assert not region.low <= int(evidence["agreements"]) <= region.high


def test_autocorrelation_warns_on_correlated_input() -> None:
    detector = AutocorrelationDetector(1e-3, 1024, max_lag=8)
    sticky = np.repeat(_bits(12, 256), 4)
    update = detector.update(sticky)
    assert update.alarms == 1
    assert update.first_alert is not None
    assert update.first_alert.outcome is Outcome.WARNING


def test_autocorrelation_skips_constant_windows() -> None:
    detector = AutocorrelationDetector(1e-3, 1024)
    update = detector.update(np.zeros(1024, dtype=np.uint8))
    assert (update.evaluations, update.alarms) == (1, 0)


def test_byte_chi_square_warns_on_non_uniform_bytes() -> None:
    detector = ByteChiSquareDetector(1e-3, 4096)
    skewed = np.random.default_rng(13).integers(0, 128, 4096, dtype=np.uint8)
    assert detector.update(skewed).alarms == 1


# --- Properties shared by all detectors ---


@pytest.mark.parametrize("factory", range(7))
def test_results_do_not_depend_on_buffer_boundaries(factory: int) -> None:
    samples = _bits(14, 30_000)
    samples[12_000:12_800] = 1  # a fault so that some detectors alarm
    samples[20_000:21_000] = samples[17_000:18_000]
    reference = _feed(_all_detectors()[factory], samples, 30_000)
    for buffer in (1, 37, 1000, 4096):
        if buffer == 1 and factory == 2:
            continue  # one-sample buffers make the repeated-block detector needlessly slow
        assert _feed(_all_detectors()[factory], samples, buffer) == reference


@pytest.mark.parametrize("factory", range(7))
def test_reset_restarts_the_stream(factory: int) -> None:
    detector = _all_detectors()[factory]
    samples = _bits(15, 9000)
    samples[3000:4000] = 0
    first = _feed(detector, samples, 2048)
    detector.reset()
    assert detector.position == 0
    assert _feed(detector, samples, 2048) == first


@pytest.mark.parametrize(("p", "h"), [(0.5, 1.0), (0.55, H_S1)])
def test_no_critical_alarms_on_a_fixed_healthy_fixture(p: float, h: float) -> None:
    samples = _bits(16, 1 << 20, p)
    critical = [
        RepetitionCountTest(ALPHA, h, "binary"),
        AdaptiveProportionTest(ALPHA, h, "binary"),
        RepeatedBlockDetector(ALPHA, h, "binary"),
        FrequencyDetector(ALPHA, p),
        RunsDetector(ALPHA, p),
        LagAgreementDetector(ALPHA, p),
    ]
    for detector in critical:
        evaluations, alarms = _feed(detector, samples, 1 << 16)
        assert evaluations > 0
        assert alarms == [], detector.spec.name


@pytest.mark.parametrize("name", ["rct", "frequency", "runs", "lag-agreement"])
def test_alarm_rates_match_exact_null_probabilities_at_relaxed_alpha(name: str) -> None:
    # Calibration at alpha = 2^-6: observed alarms per evaluation must be consistent with the
    # exact null probability (two-sided binomial test). The seed is fixed, so this is deterministic.
    alpha, p = 2**-6, 0.55
    detector: Detector = {
        "rct": lambda: RepetitionCountTest(alpha, -math.log2(p), "binary"),
        "frequency": lambda: FrequencyDetector(alpha, p, 1024),
        "runs": lambda: RunsDetector(alpha, p, 1024),
        "lag-agreement": lambda: LagAgreementDetector(alpha, p, 1024),
    }[name]()
    samples = _bits(17, 4_000_000, p)
    evaluations, alarms = _feed(detector, samples, 1 << 18)
    if name == "rct":
        # Runs alternate between values, so half are runs of ones; a run of value v reaches the
        # cutoff C with probability P(v)^(C - 1).
        c = detector.cutoff  # type: ignore[attr-defined]
        expected = 0.5 * (p ** (c - 1) + (1 - p) ** (c - 1))
    else:
        expected = detector.spec.false_alarm_probability or 0.0
    assert stats.binomtest(len(alarms), evaluations, expected).pvalue > 1e-3


def test_apt_alarm_rate_matches_the_exact_probability_not_the_cutoff_formula() -> None:
    # Uniform bytes, H = 8, W = 512, alpha = 2^-8. The cutoff formula P(Bin(512, 1/256) >= C)
    # predicts about 1.1e-3 alarms per window; counting the first sample, as SP 800-90B specifies,
    # gives the exact 4.4e-3. The observed rate must agree with the exact value.
    detector = AdaptiveProportionTest(2**-8, 8.0, "byte")
    cutoff = detector.cutoff
    formula = stats.binom.sf(cutoff - 1, 512, 1 / 256)
    exact = detector.spec.false_alarm_probability or 0.0
    assert exact > 3 * formula
    data = np.random.default_rng(19).integers(0, 256, 20_000 * 512, dtype=np.uint8)
    evaluations, alarms = _feed(detector, data, 1 << 20)
    assert evaluations == 20_000
    assert stats.binomtest(len(alarms), evaluations, exact).pvalue > 1e-3
    assert stats.binomtest(len(alarms), evaluations, formula).pvalue < 1e-6


# --- Validation ---


def test_detectors_reject_invalid_input() -> None:
    detector = RepetitionCountTest(ALPHA, 1.0, "binary")
    with pytest.raises(TypeError, match="uint8"):
        detector.update(np.zeros(10, dtype=np.int64))
    with pytest.raises(TypeError, match="uint8"):
        detector.update([0, 1, 0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="0 or 1"):
        detector.update(np.array([0, 2], dtype=np.uint8))
    with pytest.raises(ValueError, match="twice"):
        LagAgreementDetector(ALPHA, 0.5, 8, max_lag=8)


def test_empty_buffers_are_harmless() -> None:
    for detector in _all_detectors():
        update = detector.update(np.zeros(0, dtype=np.uint8))
        assert (update.evaluations, update.alarms) == (0, 0)
