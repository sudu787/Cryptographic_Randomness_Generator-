"""Tests for the health monitor: detector assembly, fail-closed status semantics, latching."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from entropylens.experiments.synthetic_sources import S0, S0_BYTE, S2
from entropylens.monitoring.alerts import (
    DetectorLevel,
    DetectorSpec,
    Outcome,
    SourceStatus,
)
from entropylens.monitoring.detectors import Detector, DetectorUpdate, RepetitionCountTest
from entropylens.monitoring.samples import Samples
from entropylens.monitoring.source_health import HealthMonitor, MonitorSettings, build_detectors

CRITICAL_ONLY = MonitorSettings(
    detectors=("rct", "apt", "repeated-block", "frequency", "runs", "lag-agreement")
)


def _healthy(n: int, seed: int = 0) -> Samples:
    return S0.generate(np.random.default_rng(seed), n)


class _Exploding(Detector):
    """A detector whose evaluation always fails."""

    def __init__(self) -> None:
        super().__init__(
            DetectorSpec("exploding", "Exploding", DetectorLevel.CRITICAL, "", "", 0.1, "", (), ""),
            "binary",
        )

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        raise RuntimeError("simulated detector bug")

    def _reset_state(self) -> None:
        pass


class _AlwaysWarns(Detector):
    """A warning-level detector that warns on every buffer."""

    def __init__(self) -> None:
        super().__init__(
            DetectorSpec("warns", "Warns", DetectorLevel.WARNING, "", "", 0.1, "", (), ""),
            "binary",
        )

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        return DetectorUpdate(
            1, np.array([offset], dtype=np.int64), self._alert(offset, "always warns", {})
        )

    def _reset_state(self) -> None:
        pass


# --- Assembly ---


def test_binary_iid_model_gets_every_binary_detector() -> None:
    detectors, disabled = build_detectors(S0)
    assert [d.spec.name for d in detectors] == [
        "rct",
        "apt",
        "repeated-block",
        "frequency",
        "runs",
        "lag-agreement",
        "autocorrelation",
    ]
    assert disabled == {"byte-chi-square": "applies only to byte sources"}


def test_non_iid_model_disables_detectors_without_exact_nulls() -> None:
    detectors, disabled = build_detectors(S2)
    assert [d.spec.name for d in detectors] == ["rct", "apt", "repeated-block"]
    for name in ("frequency", "runs", "lag-agreement", "autocorrelation"):
        assert "IID" in disabled[name]


def test_byte_model_gets_byte_detectors() -> None:
    detectors, disabled = build_detectors(S0_BYTE)
    assert [d.spec.name for d in detectors] == ["rct", "apt", "repeated-block", "byte-chi-square"]
    assert disabled["frequency"] == "applies only to binary sources"


def test_configuration_can_restrict_detectors() -> None:
    detectors, disabled = build_detectors(S0, MonitorSettings(detectors=("rct",)))
    assert [d.spec.name for d in detectors] == ["rct"]
    assert disabled["apt"] == "disabled by configuration"


def test_thresholds_follow_the_configured_alpha() -> None:
    detectors, _ = build_detectors(S0, MonitorSettings(alpha_log2=-20))
    rct = detectors[0]
    assert isinstance(rct, RepetitionCountTest)
    assert rct.cutoff == 21
    assert rct.spec.alpha == 2**-20


def test_settings_are_validated_strictly() -> None:
    assert MonitorSettings().alpha == 2**-30
    for bad in ({"alpha_log2": 0}, {"alpha_log2": -30.0}, {"detectors": ["mystery"]}, {"x": 1}):
        with pytest.raises(ValidationError):
            MonitorSettings.model_validate(bad)


# --- Status semantics ---


def test_startup_is_unknown_until_every_critical_detector_has_evaluated() -> None:
    monitor = HealthMonitor(S0, CRITICAL_ONLY)
    first = monitor.process(_healthy(1000))
    assert first.status is SourceStatus.UNKNOWN
    assert not first.gate_open
    assert any(state.outcome is Outcome.INSUFFICIENT_DATA for state in first.detectors)
    assert monitor.process(_healthy(8192, seed=1)).status is SourceStatus.HEALTHY


def test_fault_fails_the_source_and_the_failure_latches() -> None:
    monitor = HealthMonitor(S0, CRITICAL_ONLY)
    assert monitor.process(_healthy(8192)).status is SourceStatus.HEALTHY
    stuck = monitor.process(np.zeros(100, dtype=np.uint8))
    assert stuck.status is SourceStatus.FAILED
    assert not stuck.gate_open
    assert stuck.failure is not None
    assert stuck.failure.detector == "rct"
    # A run of C = 31 zeros completes by index 8192 + 30, sooner if the healthy data ended in zeros.
    assert 8192 <= stuck.failure.sample_index <= 8192 + 30
    assert stuck.failure.limitation

    later = monitor.process(_healthy(8192, seed=2))
    assert later.status is SourceStatus.FAILED  # latched despite healthy data
    assert later.failure == stuck.failure

    monitor.reset()
    assert monitor.status is SourceStatus.UNKNOWN
    assert monitor.process(_healthy(8192, seed=3)).status is SourceStatus.HEALTHY


def test_detector_exception_fails_closed() -> None:
    monitor = HealthMonitor(
        S0, detectors=[RepetitionCountTest(2**-30, 1.0, "binary"), _Exploding()]
    )
    report = monitor.process(_healthy(5000))
    assert report.status is SourceStatus.UNKNOWN
    assert not report.gate_open
    assert any(a.outcome is Outcome.ERROR and "simulated" in a.message for a in report.alerts)
    # The error latches: healthy-looking data never turns the status into HEALTHY.
    assert monitor.process(_healthy(5000, seed=1)).status is SourceStatus.UNKNOWN


@pytest.mark.parametrize(
    "bad_input",
    [
        np.zeros(10, dtype=np.int32),
        np.array([0, 1, 2], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        [0, 1, 1],
    ],
)
def test_invalid_input_fails_closed(bad_input: object) -> None:
    monitor = HealthMonitor(S0, CRITICAL_ONLY)
    monitor.process(_healthy(8192))
    report = monitor.process(bad_input)
    assert report.status is SourceStatus.UNKNOWN
    assert report.alerts[0].outcome is Outcome.ERROR


def test_warnings_degrade_but_keep_the_gate_open() -> None:
    monitor = HealthMonitor(
        S0, detectors=[RepetitionCountTest(2**-30, 1.0, "binary"), _AlwaysWarns()]
    )
    report = monitor.process(_healthy(2000))
    assert report.status is SourceStatus.DEGRADED
    assert report.gate_open


def test_source_error_fails_the_source() -> None:
    monitor = HealthMonitor(S0, CRITICAL_ONLY)
    monitor.process(_healthy(8192))
    report = monitor.report_source_error("exception", "device unplugged")
    assert report.status is SourceStatus.FAILED
    assert report.failure is not None
    assert report.failure.detector == "availability"
    assert report.failure.sample_index == 8192
    assert "device unplugged" in report.failure.message


def test_monitor_needs_a_critical_detector() -> None:
    with pytest.raises(ValueError, match="critical"):
        HealthMonitor(S0, detectors=[_AlwaysWarns()])


def test_status_gate_semantics() -> None:
    assert SourceStatus.HEALTHY.gate_open
    assert SourceStatus.DEGRADED.gate_open
    assert not SourceStatus.FAILED.gate_open
    assert not SourceStatus.UNKNOWN.gate_open
