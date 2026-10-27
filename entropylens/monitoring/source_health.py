"""The source-health monitor: detector assembly, fail-closed aggregation, output gate, latching.

Status semantics (docs/methodology.md section 4.5):

========== ================================================================== ===========
Status     Condition                                                          Output gate
========== ================================================================== ===========
HEALTHY    every enabled critical detector evaluated at least once; no alarm   open
DEGRADED   as HEALTHY, with a warning in the latest buffer                     open
FAILED     any critical alarm or source-availability failure (latched)         closed
UNKNOWN    a detector error, invalid input, or a critical detector without     closed
           any completed evaluation yet (start-up)
========== ================================================================== ===========

No code path maps an exception to HEALTHY (acceptance criterion AC3). ``FAILED`` and errors stay
latched until :meth:`HealthMonitor.reset` is called explicitly, mirroring SP 800-90B's error state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from entropylens.monitoring.alerts import Alert, DetectorLevel, Outcome, SourceStatus
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
    availability_alert,
)
from entropylens.monitoring.samples import SampleKind, validate_samples


class NominalModel(Protocol):
    """What the monitor needs to know about the source it is calibrated to."""

    @property
    def kind(self) -> SampleKind:
        """Sample kind."""
        ...

    @property
    def claimed_entropy(self) -> float:
        """Claimed min-entropy per sample."""
        ...

    @property
    def p_one(self) -> float:
        """Nominal probability of a 1 (binary sources)."""
        ...

    @property
    def is_iid(self) -> bool:
        """Whether samples are nominally independent."""
        ...


DetectorName = Literal[
    "rct",
    "apt",
    "repeated-block",
    "frequency",
    "runs",
    "lag-agreement",
    "autocorrelation",
    "byte-chi-square",
]
BINARY_IID_ONLY = frozenset({"frequency", "runs", "lag-agreement", "autocorrelation"})


class MonitorSettings(BaseModel):
    """Configuration of the monitor. Defaults are the pre-declared operating parameters.

    ``alpha_log2`` stores the per-evaluation false-alarm budget as a power of two (default
    ``2^-30``, docs/methodology.md section 4.4), so configurations state it exactly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    alpha_log2: Annotated[int, Field(ge=-64, le=-1)] = -30
    warning_alpha: Annotated[float, Field(gt=0, lt=1)] = 1e-3
    history_samples: Annotated[int, Field(ge=64, le=1 << 22)] = 1 << 16
    window_samples: Annotated[int, Field(ge=256, le=1 << 16)] = 4096
    max_lag: Annotated[int, Field(ge=1, le=32)] = 8
    autocorrelation_max_lag: Annotated[int, Field(ge=1, le=256)] = 64
    detectors: tuple[DetectorName, ...] | None = None

    @property
    def alpha(self) -> float:
        """Per-evaluation false-alarm budget of each critical detector."""
        return 2.0**self.alpha_log2


def build_detectors(
    model: NominalModel, settings: MonitorSettings | None = None
) -> tuple[list[Detector], dict[str, str]]:
    """Create the applicable detectors for ``model``.

    Returns:
        The detectors, and a mapping from each inapplicable or disabled detector name to the
        reason. Disabled detectors are recorded in run metadata; they never silently pass.
    """
    settings = settings or MonitorSettings()
    alpha, h, kind = settings.alpha, model.claimed_entropy, model.kind
    wanted = set(settings.detectors) if settings.detectors is not None else None
    detectors: list[Detector] = []
    disabled: dict[str, str] = {}

    def add(name: str, factory: Callable[..., Detector], *args: object) -> None:
        if wanted is not None and name not in wanted:
            disabled[name] = "disabled by configuration"
        elif name in BINARY_IID_ONLY and kind != "binary":
            disabled[name] = "applies only to binary sources"
        elif name in BINARY_IID_ONLY and not model.is_iid:
            disabled[name] = "exact null distribution implemented only for IID models"
        elif name == "byte-chi-square" and kind != "byte":
            disabled[name] = "applies only to byte sources"
        else:
            detectors.append(factory(*args))

    window, p0 = settings.window_samples, model.p_one if kind == "binary" else 0.5
    add("rct", RepetitionCountTest, alpha, h, kind)
    add("apt", AdaptiveProportionTest, alpha, h, kind)
    add("repeated-block", RepeatedBlockDetector, alpha, h, kind, settings.history_samples)
    add("frequency", FrequencyDetector, alpha, p0, window)
    add("runs", RunsDetector, alpha, p0, window)
    add("lag-agreement", LagAgreementDetector, alpha, p0, window, settings.max_lag)
    add(
        "autocorrelation",
        AutocorrelationDetector,
        settings.warning_alpha,
        window,
        settings.autocorrelation_max_lag,
    )
    add("byte-chi-square", ByteChiSquareDetector, settings.warning_alpha, window)
    return detectors, disabled


@dataclass(frozen=True, slots=True)
class DetectorState:
    """Per-detector summary since the last reset, plus the outcome for the latest buffer."""

    name: str
    level: DetectorLevel
    outcome: Outcome
    evaluations: int
    alarms: int


@dataclass(frozen=True, slots=True)
class MonitorReport:
    """Result of feeding one buffer (or one source event) to the monitor."""

    status: SourceStatus
    samples_processed: int
    alerts: tuple[Alert, ...]
    detectors: tuple[DetectorState, ...]
    failure: Alert | None

    @property
    def gate_open(self) -> bool:
        """Whether output derived from this buffer may be released."""
        return self.status.gate_open


class HealthMonitor:
    """Streaming, fail-closed health monitor for one simulated noise source."""

    def __init__(
        self,
        model: NominalModel,
        settings: MonitorSettings | None = None,
        *,
        detectors: Sequence[Detector] | None = None,
    ) -> None:
        if detectors is None:
            built, self.disabled = build_detectors(model, settings)
            self._detectors = list(built)
        else:
            self._detectors, self.disabled = list(detectors), {}
        if not any(d.spec.level is DetectorLevel.CRITICAL for d in self._detectors):
            raise ValueError("a monitor needs at least one critical detector")
        self.model = model
        self._evaluations = {d.spec.name: 0 for d in self._detectors}
        self._alarms = {d.spec.name: 0 for d in self._detectors}
        self._processed = 0
        self._failure: Alert | None = None
        self._error: Alert | None = None
        self._status = SourceStatus.UNKNOWN

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The active detectors."""
        return tuple(self._detectors)

    @property
    def status(self) -> SourceStatus:
        """Current aggregate status."""
        return self._status

    @property
    def failure(self) -> Alert | None:
        """The first latched critical alert since the last reset, if any."""
        return self._failure

    def process(self, samples: object) -> MonitorReport:
        """Feed the next buffer of raw samples and return the updated status."""
        try:
            buffer = validate_samples(samples, self.model.kind)
        except (TypeError, ValueError) as exc:
            self._error = self._error or Alert(
                "monitor", Outcome.ERROR, -1, f"invalid input: {exc}"
            )
            return self._report((self._error,), {})

        alerts: list[Alert] = []
        outcomes: dict[str, Outcome] = {}
        for detector in self._detectors:
            name = detector.spec.name
            try:
                update = detector.update(buffer)
            except Exception as exc:  # fail closed: any detector fault means UNKNOWN, never HEALTHY
                error = Alert(name, Outcome.ERROR, -1, f"detector failed: {exc!r}")
                self._error = self._error or error
                alerts.append(error)
                outcomes[name] = Outcome.ERROR
                continue
            self._evaluations[name] += update.evaluations
            self._alarms[name] += update.alarms
            if update.first_alert is not None:
                alerts.append(update.first_alert)
                outcomes[name] = update.first_alert.outcome
            elif self._evaluations[name] == 0:
                outcomes[name] = Outcome.INSUFFICIENT_DATA
            else:
                outcomes[name] = Outcome.PASS

        critical = [a for a in alerts if a.outcome is Outcome.CRITICAL]
        if critical and self._failure is None:
            self._failure = min(critical, key=lambda a: a.sample_index)
        self._processed += buffer.size
        return self._report(tuple(alerts), outcomes)

    def report_source_error(self, mode: str, detail: str = "") -> MonitorReport:
        """Record a source-availability failure (exception, short read, timeout)."""
        alert = availability_alert(self._processed, mode, detail)
        self._failure = self._failure or alert
        return self._report((alert,), {})

    def reset(self) -> None:
        """Explicitly clear latched failures and errors and restart every detector."""
        for detector in self._detectors:
            detector.reset()
        self._evaluations = dict.fromkeys(self._evaluations, 0)
        self._alarms = dict.fromkeys(self._alarms, 0)
        self._processed = 0
        self._failure = self._error = None
        self._status = SourceStatus.UNKNOWN

    def _report(self, alerts: tuple[Alert, ...], outcomes: dict[str, Outcome]) -> MonitorReport:
        critical = [d for d in self._detectors if d.spec.level is DetectorLevel.CRITICAL]
        if self._failure is not None:
            status = SourceStatus.FAILED
        elif self._error is not None or any(self._evaluations[d.spec.name] == 0 for d in critical):
            status = SourceStatus.UNKNOWN
        elif any(outcome is Outcome.WARNING for outcome in outcomes.values()):
            status = SourceStatus.DEGRADED
        else:
            status = SourceStatus.HEALTHY
        self._status = status
        states = tuple(
            DetectorState(
                d.spec.name,
                d.spec.level,
                outcomes.get(d.spec.name, Outcome.INSUFFICIENT_DATA),
                self._evaluations[d.spec.name],
                self._alarms[d.spec.name],
            )
            for d in self._detectors
        )
        return MonitorReport(status, self._processed, alerts, states, self._failure)
