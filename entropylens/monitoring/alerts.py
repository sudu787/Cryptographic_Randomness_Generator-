"""Alert, status and detector-specification types shared by the monitor and its detectors.

Every alert carries three things: the evidence (what was measured, against which threshold), the
explanation (why that is unlikely under the nominal model), and the limitation (what the alert does
**not** establish). The dashboard displays all three.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class DetectorLevel(StrEnum):
    """Whether a detector may close the output gate (critical) or only warn."""

    CRITICAL = "critical"
    WARNING = "warning"


class Outcome(StrEnum):
    """Result of one detector over the most recent input."""

    PASS = "pass"  # noqa: S105  (an outcome label, not a password)
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"
    INSUFFICIENT_DATA = "insufficient-data"


class SourceStatus(StrEnum):
    """Aggregate health of the monitored source (docs/methodology.md section 4.5)."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @property
    def gate_open(self) -> bool:
        """Output may be released only while the source is healthy or merely degraded."""
        return self in (SourceStatus.HEALTHY, SourceStatus.DEGRADED)


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """Static, documented description of a detector.

    Attributes:
        name: Stable identifier used in results and configuration.
        title: Human-readable name.
        level: Critical detectors can fail the source; warning detectors are exploratory.
        null_model: The nominal model under which the threshold is derived.
        statistic: What is measured, and over which evaluation unit.
        alpha: False-alarm budget per evaluation (``None`` for non-statistical detectors).
        threshold: The threshold with its derivation.
        targets: Fault families (docs/methodology.md section 3) the detector is designed for.
        limitation: What a pass, or an alarm, does not establish.
        false_alarm_probability: Worst-case alarm probability per evaluation under the null model,
            computed exactly where possible (a bound for union-bound thresholds). It can differ
            from ``alpha``; the APT's can exceed it (see ``thresholds.apt_cutoff``).
    """

    name: str
    title: str
    level: DetectorLevel
    null_model: str
    statistic: str
    alpha: float | None
    threshold: str
    targets: tuple[str, ...]
    limitation: str
    false_alarm_probability: float | None = None


@dataclass(frozen=True, slots=True)
class Alert:
    """One explained detector event.

    Attributes:
        detector: Name of the detector that raised the alert.
        outcome: ``CRITICAL``, ``WARNING`` or ``ERROR``.
        sample_index: Absolute index of the sample at which the event was raised (``-1`` if the
            event is not tied to a sample, such as a detector error).
        message: Plain-language explanation of what was observed and why it is unexpected.
        evidence: Measured values and thresholds, safe to log (never raw sample data).
        limitation: What this alert does not establish.
    """

    detector: str
    outcome: Outcome
    sample_index: int
    message: str
    evidence: Mapping[str, float | int | str] = field(default_factory=dict)
    limitation: str = ""
