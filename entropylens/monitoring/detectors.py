"""Streaming source-health detectors (docs/methodology.md section 4).

Each detector consumes consecutive buffers of raw samples and reports, for every buffer:

* ``evaluations``: how many evaluation units completed (runs, windows or blocks). Each unit has
  false-alarm probability at most ``alpha`` under the nominal model.
* ``alarm_indices``: the absolute stream index at which each alarm was raised, at most one per
  evaluation unit.
* ``first_alert``: a fully explained :class:`Alert` for the first alarm in the buffer.

Critical detectors use exact null distributions or rigorous bounds (see ``thresholds.py``). Warning
detectors use approximations and are labelled exploratory.

Deviations from the SP 800-90B pseudo-code, which affect alarm *counting* but never the time of the
first alarm:

* The RCT signals once per run, when the run first reaches C; the standard signals again at every
  further repetition.
* The APT signals once per window, when the count first reaches C.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from entropylens.monitoring.alerts import Alert, DetectorLevel, DetectorSpec, Outcome
from entropylens.monitoring.samples import BITS_PER_SAMPLE, SampleKind, Samples, validate_samples
from entropylens.monitoring.thresholds import (
    AcceptanceRegion,
    apt_cutoff,
    binomial_region,
    collision_block_samples,
    rct_cutoff,
    transition_region,
    validate_alpha,
)

IndexArray = npt.NDArray[np.int64]
Evidence = dict[str, float | int | str]
_NO_ALARMS: IndexArray = np.zeros(0, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class DetectorUpdate:
    """What one detector observed in one buffer."""

    evaluations: int
    alarm_indices: IndexArray
    first_alert: Alert | None = None

    @property
    def alarms(self) -> int:
        """Number of alarms raised in the buffer."""
        return int(self.alarm_indices.size)


class Detector(ABC):
    """Base class for streaming detectors. Indices are absolute positions in the stream."""

    def __init__(self, spec: DetectorSpec, kind: SampleKind) -> None:
        self.spec = spec
        self.kind = kind
        self._position = 0

    @property
    def position(self) -> int:
        """Number of samples consumed since construction or the last reset."""
        return self._position

    def update(self, samples: Samples) -> DetectorUpdate:
        """Process the next buffer of the stream."""
        validate_samples(samples, self.kind)
        offset = self._position
        result = self._process(samples, offset) if samples.size else DetectorUpdate(0, _NO_ALARMS)
        self._position += samples.size
        return result

    def reset(self) -> None:
        """Forget all state, as if the stream started again."""
        self._position = 0
        self._reset_state()

    @abstractmethod
    def _process(self, samples: Samples, offset: int) -> DetectorUpdate: ...

    @abstractmethod
    def _reset_state(self) -> None: ...

    def _alert(self, index: int, message: str, evidence: Evidence) -> Alert:
        outcome = Outcome.CRITICAL if self.spec.level is DetectorLevel.CRITICAL else Outcome.WARNING
        return Alert(self.spec.name, outcome, index, message, evidence, self.spec.limitation)


def _alpha_text(alpha: float) -> str:
    exponent = math.log2(alpha)
    return f"2^{exponent:.0f}" if exponent.is_integer() else f"{alpha:.3g}"


# --- SP 800-90B section 4.4 tests ---


class RepetitionCountTest(Detector):
    """SP 800-90B Repetition Count Test. The evaluation unit is one run of identical samples."""

    def __init__(self, alpha: float, h: float, kind: SampleKind) -> None:
        cutoff = rct_cutoff(alpha, h, BITS_PER_SAMPLE[kind])
        super().__init__(
            DetectorSpec(
                name="rct",
                title="Repetition Count Test (SP 800-90B 4.4.1)",
                level=DetectorLevel.CRITICAL,
                null_model=f"samples with min-entropy at least H = {h:.4f} bits each",
                statistic="length of each run of identical consecutive samples",
                alpha=alpha,
                threshold=cutoff.derivation,
                targets=("F1", "F2", "F10"),
                limitation=(
                    "Detects only near-total failures such as a stuck source. Bias, correlation, "
                    "patterns and predictability that keep runs short pass unnoticed."
                ),
                false_alarm_probability=cutoff.false_alarm_probability,
            ),
            kind,
        )
        self.cutoff = cutoff.value
        self._h = h
        self._last = -1
        self._run = 0

    def _reset_state(self) -> None:
        self._last, self._run = -1, 0

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        n = samples.size
        boundaries = np.flatnonzero(samples[1:] != samples[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        lengths = np.diff(np.concatenate((starts, [n])))
        prior = np.zeros(starts.size, dtype=np.int64)
        continues = int(samples[0]) == self._last
        if continues:
            prior[0] = self._run
        totals = prior + lengths
        hit = (prior < self.cutoff) & (totals >= self.cutoff)
        indices = (offset + starts[hit] + self.cutoff - prior[hit] - 1).astype(np.int64)
        self._last, self._run = int(samples[-1]), int(totals[-1])

        first = None
        if indices.size:
            run = int(totals[hit][0])
            value = int(samples[starts[hit][0]])
            first = self._alert(
                int(indices[0]),
                f"{self.cutoff} identical samples (value {value}) in a row, reaching the cutoff "
                f"C = {self.cutoff}. With H = {self._h:.4f} bits per sample, a run this long has "
                f"probability at most {_alpha_text(self.spec.alpha or 0)}.",
                {"run_length_observed": run, "cutoff": self.cutoff, "value": value},
            )
        return DetectorUpdate(int(starts.size - continues), indices, first)


class AdaptiveProportionTest(Detector):
    """SP 800-90B Adaptive Proportion Test over non-overlapping windows of W samples."""

    def __init__(self, alpha: float, h: float, kind: SampleKind, window: int | None = None) -> None:
        window = window or (1024 if kind == "binary" else 512)
        cutoff = apt_cutoff(alpha, h, window, BITS_PER_SAMPLE[kind])
        super().__init__(
            DetectorSpec(
                name="apt",
                title="Adaptive Proportion Test (SP 800-90B 4.4.2)",
                level=DetectorLevel.CRITICAL,
                null_model=f"samples with min-entropy at least H = {h:.4f} bits each",
                statistic=f"occurrences of each window's first value in the window (W={window})",
                alpha=alpha,
                threshold=cutoff.derivation,
                targets=("F1", "F3", "F7", "F10"),
                limitation=(
                    "Detects one value becoming far too common. Patterns, replays and correlation "
                    "that keep value frequencies balanced pass unnoticed."
                ),
                false_alarm_probability=cutoff.false_alarm_probability,
            ),
            kind,
        )
        self.cutoff = cutoff.value
        self.window = window
        self._reference = 0
        self._count = 0
        self._filled = 0
        self._alarmed = False

    def _reset_state(self) -> None:
        self._reference = self._count = self._filled = 0
        self._alarmed = False

    def _scan(self, segment: Samples, offset: int) -> int | None:
        """Continue the current window over ``segment``; return an alarm index if one is raised."""
        counts = self._count + np.cumsum(segment == self._reference)
        self._count = int(counts[-1])
        self._filled += segment.size
        if self._alarmed:
            return None
        reached = np.flatnonzero(counts >= self.cutoff)
        if reached.size:
            self._alarmed = True
            return offset + int(reached[0])
        return None

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        window, n = self.window, samples.size
        alarms: list[int] = []
        counts_at_alarm: list[int] = []
        evaluations = 0
        i = 0
        if self._filled:  # finish the window carried over from the previous buffer
            take = min(window - self._filled, n)
            if (hit := self._scan(samples[:take], offset)) is not None:
                alarms.append(hit)
                counts_at_alarm.append(self.cutoff)
            i = take
            if self._filled == window:
                evaluations += 1
                self._filled = 0

        full = (n - i) // window
        if full:
            blocks = samples[i : i + full * window].reshape(full, window)
            counts = np.cumsum(blocks == blocks[:, :1], axis=1)
            rows = np.flatnonzero(counts[:, -1] >= self.cutoff)
            first_reach = np.argmax(counts[rows] >= self.cutoff, axis=1)
            alarms.extend((offset + i + rows * window + first_reach).tolist())
            counts_at_alarm.extend(counts[rows, -1].tolist())
            evaluations += full
            i += full * window

        if i < n:  # start a new partial window
            self._reference = int(samples[i])
            self._count, self._filled, self._alarmed = 0, 0, False
            if (hit := self._scan(samples[i:], offset + i)) is not None:
                alarms.append(hit)
                counts_at_alarm.append(self.cutoff)

        first = None
        if alarms:
            first = self._alert(
                alarms[0],
                f"One value occurred at least {self.cutoff} times within a window of "
                f"{window} samples (cutoff C = {self.cutoff}). Under the nominal model this has "
                f"probability at most {_alpha_text(self.spec.alpha or 0)} per window.",
                {"count_in_window": counts_at_alarm[0], "cutoff": self.cutoff, "window": window},
            )
        return DetectorUpdate(evaluations, np.asarray(alarms, dtype=np.int64), first)


# --- Repeated-block detector ---


def bit_windows(bits: Samples, width: int) -> npt.NDArray[np.uint64]:
    """Integer value, most significant bit first, of ``bits[i:i+width]`` for every valid ``i``.

    ``bits`` holds one bit per element (0 or 1) and ``width`` is at most 64. The computation packs
    the bits and assembles each 64-bit window from two overlapping big-endian words, so it costs a
    handful of vectorised passes regardless of ``width``.
    """
    if not 1 <= width <= 64:
        raise ValueError("width must be between 1 and 64")
    n = bits.size - width + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    padded = np.concatenate((np.packbits(bits), np.zeros(9, dtype=np.uint8)))
    n_bytes = -(-n // 8)
    words = (
        np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(padded, 8)[:n_bytes])
        .view(">u8")
        .ravel()
        .astype(np.uint64)
    )
    following = padded[8 : 8 + n_bytes].astype(np.uint64)
    out = np.empty(n_bytes * 8, dtype=np.uint64)
    out[0::8] = words
    for shift in range(1, 8):
        out[shift::8] = (words << np.uint64(shift)) | (following >> np.uint64(8 - shift))
    return out[:n] >> np.uint64(64 - width)


def window_keys(samples: Samples, kind: SampleKind, length: int) -> npt.NDArray[np.uint64]:
    """Exact keys of every ``length``-sample window: shape ``(positions, ceil(length*bits/64))``."""
    bits_per_sample = BITS_PER_SAMPLE[kind]
    bits = samples if kind == "binary" else np.unpackbits(samples)
    positions = samples.size - length + 1
    total_bits = length * bits_per_sample
    columns = []
    for start in range(0, total_bits, 64):
        values = bit_windows(bits[start:], min(64, total_bits - start))
        columns.append(values[::bits_per_sample][:positions])
    return np.stack(columns, axis=1)


class RepeatedBlockDetector(Detector):
    """Exact repeat of a k-sample block anywhere in the recent history.

    Each aligned block of ``k`` samples is compared with every ``k``-sample window (at any offset)
    that ends before the block starts, within the last ``history_samples`` positions. ``k`` is the
    smallest length for which the union bound keeps the false-alarm probability per block at most
    ``alpha``.
    """

    def __init__(
        self, alpha: float, h: float, kind: SampleKind, history_samples: int = 1 << 16
    ) -> None:
        block = collision_block_samples(alpha, h, history_samples)
        super().__init__(
            DetectorSpec(
                name="repeated-block",
                title="Repeated-block detector",
                level=DetectorLevel.CRITICAL,
                null_model=(
                    f"independent samples with min-entropy at least H = {h:.4f} bits each "
                    "(per-sample conditional bound for Markov models)"
                ),
                statistic=(
                    f"exact match of each aligned {block.value}-sample block with any earlier "
                    f"window in the last {history_samples} samples"
                ),
                alpha=alpha,
                threshold=block.derivation,
                targets=("F4", "F5", "F8a", "F1"),
                limitation=(
                    "Detects exact repetition within the history window only. Replays from "
                    "further back, near-repeats and fresh low-entropy output pass unnoticed."
                ),
                false_alarm_probability=block.false_alarm_probability,
            ),
            kind,
        )
        self.block = block.value
        self.history = history_samples
        self._h = h
        self._tail: Samples = np.zeros(0, dtype=np.uint8)

    def _reset_state(self) -> None:
        self._tail = np.zeros(0, dtype=np.uint8)

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        k = self.block
        data = np.concatenate((self._tail, samples))
        base = offset - self._tail.size  # absolute index of data[0]
        self._tail = data[-(self.history + k) :].copy()
        end = base + data.size
        first_block = -(-max(offset - k + 1, 0) // k) * k
        block_starts = np.arange(first_block, end - k + 1, k, dtype=np.int64)
        if block_starts.size == 0:
            return DetectorUpdate(0, _NO_ALARMS)

        keys = window_keys(data, self.kind, k)
        if keys.shape[1] == 1:
            _, group = np.unique(keys[:, 0], return_inverse=True)
        else:
            _, group = np.unique(keys, axis=0, return_inverse=True)
        group = group.ravel().astype(np.int64)
        span = np.int64(keys.shape[0] + 1)
        ordered = np.sort(group * span + np.arange(keys.shape[0], dtype=np.int64))

        relative = block_starts - base
        target = group[relative] * span + (relative - k)  # latest allowed earlier window
        found = np.searchsorted(ordered, target, side="right") - 1
        candidate = ordered[np.maximum(found, 0)]
        match = (
            (found >= 0)
            & (candidate // span == group[relative])
            & (candidate % span >= relative - self.history)
        )
        hits = np.flatnonzero(match)
        indices = (block_starts[hits] + k - 1).astype(np.int64)

        first = None
        if hits.size:
            distance = int(relative[hits[0]] - candidate[hits[0]] % span)
            first = self._alert(
                int(indices[0]),
                f"A block of {k} samples ending at sample {int(indices[0])} exactly repeats the "
                f"samples seen {distance} positions earlier. With H = {self._h:.4f} bits per "
                f"sample, any repeat within the last {self.history} positions has probability "
                f"at most {_alpha_text(self.spec.alpha or 0)} per block.",
                {"block_samples": k, "repeat_distance": distance, "history": self.history},
            )
        return DetectorUpdate(int(block_starts.size), indices, first)


# --- Windowed detectors for IID binary models ---


class _WindowedDetector(Detector):
    """Evaluates non-overlapping windows; an alarm is reported at a window's last sample."""

    def __init__(self, spec: DetectorSpec, kind: SampleKind, window: int) -> None:
        super().__init__(spec, kind)
        self.window = window
        self._pending: Samples = np.zeros(0, dtype=np.uint8)

    def _reset_state(self) -> None:
        self._pending = np.zeros(0, dtype=np.uint8)

    @abstractmethod
    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        """Return the alarm flag per window and a function explaining window ``i``."""

    def _process(self, samples: Samples, offset: int) -> DetectorUpdate:
        start = offset - self._pending.size
        data = np.concatenate((self._pending, samples)) if self._pending.size else samples
        count = data.size // self.window
        self._pending = data[count * self.window :].copy()
        if count == 0:
            return DetectorUpdate(0, _NO_ALARMS)
        rows = data[: count * self.window].reshape(count, self.window)
        alarm, explain = self._evaluate(rows)
        hits = np.flatnonzero(alarm)
        indices = (start + (hits + 1) * self.window - 1).astype(np.int64)
        first = None
        if hits.size:
            message, evidence = explain(int(hits[0]))
            first = self._alert(int(indices[0]), message, evidence)
        return DetectorUpdate(count, indices, first)


def _region_text(region: AcceptanceRegion) -> str:
    return f"[{region.low}, {region.high}]"


class FrequencyDetector(_WindowedDetector):
    """Count of ones per window against the exact Binomial(W, p0) null (two-sided)."""

    def __init__(self, alpha: float, p0: float, window: int = 4096) -> None:
        self.region = binomial_region(window, p0, alpha)
        super().__init__(
            DetectorSpec(
                name="frequency",
                title="Windowed frequency (monobit) detector",
                level=DetectorLevel.CRITICAL,
                null_model=f"IID Bernoulli(p0 = {p0:.4f}) samples",
                statistic=f"number of ones in each window of {window} samples",
                alpha=alpha,
                threshold=self.region.derivation,
                targets=("F3", "F7", "F1"),
                limitation=(
                    "Detects a shift in the proportion of ones. Correlation, patterns and "
                    "replays with the nominal proportion pass unnoticed."
                ),
                false_alarm_probability=self.region.false_alarm_probability,
            ),
            "binary",
            window,
        )
        self.p0 = p0

    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        ones = rows.sum(axis=1, dtype=np.int64)
        alarm = (ones < self.region.low) | (ones > self.region.high)

        def explain(i: int) -> tuple[str, Evidence]:
            return (
                f"{int(ones[i])} ones in a window of {self.window} samples; about "
                f"{self.window * self.p0:.0f} are expected. The exact acceptance region "
                f"{_region_text(self.region)} has false-alarm probability "
                f"{self.region.false_alarm_probability:.3g} per window.",
                {"ones": int(ones[i]), "low": self.region.low, "high": self.region.high},
            )

        return alarm, explain


class RunsDetector(_WindowedDetector):
    """Number of value changes per window against its exact IID null distribution (two-sided)."""

    def __init__(self, alpha: float, p0: float, window: int = 4096) -> None:
        self.region = transition_region(window, p0, alpha)
        super().__init__(
            DetectorSpec(
                name="runs",
                title="Windowed runs (transition count) detector",
                level=DetectorLevel.CRITICAL,
                null_model=f"IID Bernoulli(p0 = {p0:.4f}) samples",
                statistic=f"number of value changes in each window of {window} samples",
                alpha=alpha,
                threshold=self.region.derivation,
                targets=("F6", "F4", "F1"),
                limitation=(
                    "Detects too many or too few runs (oscillation or stickiness). Balanced "
                    "patterns and replays with typical run structure pass unnoticed."
                ),
                false_alarm_probability=self.region.false_alarm_probability,
            ),
            "binary",
            window,
        )

    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        changes = np.count_nonzero(rows[:, 1:] != rows[:, :-1], axis=1)
        alarm = (changes < self.region.low) | (changes > self.region.high)

        def explain(i: int) -> tuple[str, Evidence]:
            return (
                f"{int(changes[i])} value changes in a window of {self.window} samples, outside "
                f"the exact acceptance region {_region_text(self.region)}. Too few changes "
                "suggest sticky (correlated) output; too many suggest oscillation.",
                {"transitions": int(changes[i]), "low": self.region.low, "high": self.region.high},
            )

        return alarm, explain


class LagAgreementDetector(_WindowedDetector):
    """Agreement of disjoint sample pairs at lags 1..K, exact binomial nulls, Bonferroni over K."""

    def __init__(self, alpha: float, p0: float, window: int = 4096, max_lag: int = 8) -> None:
        if window < 2 * max_lag:
            raise ValueError("window must be at least twice max_lag")
        agree = p0 * p0 + (1 - p0) * (1 - p0)
        self.lags = tuple(range(1, max_lag + 1))
        self.pairs = {k: (window // (2 * k)) * k for k in self.lags}
        self.regions = {
            k: binomial_region(self.pairs[k], agree, alpha / max_lag) for k in self.lags
        }
        super().__init__(
            DetectorSpec(
                name="lag-agreement",
                title=f"Lag-k agreement detector (k = 1..{max_lag})",
                level=DetectorLevel.CRITICAL,
                null_model=f"IID Bernoulli(p0 = {p0:.4f}) samples",
                statistic=(
                    "agreements between disjoint pairs (x[j], x[j+k]) in each window of "
                    f"{window} samples, for each lag k"
                ),
                alpha=alpha,
                threshold=(
                    f"exact Binomial(pairs, q = p0^2 + (1-p0)^2 = {agree:.4f}) region at "
                    f"alpha/{max_lag} per lag (Bonferroni)"
                ),
                targets=("F6", "F4"),
                limitation=(
                    f"Detects dependence at lags up to {max_lag} only. Longer-range patterns and "
                    "replays pass unnoticed."
                ),
                false_alarm_probability=min(
                    1.0, sum(r.false_alarm_probability for r in self.regions.values())
                ),
            ),
            "binary",
            window,
        )

    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        counts: dict[int, npt.NDArray[np.int64]] = {}
        alarm = np.zeros(rows.shape[0], dtype=bool)
        for k in self.lags:
            blocks = self.window // (2 * k)
            paired = rows[:, : blocks * 2 * k].reshape(rows.shape[0], blocks, 2, k)
            counts[k] = np.count_nonzero(paired[:, :, 0, :] == paired[:, :, 1, :], axis=(1, 2))
            region = self.regions[k]
            alarm |= (counts[k] < region.low) | (counts[k] > region.high)

        def explain(i: int) -> tuple[str, Evidence]:
            lag = next(
                k
                for k in self.lags
                if not self.regions[k].low <= counts[k][i] <= self.regions[k].high
            )
            region = self.regions[lag]
            return (
                f"At lag {lag}, {int(counts[lag][i])} of {self.pairs[lag]} disjoint sample pairs "
                f"agree, outside the exact acceptance region {_region_text(region)}. This "
                "indicates dependence between samples that far apart.",
                {"lag": lag, "agreements": int(counts[lag][i]), "pairs": self.pairs[lag]},
            )

        return alarm, explain


# --- Exploratory (warning-level) detectors ---


class AutocorrelationDetector(_WindowedDetector):
    """EXPLORATORY: sample autocorrelation at lags 1..K with a normal approximation."""

    def __init__(self, warning_alpha: float, window: int = 4096, max_lag: int = 64) -> None:
        warning_alpha = validate_alpha(warning_alpha)
        self.max_lag = max_lag
        self.critical_z = float(stats.norm.isf(warning_alpha / (2 * max_lag)))
        super().__init__(
            DetectorSpec(
                name="autocorrelation",
                title="Autocorrelation (exploratory)",
                level=DetectorLevel.WARNING,
                null_model="IID samples; r_k approximately Normal(0, 1/W)",
                statistic=f"max |r_k| * sqrt(W) over lags 1..{max_lag}, windows of {window}",
                alpha=warning_alpha,
                threshold=f"|z| > {self.critical_z:.3f} (Bonferroni over {max_lag} lags)",
                targets=("F6", "F4"),
                limitation=(
                    "Exploratory: the normal approximation is unreliable in extreme tails, so "
                    "this detector may only warn. Warnings are expected at the stated rate."
                ),
            ),
            "binary",
            window,
        )

    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        centred = rows - rows.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centred, n=2 * self.window, axis=1)
        autocovariance = np.fft.irfft(spectrum * spectrum.conj(), axis=1)[:, : self.max_lag + 1]
        variance = autocovariance[:, 0]
        valid = variance > 1e-12
        z = np.zeros((rows.shape[0], self.max_lag))
        z[valid] = np.abs(autocovariance[valid, 1:] / variance[valid, None]) * math.sqrt(
            self.window
        )
        peak = z.max(axis=1)
        alarm = valid & (peak > self.critical_z)

        def explain(i: int) -> tuple[str, Evidence]:
            lag = int(np.argmax(z[i])) + 1
            return (
                f"Exploratory: autocorrelation at lag {lag} gives |z| = {peak[i]:.2f} > "
                f"{self.critical_z:.2f}. This suggests dependence, but the approximation is weak "
                "in the tails.",
                {"lag": lag, "z": round(float(peak[i]), 3)},
            )

        return alarm, explain


class ByteChiSquareDetector(_WindowedDetector):
    """EXPLORATORY (byte sources): Pearson chi-square against uniform bytes."""

    def __init__(self, warning_alpha: float, window: int = 4096) -> None:
        self.warning_alpha = validate_alpha(warning_alpha)
        super().__init__(
            DetectorSpec(
                name="byte-chi-square",
                title="Byte chi-square (exploratory)",
                level=DetectorLevel.WARNING,
                null_model="IID uniform bytes; chi-square approximation with 255 df",
                statistic=f"Pearson chi-square of byte counts in windows of {window}",
                alpha=self.warning_alpha,
                threshold=f"p-value < {self.warning_alpha:.3g}",
                targets=("F2", "F1"),
                limitation=(
                    "Exploratory: the chi-square approximation is unreliable in extreme tails. "
                    "Uniform-looking but predictable output passes."
                ),
            ),
            "byte",
            window,
        )

    def _evaluate(
        self, rows: Samples
    ) -> tuple[npt.NDArray[np.bool_], Callable[[int], tuple[str, Evidence]]]:
        m = rows.shape[0]
        flat = rows.astype(np.int64) + 256 * np.arange(m, dtype=np.int64)[:, None]
        counts = np.bincount(flat.ravel(), minlength=256 * m).reshape(m, 256)
        expected = self.window / 256
        statistic = ((counts - expected) ** 2 / expected).sum(axis=1)
        p_values = stats.chi2.sf(statistic, 255)
        alarm = p_values < self.warning_alpha

        def explain(i: int) -> tuple[str, Evidence]:
            return (
                f"Exploratory: byte counts give chi-square = {statistic[i]:.1f} (p = "
                f"{p_values[i]:.3g}). Byte frequencies look non-uniform.",
                {"chi_square": round(float(statistic[i]), 2), "p_value": float(p_values[i])},
            )

        return alarm, explain


# --- Source availability (not statistical) ---

AVAILABILITY_SPEC = DetectorSpec(
    name="availability",
    title="Source availability",
    level=DetectorLevel.CRITICAL,
    null_model="the source delivers every requested sample without error",
    statistic="exceptions, short reads and timeouts reported by the source",
    alpha=None,
    threshold="any event is immediately critical",
    targets=("F9",),
    limitation="Detects only failures the source itself reports. Wrong but delivered data passes.",
)


def availability_alert(index: int, mode: str, detail: str = "") -> Alert:
    """Explained alert for a source-availability failure at stream index ``index``."""
    suffix = f": {detail}" if detail else ""
    return Alert(
        AVAILABILITY_SPEC.name,
        Outcome.CRITICAL,
        index,
        f"The source reported a failure ({mode}){suffix}. No output may be produced from it.",
        {"mode": mode},
        AVAILABILITY_SPEC.limitation,
    )
