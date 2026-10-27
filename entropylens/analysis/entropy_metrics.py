"""Descriptive entropy metrics and diagnostics for observed data.

Every function here describes **a finite sample**, not the source that produced it, and none
measures unpredictability. The ChaCha20 keystream under a published key scores close to 8 bits per
byte on every metric here, yet an attacker who knows the key predicts every byte
(``docs/threat_model.md`` section 5). Entropy *estimation* for an entropy source is a separate,
harder problem (SP 800-90B section 6); the most-common-value estimate below is one of its
estimators, provided for comparison only.
"""

from __future__ import annotations

import lzma
import math
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Symbols = npt.NDArray[np.uint8]
MCV_Z = 2.576  # Z(1 - 0.005), as in SP 800-90B section 6.3.1


def _as_array(data: bytes | bytearray | Symbols) -> Symbols:
    if isinstance(data, bytes | bytearray):
        return np.frombuffer(bytes(data), dtype=np.uint8)
    if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
        return data
    raise TypeError("expected bytes or a one-dimensional uint8 array")


def byte_histogram(data: bytes | Symbols) -> npt.NDArray[np.int64]:
    """Counts of each byte value 0..255."""
    return np.bincount(_as_array(data), minlength=256).astype(np.int64)


def shannon_entropy(data: bytes | Symbols, alphabet: int = 256) -> float:
    """Plug-in Shannon entropy of the empirical symbol distribution, in bits per symbol.

    This is the entropy of the *sample's frequency table*. It is biased low for small samples and
    says nothing about predictability: any permutation of the sample scores the same.
    """
    symbols = _as_array(data)
    if symbols.size == 0:
        raise ValueError("cannot compute the entropy of an empty sample")
    if symbols.size and int(symbols.max()) >= alphabet:
        raise ValueError(f"symbols must be below the alphabet size {alphabet}")
    counts = np.bincount(symbols, minlength=alphabet)
    p = counts[counts > 0] / symbols.size
    return float(-(p * np.log2(p)).sum())


def pattern_entropy(bits: Symbols, width: int = 8) -> float:
    """Plug-in Shannon entropy of overlapping ``width``-bit patterns, in bits per pattern.

    For ideal random bits it approaches ``width``. Repetition and short-range dependence lower it.
    """
    if not 1 <= width <= 16:
        raise ValueError("width must be between 1 and 16")
    bits = _as_array(bits)
    if bits.size < width:
        raise ValueError("the sample is shorter than one pattern")
    if int(bits.max(initial=0)) > 1:
        raise ValueError("pattern_entropy expects a 0/1 bit array")
    windows = np.lib.stride_tricks.sliding_window_view(bits.astype(np.int64), width)
    codes = windows @ (1 << np.arange(width - 1, -1, -1))
    counts = np.bincount(codes, minlength=1 << width)
    p = counts[counts > 0] / codes.size
    return float(-(p * np.log2(p)).sum())


def sliding_pattern_entropy(
    bits: Symbols, window: int = 4096, step: int = 1024, width: int = 8
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Pattern entropy over sliding windows: (window end indices, entropy per window).

    Descriptive only (it has no exact null distribution), so it is a plot, not a detector.
    """
    bits = _as_array(bits)
    if window < width or step < 1:
        raise ValueError("require window >= width and step >= 1")
    ends = np.arange(window, bits.size + 1, step, dtype=np.int64)
    values = np.array([pattern_entropy(bits[end - window : end], width) for end in ends])
    return ends - 1, values


@dataclass(frozen=True, slots=True)
class MinEntropyEstimate:
    """SP 800-90B most-common-value estimate: ``H = -log2(p_upper)``."""

    most_common: int
    p_hat: float
    p_upper: float
    min_entropy: float
    samples: int


def most_common_value_estimate(data: bytes | Symbols) -> MinEntropyEstimate:
    """SP 800-90B section 6.3.1: min-entropy from a 99% upper bound on the most common value.

    ``p_u = min(1, p_hat + 2.576 sqrt(p_hat (1 - p_hat) / (L - 1)))``. The estimate assumes IID
    samples; applied to output data it describes that data, not the underlying source.
    """
    symbols = _as_array(data)
    if symbols.size < 2:
        raise ValueError("need at least two samples")
    counts = np.bincount(symbols)
    most_common = int(counts.argmax())
    p_hat = counts[most_common] / symbols.size
    p_upper = min(1.0, p_hat + MCV_Z * math.sqrt(p_hat * (1 - p_hat) / (symbols.size - 1)))
    return MinEntropyEstimate(most_common, float(p_hat), p_upper, -math.log2(p_upper), symbols.size)


@dataclass(frozen=True, slots=True)
class CompressionHeuristic:
    """HEURISTIC: compressed size divided by original size (1.0 means no compression found)."""

    zlib_ratio: float
    lzma_ratio: float


def compression_ratio(data: bytes) -> CompressionHeuristic:
    """Compressibility with zlib (level 9) and LZMA.

    A ratio well below 1 shows redundancy that a general-purpose compressor can exploit. A ratio
    near (or above) 1 shows only that *these* compressors found none; it is not evidence of high
    entropy, and keystream from a known key is incompressible too.
    """
    if not isinstance(data, bytes | bytearray) or len(data) == 0:
        raise ValueError("expected non-empty bytes")
    size = len(data)
    return CompressionHeuristic(
        zlib_ratio=len(zlib.compress(bytes(data), 9)) / size,
        lzma_ratio=len(lzma.compress(bytes(data), preset=9)) / size,
    )


def autocorrelation(bits: Symbols, max_lag: int = 64) -> npt.NDArray[np.float64]:
    """Sample autocorrelation r_1..r_max_lag of a 0/1 sequence (diagnostic plot data).

    For IID bits each ``r_k`` is approximately Normal(0, 1/n), so ``+/- 1.96/sqrt(n)`` bands show
    the pointwise 95% range. Many lags are examined at once, so some excursions are expected.
    """
    values = _as_array(bits).astype(np.float64)
    if values.size <= max_lag or max_lag < 1:
        raise ValueError("need more samples than max_lag, and max_lag >= 1")
    centred = values - values.mean()
    spectrum = np.fft.rfft(centred, n=2 * values.size)
    autocovariance = np.fft.irfft(spectrum * spectrum.conj())[: max_lag + 1]
    if autocovariance[0] <= 0:
        return np.zeros(max_lag)
    return np.asarray(autocovariance[1:] / autocovariance[0], dtype=np.float64)
