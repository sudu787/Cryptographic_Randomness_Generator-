"""Tests for descriptive entropy metrics and diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from entropylens.analysis import entropy_metrics as em
from entropylens.security.crypto_reference import ChaCha20Reference


def test_shannon_entropy_extremes() -> None:
    assert em.shannon_entropy(bytes(range(256))) == pytest.approx(8.0)
    assert em.shannon_entropy(bytes(1000)) == 0.0
    assert em.shannon_entropy(b"\x00\x01" * 50) == pytest.approx(1.0)


def test_shannon_entropy_cannot_see_predictability() -> None:
    # The negative-control keystream is fully predictable, yet scores near the maximum.
    keystream = ChaCha20Reference.negative_control().generate(1 << 16)
    assert em.shannon_entropy(keystream) > 7.99


def test_pattern_entropy_falls_for_repetition() -> None:
    random_bits = np.random.default_rng(0).integers(0, 2, 100_000, dtype=np.uint8)
    repeated = np.tile(random_bits[:16], 6250)
    assert em.pattern_entropy(random_bits, 8) == pytest.approx(8.0, abs=0.01)
    assert em.pattern_entropy(repeated, 8) <= 4.0  # at most 16 distinct patterns


def test_sliding_pattern_entropy_tracks_a_change() -> None:
    bits = np.random.default_rng(1).integers(0, 2, 20_000, dtype=np.uint8)
    bits[10_000:] = 0
    ends, values = em.sliding_pattern_entropy(bits, window=2048, step=2048)
    assert ends[0] == 2047
    assert values[0] > 7.8
    assert values[-1] == 0.0


def test_most_common_value_matches_the_sp800_90b_example() -> None:
    # SP 800-90B 6.3.1: p_hat = 0.4, L = 20. The text prints p_u = 0.6895 and H = 0.5363 but also
    # "-log2(0.6822)"; sqrt(0.24 / 19) gives p_u = 0.6895, consistent with H = 0.5363.
    data = np.array([0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1], dtype=np.uint8)
    estimate = em.most_common_value_estimate(data)
    assert estimate.most_common == 1
    assert estimate.p_hat == pytest.approx(0.4)
    assert estimate.p_upper == pytest.approx(0.6895, abs=5e-5)
    assert estimate.min_entropy == pytest.approx(0.5363, abs=5e-5)


def test_most_common_value_is_capped_at_probability_one() -> None:
    estimate = em.most_common_value_estimate(bytes(100))
    assert estimate.p_upper == 1.0
    assert estimate.min_entropy == 0.0


def test_compression_heuristic() -> None:
    assert em.compression_ratio(bytes(10_000)).zlib_ratio < 0.01
    random_bytes = np.random.default_rng(2).integers(0, 256, 10_000, dtype=np.uint8).tobytes()
    ratios = em.compression_ratio(random_bytes)
    assert ratios.zlib_ratio > 0.99
    assert ratios.lzma_ratio > 0.99


def test_autocorrelation_diagnostic() -> None:
    sticky = np.repeat(np.random.default_rng(3).integers(0, 2, 5000, dtype=np.uint8), 4)
    r = em.autocorrelation(sticky, max_lag=8)
    assert r.shape == (8,)
    assert r[0] > 0.6  # adjacent samples usually equal
    white = np.random.default_rng(4).integers(0, 2, 20_000, dtype=np.uint8)
    assert np.abs(em.autocorrelation(white, 16)).max() < 5 / math.sqrt(20_000)
    assert em.autocorrelation(np.zeros(100, dtype=np.uint8), 4).tolist() == [0.0] * 4


def test_byte_histogram() -> None:
    histogram = em.byte_histogram(b"\x00\x00\xff")
    assert histogram.shape == (256,)
    assert histogram[0] == 2
    assert histogram[255] == 1


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        em.shannon_entropy(b"")
    with pytest.raises(ValueError, match="alphabet"):
        em.shannon_entropy(np.array([3], dtype=np.uint8), alphabet=2)
    with pytest.raises(ValueError, match="0/1"):
        em.pattern_entropy(np.array([0, 2, 1, 1], dtype=np.uint8), 2)
    with pytest.raises(ValueError, match="two samples"):
        em.most_common_value_estimate(b"\x00")
    with pytest.raises(ValueError, match="non-empty"):
        em.compression_ratio(b"")
    with pytest.raises(TypeError, match="expected"):
        em.shannon_entropy("text")  # type: ignore[arg-type]
