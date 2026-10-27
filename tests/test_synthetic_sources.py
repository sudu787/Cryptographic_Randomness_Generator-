"""Tests for simulated noise-source models.

Seeds are fixed, so the statistical checks are deterministic rather than flaky.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from entropylens.experiments.synthetic_sources import (
    MODELS,
    S0,
    S0_BYTE,
    S1,
    S2,
    SourceModel,
    binary_chain,
)


def test_model_registry_and_claimed_entropies() -> None:
    assert set(MODELS) == {"S0", "S1", "S2", "S0-byte"}
    assert S0.claimed_entropy == 1.0
    assert S1.claimed_entropy == pytest.approx(-math.log2(0.55))
    assert S2.claimed_entropy == pytest.approx(-math.log2(0.55))
    assert S0_BYTE.claimed_entropy == 8.0


def test_model_properties() -> None:
    assert S0.is_iid
    assert S1.is_iid
    assert S0_BYTE.is_iid
    assert not S2.is_iid
    assert S1.p_one == pytest.approx(0.55)
    assert S2.p_one == pytest.approx(0.5)


@pytest.mark.parametrize("model", [S0, S1, S2, S0_BYTE], ids=lambda m: m.name)
def test_generation_is_reproducible_and_typed(model: SourceModel) -> None:
    first = model.generate(np.random.default_rng(3), 10_000)
    second = model.generate(np.random.default_rng(3), 10_000)
    assert first.dtype == np.uint8
    assert first.shape == (10_000,)
    assert np.array_equal(first, second)
    assert int(first.max()) <= (1 if model.kind == "binary" else 255)


def test_iid_model_frequencies() -> None:
    samples = S1.generate(np.random.default_rng(11), 1_000_000)
    assert samples.mean() == pytest.approx(0.55, abs=0.002)


@pytest.mark.parametrize(("p01", "p11"), [(0.45, 0.55), (0.3, 0.7), (0.2, 0.9)])
def test_markov_chain_transition_frequencies(p01: float, p11: float) -> None:
    samples = binary_chain(np.random.default_rng(5), 1_000_000, p01, p11)
    previous, current = samples[:-1], samples[1:]
    assert current[previous == 0].mean() == pytest.approx(p01, abs=0.003)
    assert current[previous == 1].mean() == pytest.approx(p11, abs=0.003)
    assert samples.mean() == pytest.approx(p01 / (p01 + 1 - p11), abs=0.01)


def test_chain_continues_from_the_previous_sample() -> None:
    # With strong stickiness, the first sample almost always repeats the given previous value.
    rng = np.random.default_rng(9)
    firsts = [int(binary_chain(rng, 5, 0.01, 0.99, previous=1)[0]) for _ in range(200)]
    assert sum(firsts) >= 190


def test_byte_model_is_roughly_uniform() -> None:
    counts = np.bincount(S0_BYTE.generate(np.random.default_rng(2), 256_000), minlength=256)
    assert counts.min() > 800
    assert counts.max() < 1200


def test_invalid_models_and_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="claimed_entropy"):
        SourceModel("bad", "binary", 1.5)
    with pytest.raises(ValueError, match="transition"):
        SourceModel("bad", "binary", 0.5, 0.0, 0.5)
    with pytest.raises(ValueError, match="byte models"):
        SourceModel("bad", "byte", 8.0, 0.4, 0.6)
    with pytest.raises(ValueError, match="non-negative"):
        binary_chain(np.random.default_rng(0), -1, 0.5, 0.5)
    assert binary_chain(np.random.default_rng(0), 0, 0.4, 0.6).size == 0
