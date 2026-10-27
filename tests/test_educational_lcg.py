"""Tests for Layer B: the educational LCG and the demonstration attacks on it.

The LCG is deterministic, so exact known-answer assertions are appropriate here, unlike for the
production secure source.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from entropylens.security.base import ByteGenerator
from entropylens.security.educational_lcg import (
    DRAND48_HIGH32,
    NUMERICAL_RECIPES_32,
    PRESETS,
    EducationalLCG,
    LCGParameters,
    VectorizedLCG,
    has_full_period,
    predict_outputs,
    recover_parameters,
    recover_seed,
    recover_truncated_state,
)

# First outputs of x[n+1] = (1664525*x[n] + 1013904223) mod 2^32 from x[0] = 0. These match the
# sequence published for the Numerical Recipes "quick and dirty" generator; the second term was
# also checked by hand: (1664525*1013904223 + 1013904223) mod 2^32 = 1196435762.
NR_OUTPUTS_FROM_SEED_0 = [1013904223, 1196435762, 3519870697, 2868466484, 1649599747, 2670642822]


def _period(params: LCGParameters, seed: int) -> int:
    """Brute-force cycle length from ``seed``; only for tiny moduli."""
    state, steps = params.step(seed), 1
    while state != seed and steps <= params.modulus:
        state, steps = params.step(state), steps + 1
    return steps


def _toy(multiplier: int, increment: int) -> LCGParameters:
    return LCGParameters("toy-256", multiplier, increment, 256, 0, 8, "test")


# --- Generator behaviour ---


def test_numerical_recipes_known_answer() -> None:
    assert EducationalLCG(0).outputs(6) == NR_OUTPUTS_FROM_SEED_0


def test_identical_seeds_give_identical_streams() -> None:
    assert EducationalLCG(12345).generate(4096) == EducationalLCG(12345).generate(4096)


@pytest.mark.parametrize(("seed_a", "seed_b"), [(0, 1), (1, 2), (12345, 54321)])
def test_selected_different_seeds_give_different_streams(seed_a: int, seed_b: int) -> None:
    # Checked for these specific seeds only: different seeds need not always differ in a prefix.
    assert EducationalLCG(seed_a).generate(64) != EducationalLCG(seed_b).generate(64)


def test_bytes_are_little_endian_outputs() -> None:
    expected = b"".join(value.to_bytes(4, "little") for value in NR_OUTPUTS_FROM_SEED_0[:2])
    assert EducationalLCG(0).generate(8) == expected


@pytest.mark.parametrize("n_bytes", [0, 1, 3, 4, 5, 1001])
def test_generate_returns_requested_length(n_bytes: int) -> None:
    assert len(EducationalLCG(7).generate(n_bytes)) == n_bytes


@pytest.mark.parametrize("splits", [(3, 5), (1, 1, 1, 1, 4), (0, 8), (7, 1)])
def test_byte_stream_is_continuous_across_calls(splits: tuple[int, ...]) -> None:
    lcg = EducationalLCG(99)
    assert b"".join(lcg.generate(n) for n in splits) == EducationalLCG(99).generate(sum(splits))


def test_drand48_preset_outputs_top_32_bits_of_state() -> None:
    lcg = EducationalLCG(0x1234_5678_9ABC, DRAND48_HIGH32)
    state = 0x1234_5678_9ABC
    for _ in range(5):
        state = (0x5DEECE66D * state + 0xB) % (1 << 48)
        assert lcg.next_output() == state >> 16
        assert lcg.state == state


def test_lowest_bit_alternates_for_power_of_two_modulus() -> None:
    low_bits = [value & 1 for value in EducationalLCG(2024).outputs(64)]
    assert all(a != b for a, b in pairwise(low_bits))


def test_info_marks_generator_insecure_and_deterministic() -> None:
    lcg = EducationalLCG(1)
    assert isinstance(lcg, ByteGenerator)
    assert lcg.info.identifier == "lcg-numerical-recipes-32"
    assert lcg.info.security_status == "educational-insecure"
    assert lcg.info.deterministic is True
    assert "mod 2^32" in lcg.info.backend


def test_seeded_from_clock_uses_whole_seconds() -> None:
    assert EducationalLCG.seeded_from_clock(clock=lambda: 1_700_000_000.9).seed == 1_700_000_000


def test_presets_registry() -> None:
    assert PRESETS == {
        "numerical-recipes-32": NUMERICAL_RECIPES_32,
        "drand48-high32": DRAND48_HIGH32,
    }


# --- Input validation ---


@pytest.mark.parametrize("seed", [-1, 1 << 32])
def test_out_of_range_seeds_are_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        EducationalLCG(seed)


@pytest.mark.parametrize("seed", ["1", 1.0, True, None])
def test_non_integer_seeds_are_rejected(seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        EducationalLCG(seed)  # type: ignore[arg-type]


def test_generate_and_outputs_validate_arguments() -> None:
    lcg = EducationalLCG(1)
    with pytest.raises(ValueError, match="non-negative"):
        lcg.generate(-1)
    with pytest.raises(TypeError, match="integer"):
        lcg.generate(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        lcg.outputs(-1)
    with pytest.raises(TypeError, match="count"):
        lcg.outputs(True)


@pytest.mark.parametrize(
    ("multiplier", "increment", "modulus", "shift", "bits"),
    [
        (5, 3, 1, 0, 8),  # modulus too small
        (0, 3, 256, 0, 8),  # multiplier zero
        (256, 3, 256, 0, 8),  # multiplier >= modulus
        (5, 256, 256, 0, 8),  # increment >= modulus
        (5, 3, 256, 0, 12),  # output bits not a multiple of 8
        (5, 3, 256, 1, 8),  # output window beyond the state
    ],
)
def test_invalid_parameters_are_rejected(
    multiplier: int, increment: int, modulus: int, shift: int, bits: int
) -> None:
    with pytest.raises(ValueError, match=r"modulus|multiplier|increment|output"):
        LCGParameters("bad", multiplier, increment, modulus, shift, bits, "test")


# --- Hull-Dobell period theorem ---


def test_presets_have_full_period() -> None:
    for params in PRESETS.values():
        assert has_full_period(params.multiplier, params.increment, params.modulus)


@pytest.mark.parametrize(
    ("multiplier", "increment", "expected"),
    [
        (5, 3, True),  # a - 1 = 4, c odd: all conditions hold
        (3, 3, False),  # a - 1 = 2 is not divisible by 4
        (5, 2, False),  # gcd(c, m) = 2
    ],
)
def test_hull_dobell_agrees_with_brute_force(
    multiplier: int, increment: int, expected: bool
) -> None:
    params = _toy(multiplier, increment)
    assert has_full_period(multiplier, increment, 256) is expected
    assert all(_period(params, seed) == 256 for seed in range(256)) is expected


def test_multiplicative_lcg_never_has_full_period() -> None:
    assert has_full_period(5, 0, 256) is False


def test_full_period_with_non_power_of_two_modulus() -> None:
    # m = 45 = 3^2 * 5: a - 1 = 15 is divisible by 3 and 5; 4 does not divide m.
    assert has_full_period(16, 7, 45) is True
    assert has_full_period(4, 7, 45) is False  # a - 1 = 3 is not divisible by 5


def test_hull_dobell_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="require"):
        has_full_period(0, 1, 16)


# --- Vectorised implementation (benchmarks) ---


@pytest.mark.parametrize("params", [NUMERICAL_RECIPES_32, DRAND48_HIGH32], ids=lambda p: p.name)
@pytest.mark.parametrize("splits", [(10_000,), (1, 3, 4, 9999), (8192, 8193)])
def test_vectorized_lcg_emits_exactly_the_scalar_stream(
    params: LCGParameters, splits: tuple[int, ...]
) -> None:
    fast = VectorizedLCG(123_456, params, block=1000)
    expected = EducationalLCG(123_456, params).generate(sum(splits))
    assert b"".join(fast.generate(n) for n in splits) == expected


def test_vectorized_lcg_is_labelled_insecure() -> None:
    info = VectorizedLCG(1).info
    assert info.identifier == "lcg-numerical-recipes-32-numpy"
    assert info.security_status == "educational-insecure"
    assert info.deterministic is True


def test_vectorized_lcg_rejects_unsupported_parameters() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        VectorizedLCG(1, LCGParameters("m257", 5, 3, 257, 0, 8, "test"))
    with pytest.raises(ValueError, match="widths"):
        VectorizedLCG(1, LCGParameters("w24", 5, 3, 1 << 32, 8, 24, "test"))


# --- Demonstration attacks ---


def test_one_full_state_output_predicts_all_later_outputs() -> None:
    victim = EducationalLCG(987_654_321)
    (observed,) = victim.outputs(1)
    assert predict_outputs(NUMERICAL_RECIPES_32, observed, 10) == victim.outputs(10)


def test_three_outputs_reveal_unknown_multiplier_and_increment() -> None:
    secret = LCGParameters("secret", 0x2545F491, 0x6B8B4567, 1 << 32, 0, 32, "test")
    assert has_full_period(secret.multiplier, secret.increment, secret.modulus)
    victim = EducationalLCG(424_242, secret)
    observed = victim.outputs(3)

    recovered = recover_parameters(observed, 1 << 32)

    assert recovered == [(secret.multiplier, secret.increment)]
    attacker = LCGParameters("recovered", *recovered[0], 1 << 32, 0, 32, "attack")
    assert predict_outputs(attacker, observed[-1], 10) == victim.outputs(10)


def test_ambiguous_observations_still_contain_the_true_parameters() -> None:
    # Even increment with an even seed keeps every difference even, so several (a, c) fit.
    params = LCGParameters("even", 5, 2, 1 << 16, 0, 16, "test")
    observed = EducationalLCG(4, params).outputs(4)
    recovered = recover_parameters(observed, 1 << 16)
    assert (5, 2) in recovered
    assert len(recovered) > 1


def test_non_lcg_sequence_yields_no_parameters() -> None:
    assert recover_parameters([0, 1, 3, 4], 1 << 32) == []


@pytest.mark.parametrize(
    ("observed", "match"),
    [([1, 2], "three"), ([1, 2, 1 << 32], "lie in"), ([7, 7, 7], "candidate")],
)
def test_recover_parameters_rejects_unusable_input(observed: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        recover_parameters(observed, 1 << 32)


def test_truncated_outputs_reveal_state_with_2_to_16_guesses() -> None:
    victim = EducationalLCG(0x0123_4567_89AB, DRAND48_HIGH32)
    observed = victim.outputs(2)

    states = recover_truncated_state(DRAND48_HIGH32, observed)

    assert states == [victim.state]
    assert predict_outputs(DRAND48_HIGH32, states[0], 10) == victim.outputs(10)


def test_truncated_recovery_requires_top_bit_outputs_and_two_outputs() -> None:
    middle_bits = LCGParameters("middle", 1664525, 1013904223, 1 << 32, 8, 16, "test")
    with pytest.raises(ValueError, match="top bits"):
        recover_truncated_state(middle_bits, [1, 2])
    with pytest.raises(ValueError, match="two"):
        recover_truncated_state(DRAND48_HIGH32, [1])
    with pytest.raises(ValueError, match="lie in"):
        recover_truncated_state(DRAND48_HIGH32, [1 << 32, 0])


def test_time_based_seed_is_recovered_from_a_one_hour_window() -> None:
    seeded_at = 1_767_225_600  # a fixed Unix time, so the test is deterministic
    victim = EducationalLCG.seeded_from_clock(clock=lambda: seeded_at + 0.25)
    observed = victim.outputs(2)

    window = range(seeded_at - 1800, seeded_at + 1800)
    assert recover_seed(NUMERICAL_RECIPES_32, observed, window) == [seeded_at]


def test_recover_seed_enforces_limits() -> None:
    with pytest.raises(ValueError, match="at least one"):
        recover_seed(NUMERICAL_RECIPES_32, [], range(10))
    with pytest.raises(ValueError, match="more than 5"):
        recover_seed(NUMERICAL_RECIPES_32, [1], range(10), max_trials=5)
