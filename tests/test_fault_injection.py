"""Tests for labelled fault injection (F1-F10, N1)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from entropylens.experiments import fault_injection as fi
from entropylens.experiments.synthetic_sources import S0, S0_BYTE
from entropylens.security.crypto_reference import ChaCha20Reference

ONSET = 4096
LENGTH = 20_000


def _inject(fault: fi.FaultBase, *, kind: str = "binary", seed: int = 1) -> fi.InjectedTrace:
    model = S0 if kind == "binary" else S0_BYTE
    rng = np.random.default_rng(seed)
    healthy = model.generate(rng, LENGTH)
    return fi.inject(healthy, model.kind, fault, ONSET, rng, lambda n: model.generate(rng, n))


def _healthy(kind: str = "binary", seed: int = 1) -> np.ndarray:
    model = S0 if kind == "binary" else S0_BYTE
    return model.generate(np.random.default_rng(seed), LENGTH)


def test_samples_before_onset_are_untouched_and_labelled_healthy() -> None:
    trace = _inject(fi.BiasFault())
    assert np.array_equal(trace.samples[:ONSET], _healthy()[:ONSET])
    assert not trace.faulty[:ONSET].any()
    assert trace.onset == ONSET


def test_constant_fault() -> None:
    trace = _inject(fi.ConstantFault(value=1))
    assert (trace.samples[ONSET:] == 1).all()
    assert trace.faulty[ONSET:].all()


def test_stuck_bit_fault_on_bytes() -> None:
    trace = _inject(fi.StuckBitFault(bit=3, value=1), kind="byte")
    assert ((trace.samples[ONSET:] >> 3) & 1 == 1).all()
    untouched_bits = trace.samples[ONSET:] & 0b1111_0111
    assert np.array_equal(untouched_bits, _healthy("byte")[ONSET:] & 0b1111_0111)


def test_bias_fault_changes_the_proportion() -> None:
    trace = _inject(fi.BiasFault(p_one=0.75))
    assert trace.samples[ONSET:].mean() == pytest.approx(0.75, abs=0.02)


def test_pattern_fault_is_periodic() -> None:
    trace = _inject(fi.PatternFault(period=16))
    after = trace.samples[ONSET:]
    assert np.array_equal(after[16:], after[:-16])


def test_block_replay_copies_the_earlier_block_once() -> None:
    trace = _inject(fi.BlockReplayFault(block_samples=512, distance=3000))
    healthy = _healthy()
    copied = trace.samples[ONSET : ONSET + 512]
    assert np.array_equal(copied, healthy[ONSET - 3000 : ONSET - 2488])
    assert np.array_equal(trace.samples[ONSET + 512 :], healthy[ONSET + 512 :])
    assert trace.faulty.sum() == 512


def test_block_replay_with_short_distance_becomes_periodic() -> None:
    trace = _inject(fi.BlockReplayFault(block_samples=100, distance=7))
    segment = trace.samples[ONSET - 7 : ONSET + 100]
    assert np.array_equal(segment[7:], segment[:-7])


def test_repeated_block_replay() -> None:
    trace = _inject(fi.BlockReplayFault(block_samples=64, distance=1000, repeat_every=4000))
    assert trace.faulty.sum() == 64 * len(range(ONSET, LENGTH, 4000))


def test_correlation_fault_increases_stickiness() -> None:
    after = _inject(fi.CorrelationFault(stay_probability=0.8)).samples[ONSET:]
    assert np.mean(after[1:] == after[:-1]) == pytest.approx(0.8, abs=0.02)


def test_change_point_fault_uses_given_transitions() -> None:
    after = _inject(fi.ChangePointFault(p01=0.2, p11=0.6)).samples[ONSET:]
    assert after.mean() == pytest.approx(0.2 / (0.2 + 0.4), abs=0.03)


def test_replay_from_start_reproduces_the_beginning() -> None:
    trace = _inject(fi.ReplayFromStartFault(replay_from=100))
    assert np.array_equal(trace.samples[ONSET:], _healthy()[100 : 100 + LENGTH - ONSET])
    assert trace.faulty[ONSET:].all()


def test_low_entropy_substitution_is_reproducible_lcg_output() -> None:
    first, second = (
        _inject(fi.LowEntropySubstitutionFault()),
        _inject(fi.LowEntropySubstitutionFault()),
    )
    assert np.array_equal(first.samples, second.samples)
    assert not np.array_equal(first.samples[ONSET:], _healthy()[ONSET:])


def test_negative_control_is_the_published_key_keystream() -> None:
    trace = _inject(fi.NegativeControlFault())
    expected_bytes = ChaCha20Reference.negative_control().generate((LENGTH - ONSET + 7) // 8)
    expected = np.unpackbits(np.frombuffer(expected_bytes, dtype=np.uint8))[: LENGTH - ONSET]
    assert np.array_equal(trace.samples[ONSET:], expected)


def test_outage_truncates_the_trace() -> None:
    trace = _inject(fi.OutageFault(mode="short-read"))
    assert trace.samples.size == ONSET
    assert trace.outage == fi.Outage(ONSET, "short-read")
    assert not trace.faulty.any()


def test_intermittent_fault_applies_bursts_only() -> None:
    trace = _inject(fi.IntermittentFault(base=fi.ConstantFault(value=1), mean_on=50, mean_off=500))
    active = trace.faulty[ONSET:]
    assert 0.03 < active.mean() < 0.2  # expected 50 / 550, about 0.09
    assert (trace.samples[ONSET:][active] == 1).all()
    assert np.array_equal(trace.samples[ONSET:][~active], _healthy()[ONSET:][~active])


def test_burst_mask_has_the_requested_mean_lengths() -> None:
    mask = fi.burst_mask(np.random.default_rng(4), 2_000_000, 64, 1024)
    assert mask[0]
    assert mask.mean() == pytest.approx(64 / (64 + 1024), rel=0.05)


def test_injection_is_deterministic_for_a_seed() -> None:
    fault = fi.IntermittentFault(base=fi.BiasFault(p_one=0.9))
    assert np.array_equal(_inject(fault, seed=8).samples, _inject(fault, seed=8).samples)


# --- Compatibility and validation ---


@pytest.mark.parametrize(
    ("fault", "kind"),
    [
        (fi.BiasFault(), "byte"),
        (fi.CorrelationFault(), "byte"),
        (fi.ChangePointFault(), "byte"),
        (fi.IntermittentFault(base=fi.BiasFault()), "byte"),
        (fi.StuckBitFault(), "binary"),
        (fi.ConstantFault(value=7), "binary"),
    ],
)
def test_incompatible_faults_are_rejected(fault: fi.FaultBase, kind: str) -> None:
    with pytest.raises(ValueError, match=r"only|stuck"):
        fi.check_compatible(fault, kind)  # type: ignore[arg-type]


def test_invalid_onsets_and_replays_are_rejected() -> None:
    healthy = _healthy()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="onset"):
        fi.inject(healthy, "binary", fi.ConstantFault(), 0, rng, lambda n: healthy[:n])
    with pytest.raises(ValueError, match="before the start"):
        fi.inject(
            healthy, "binary", fi.BlockReplayFault(distance=5000), ONSET, rng, lambda n: healthy[:n]
        )
    with pytest.raises(ValueError, match="before the onset"):
        fi.inject(
            healthy,
            "binary",
            fi.ReplayFromStartFault(replay_from=ONSET),
            ONSET,
            rng,
            lambda n: healthy[:n],
        )


def test_fault_specs_parse_from_json_strictly() -> None:
    adapter: TypeAdapter[fi.FaultBase] = TypeAdapter(fi.FaultSpec)
    parsed = adapter.validate_json(
        json.dumps({"fault": "F10", "base": {"fault": "F3", "p_one": 0.7}, "mean_on": 32.0})
    )
    assert isinstance(parsed, fi.IntermittentFault)
    assert isinstance(parsed.base, fi.BiasFault)
    for bad in (
        {"fault": "F3", "p_one": 1.0},
        {"fault": "F3", "p_one": "0.6"},
        {"fault": "F4", "period": 1},
        {"fault": "F1", "value": 0, "extra": True},
        {"fault": "F5", "block_samples": 512, "repeat_every": 100},
        {"fault": "F10", "base": {"fault": "F9"}},
        {"fault": "F99"},
    ):
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(bad))
