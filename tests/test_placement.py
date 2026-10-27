"""Tests for the E3 placement comparison (P1 output tests, P2 raw monitor, P3 after a DRBG)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from entropylens.experiments import fault_injection as fi
from entropylens.experiments.evaluation import CellSpec, TrialDesign
from entropylens.experiments.placement import (
    RAW_SAMPLES_PER_OUTPUT_BIT,
    output_tests,
    run_placement_cell,
    summarize_placement,
)

DESIGN = TrialDesign(
    warmup_samples=1 << 15, onset_span=1 << 12, post_onset_samples=1 << 17, buffer_samples=1 << 14
)
BLOCK_BITS = 1 << 14  # each output block covers 2^15 raw samples


def test_raw_to_output_mapping() -> None:
    assert RAW_SAMPLES_PER_OUTPUT_BIT == 2  # 512 raw samples -> one 256-bit digest


def test_output_tests_attribute_rejections_to_raw_indices() -> None:
    random_part = np.random.default_rng(0).integers(0, 256, 2 * 2048, dtype=np.uint8).tobytes()
    constant_part = hashlib.sha256(bytes(64)).digest() * 128  # 2 blocks of repeated digest
    output = random_part + constant_part
    raw_per_block = BLOCK_BITS * 2
    outcome = output_tests(output, onset=2 * raw_per_block, block_bits=BLOCK_BITS)
    assert outcome.rejections_before_onset == 0
    assert outcome.first_rejection_after == 3 * raw_per_block - 1
    assert outcome.blocks_tested == 3  # testing stops at the first post-onset rejection
    assert outcome.rejected_tests


def test_output_tests_count_pre_onset_rejections_as_false_alarms() -> None:
    constant = hashlib.sha256(b"x").digest() * 256
    outcome = output_tests(constant, onset=10**9, block_bits=BLOCK_BITS)
    assert outcome.rejections_before_onset == outcome.blocks_tested == 4
    assert outcome.first_rejection_after is None


def test_output_tests_validate_block_size() -> None:
    with pytest.raises(ValueError, match="block_bits"):
        output_tests(b"\x00" * 64, 0, block_bits=100)


def _run(fault: fi.FaultBase | None, trials: int = 3) -> dict[str, int]:
    spec = CellSpec(name="cell", model="S0", fault=fault, trials=trials, design=DESIGN)
    result = run_placement_cell(spec, 7, 0, block_bits=BLOCK_BITS)
    assert result.monitor_detectors[-1] == "availability"
    return summarize_placement(result).detected


def test_constant_source_is_seen_by_p1_and_p2_but_not_after_the_drbg() -> None:
    detected = _run(fi.ConstantFault())
    assert detected["P2"] == 3
    assert detected["P1"] == 3  # identical conditioner inputs give identical digests
    assert detected["P3"] == 0  # the stateful DRBG hides even a constant source


def test_bias_is_masked_by_conditioning() -> None:
    detected = _run(fi.BiasFault(p_one=0.7))
    assert detected["P2"] == 3
    assert detected["P1"] == 0
    assert detected["P3"] == 0


def test_healthy_control_and_summary_fields() -> None:
    spec = CellSpec(name="healthy", model="S0", trials=2, design=DESIGN)
    summary = summarize_placement(run_placement_cell(spec, 7, 1, block_bits=BLOCK_BITS))
    assert summary.trials == 2
    assert set(summary.detection) == {"P1", "P2", "P3"}
    assert summary.detected["P2"] == 0
    assert summary.median_latency["P2"] is None
