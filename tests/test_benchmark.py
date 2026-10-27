"""Tests for the benchmark harness: structure and bookkeeping, not specific timings."""

from __future__ import annotations

import pytest

from entropylens.analysis import benchmark as bm
from entropylens.experiments.synthetic_sources import S0
from entropylens.security.educational_lcg import EducationalLCG
from entropylens.security.secure_source import SecureSource


def test_time_calls_records_every_repetition() -> None:
    calls: list[int] = []
    timing = bm.time_calls(calls.append, 7, repetitions=5, warmup=2)
    assert calls == [7] * 7
    assert len(timing.wall_ns) == len(timing.cpu_ns) == 5
    assert all(t >= 0 for t in timing.wall_ns)
    assert timing.units_per_call == 7


def test_summarize_timing() -> None:
    timing = bm.Timing((1_000_000,) * 4 + (3_000_000,), (500_000,) * 5, bm.MIB)
    summary = bm.summarize_timing(timing)
    assert summary.calls == 5
    assert summary.median_us == pytest.approx(1000.0)
    assert summary.p95_us > summary.median_us
    assert summary.cpu_fraction == pytest.approx(2.5 / 7)
    assert summary.throughput_mib_s.estimate == pytest.approx(1000.0)  # 1 MiB per millisecond


def test_benchmark_generators_covers_every_cell_and_records_labels() -> None:
    factories = {"os": SecureSource, "lcg": lambda: EducationalLCG(1)}
    results = bm.benchmark_generators(factories, (64, 1024), repetitions=3, warmup=1)
    cells = {(r.generator, r.buffer_bytes) for r in results}
    assert cells == {
        ("os-csprng", 64),
        ("os-csprng", 1024),
        ("lcg-numerical-recipes-32", 64),
        ("lcg-numerical-recipes-32", 1024),
    }
    statuses = {r.generator: r.security_status for r in results}
    assert statuses == {
        "os-csprng": "production",
        "lcg-numerical-recipes-32": "educational-insecure",
    }
    assert all(r.peak_memory_bytes is not None and r.peak_memory_bytes > 0 for r in results)


def test_default_generators_are_the_e6_set() -> None:
    names = {factory().info.identifier for factory in bm.default_generators().values()}
    assert names == {
        "os-csprng",
        "lcg-numerical-recipes-32",
        "lcg-numerical-recipes-32-numpy",
        "chacha20-random-key",
    }


def test_pipeline_benchmark_reports_both_variants_with_ratio_intervals() -> None:
    results = bm.benchmark_pipeline(S0, (4096,), repetitions=4, warmup=1)
    assert {r.variant for r in results} == {"end-to-end", "processing-only"}
    for result in results:
        assert len(result.p0.wall_ns) == len(result.p2.wall_ns) == 4
        ratio = result.throughput_ratio
        assert ratio.low <= ratio.estimate <= ratio.high
        assert ratio.estimate > 0


def test_ratio_interval_is_exact_for_constant_timings() -> None:
    p0 = bm.Timing((100,) * 10, (100,) * 10, 1)
    p2 = bm.Timing((400,) * 10, (400,) * 10, 1)
    ratio = bm.ratio_interval(p0, p2)
    assert (ratio.estimate, ratio.low, ratio.high) == (0.25, 0.25, 0.25)


def test_machine_metadata_has_the_documented_fields() -> None:
    metadata = bm.machine_metadata()
    for key in ("timestamp_utc", "entropylens", "python", "os", "numpy", "cryptography"):
        assert metadata[key]
    assert "git_commit" in metadata


def test_time_calls_validates_arguments() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        bm.time_calls(print, 1, repetitions=0, warmup=0)
