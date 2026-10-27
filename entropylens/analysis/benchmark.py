"""Benchmark harness (docs/methodology.md section 9 and experiment E6).

Interpretation rules, repeated wherever results are shown:

* Throughput is **not** a security metric. The fastest generator is not the most secure.
* The APIs differ: ``secrets.token_bytes`` makes a system call per request, ChaCha20 runs in
  OpenSSL's C code, and the LCG runs in Python or NumPy. Results describe cost on this machine and
  software stack, not intrinsic algorithm speed.

Generated output is discarded immediately. Benchmark records contain timings and metadata only.
"""

from __future__ import annotations

import gc
import os
import platform
import ssl
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

import entropylens
from entropylens.analysis.intervals import Interval, bootstrap
from entropylens.experiments.conditioning import sha256_condition
from entropylens.experiments.synthetic_sources import SourceModel
from entropylens.monitoring.source_health import HealthMonitor, MonitorSettings
from entropylens.security.base import ByteGenerator
from entropylens.security.crypto_reference import ChaCha20Reference
from entropylens.security.educational_lcg import EducationalLCG, VectorizedLCG
from entropylens.security.secure_source import SecureSource
from entropylens.storage.repository import to_jsonable

MIB = 1 << 20
DEFAULT_BUFFER_BYTES = (64, 1024, 16 * 1024, 64 * 1024, MIB)
DEFAULT_PIPELINE_SAMPLES = (1024, 16 * 1024, 64 * 1024, 256 * 1024, MIB)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(start: Path) -> str | None:
    """Commit hash read from ``.git`` files (no subprocess), or None outside a repository."""
    for directory in (start, *start.parents):
        head = directory / ".git" / "HEAD"
        if head.is_file():
            content = head.read_text(encoding="utf-8").strip()
            if content.startswith("ref: "):
                ref = directory / ".git" / content[5:]
                return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
            return content
    return None


def machine_metadata() -> dict[str, str | int | float | bool | None]:
    """Machine and software metadata recorded with every benchmark and experiment run."""
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "entropylens": entropylens.__version__,
        "git_commit": _git_commit(Path.cwd()),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_bytes": int(psutil.virtual_memory().total),
        "on_ac_power": None if battery is None else bool(battery.power_plugged),
        "gc_enabled": gc.isenabled(),
        "python_openssl": ssl.OPENSSL_VERSION,
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "cryptography": _package_version("cryptography"),
        "pydantic": _package_version("pydantic"),
    }


@dataclass(frozen=True, slots=True)
class Timing:
    """Raw per-call timings in nanoseconds. ``units`` is bytes or samples processed per call."""

    wall_ns: tuple[int, ...]
    cpu_ns: tuple[int, ...]
    units_per_call: int


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Latency percentiles, CPU share and throughput (median with a bootstrap interval)."""

    calls: int
    mean_us: float
    median_us: float
    p95_us: float
    iqr_us: float
    cpu_fraction: float
    throughput_mib_s: Interval


def time_calls(
    call: Callable[[int], object], argument: int, *, repetitions: int, warmup: int
) -> Timing:
    """Time ``call(argument)`` ``repetitions`` times after ``warmup`` untimed calls."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup non-negative")
    for _ in range(warmup):
        call(argument)
    wall, cpu = [], []
    for _ in range(repetitions):
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        call(argument)
        wall.append(time.perf_counter_ns() - wall_start)
        cpu.append(time.process_time_ns() - cpu_start)
    return Timing(tuple(wall), tuple(cpu), argument)


def summarize_timing(timing: Timing, *, seed: int = 0) -> TimingSummary:
    """Summarise a :class:`Timing`; the bootstrap seed is recorded by the caller."""
    wall = np.asarray(timing.wall_ns, dtype=np.float64)
    safe = np.maximum(wall, 1.0)  # a zero reading would mean infinite throughput
    throughput = timing.units_per_call / MIB / (safe / 1e9)
    q1, q3 = np.percentile(wall, [25, 75])
    return TimingSummary(
        calls=wall.size,
        mean_us=float(wall.mean() / 1e3),
        median_us=float(np.median(wall) / 1e3),
        p95_us=float(np.percentile(wall, 95) / 1e3),
        iqr_us=float((q3 - q1) / 1e3),
        cpu_fraction=float(sum(timing.cpu_ns) / max(1, sum(timing.wall_ns))),
        throughput_mib_s=bootstrap(throughput, seed=seed, resamples=2000),
    )


def peak_memory(call: Callable[[int], object], argument: int) -> int:
    """Peak Python-heap allocation (bytes) during one call, measured separately from timing."""
    tracemalloc.start()
    try:
        call(argument)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@dataclass(frozen=True, slots=True)
class GeneratorBenchmark:
    """One (generator, buffer size) cell."""

    generator: str
    display_name: str
    security_status: str
    buffer_bytes: int
    timing: Timing
    peak_memory_bytes: int | None


def default_generators() -> dict[str, Callable[[], ByteGenerator]]:
    """The generators compared in E6. Deterministic ones use fixed, recorded seeds."""
    return {
        "os-csprng": SecureSource,
        "lcg-numerical-recipes-32": lambda: EducationalLCG(1),
        "lcg-numerical-recipes-32-numpy": lambda: VectorizedLCG(1),
        "chacha20-random-key": ChaCha20Reference,
    }


def benchmark_generators(
    factories: Mapping[str, Callable[[], ByteGenerator]] | None = None,
    buffer_sizes: Sequence[int] = DEFAULT_BUFFER_BYTES,
    *,
    repetitions: int = 50,
    warmup: int = 5,
    order_seed: int = 0,
    measure_memory: bool = True,
) -> list[GeneratorBenchmark]:
    """Time each generator at each buffer size, visiting cells in a seeded random order.

    The random order spreads thermal and power drift across cells. Output is discarded.
    """
    factories = factories or default_generators()
    cells = [(name, size) for name in factories for size in buffer_sizes]
    order = np.random.default_rng(order_seed).permutation(len(cells))
    results = []
    for index in order:
        name, size = cells[int(index)]
        generator = factories[name]()
        timing = time_calls(generator.generate, size, repetitions=repetitions, warmup=warmup)
        memory = peak_memory(factories[name]().generate, size) if measure_memory else None
        info = generator.info
        results.append(
            GeneratorBenchmark(
                info.identifier, info.display_name, info.security_status, size, timing, memory
            )
        )
    return sorted(results, key=lambda r: (r.generator, r.buffer_bytes))


# --- Monitoring overhead (E6, hypothesis H3) ---


@dataclass(frozen=True, slots=True)
class PipelineBenchmark:
    """P0 (conditioning only) against P2 (monitor + conditioning) at one buffer size.

    ``throughput_ratio`` is median P2 throughput over median P0 throughput, with a percentile
    bootstrap interval from resampling both timing sets.
    """

    variant: str
    buffer_samples: int
    p0: Timing
    p2: Timing
    throughput_ratio: Interval


def ratio_interval(p0: Timing, p2: Timing, *, seed: int = 0, resamples: int = 2000) -> Interval:
    """Bootstrap interval for ``median(P0 time) / median(P2 time)`` (= throughput ratio P2/P0)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(p0.wall_ns, dtype=np.float64)
    b = np.asarray(p2.wall_ns, dtype=np.float64)
    estimate = float(np.median(a) / np.median(b))
    replicates = np.median(rng.choice(a, (resamples, a.size)), axis=1) / np.median(
        rng.choice(b, (resamples, b.size)), axis=1
    )
    low, high = np.quantile(replicates, [0.025, 0.975])
    return Interval(estimate, float(low), float(high), "percentile-bootstrap")


def benchmark_pipeline(
    model: SourceModel,
    buffer_sizes: Sequence[int] = DEFAULT_PIPELINE_SAMPLES,
    *,
    repetitions: int = 30,
    warmup: int = 3,
    seed: int = 0,
    settings: MonitorSettings | None = None,
) -> list[PipelineBenchmark]:
    """Measure monitoring overhead in two variants.

    * ``end-to-end`` (the H3 definition): simulate the raw samples, then condition (P0) or monitor
      and condition (P2).
    * ``processing-only``: the same, with sample generation excluded from the timed region.

    P0 and P2 calls alternate so that both see the same drift. Every timed call gets a distinct
    healthy buffer, because a repeated buffer would trip the repeated-block detector.
    """
    results = []
    for size in buffer_sizes:
        calls = warmup + repetitions
        rng = np.random.default_rng([seed, size])
        buffers = [model.generate(rng, size) for _ in range(2 * calls)]
        for variant in ("end-to-end", "processing-only"):
            monitor = HealthMonitor(model, settings)
            p0_ns: list[int] = []
            p2_ns: list[int] = []
            p0_cpu: list[int] = []
            p2_cpu: list[int] = []
            gen_rng = np.random.default_rng([seed, size, 1])
            for i in range(calls):
                for monitored, wall, cpu in ((False, p0_ns, p0_cpu), (True, p2_ns, p2_cpu)):
                    buffer = buffers[2 * i + monitored]
                    cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
                    if variant == "end-to-end":
                        buffer = model.generate(gen_rng, size)
                    if monitored:
                        monitor.process(buffer)
                    sha256_condition(buffer, model.kind)
                    if i >= warmup:
                        wall.append(time.perf_counter_ns() - start)
                        cpu.append(time.process_time_ns() - cpu_start)
            p0 = Timing(tuple(p0_ns), tuple(p0_cpu), size)
            p2 = Timing(tuple(p2_ns), tuple(p2_cpu), size)
            results.append(
                PipelineBenchmark(variant, size, p0, p2, ratio_interval(p0, p2, seed=seed))
            )
    return results


INTERPRETATION_NOTES = (
    "Throughput is not a security metric; the fastest generator is not the most secure.",
    "APIs differ: secrets.token_bytes makes a system call per request, ChaCha20 runs in OpenSSL, "
    "the LCG runs in Python or NumPy. Results describe this machine and software stack only.",
)


def benchmark_payload(
    generators: Sequence[GeneratorBenchmark],
    pipeline: Sequence[PipelineBenchmark],
    *,
    seed: int,
    extra_notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Result-file payload: raw timings, summaries and the interpretation notes."""
    return {
        "notes": [*extra_notes, *INTERPRETATION_NOTES],
        "bootstrap_seed": seed,
        "order_seed": seed,
        "generators": [
            {**to_jsonable(g), "summary": to_jsonable(summarize_timing(g.timing, seed=seed))}
            for g in generators
        ],
        "pipeline": [
            {
                **to_jsonable(p),
                "p0_summary": to_jsonable(summarize_timing(p.p0, seed=seed)),
                "p2_summary": to_jsonable(summarize_timing(p.p2, seed=seed)),
            }
            for p in pipeline
        ],
    }


def throughputs(timing: Timing) -> npt.NDArray[np.float64]:
    """Per-call throughput in MiB/s (units per call / wall time)."""
    wall = np.maximum(np.asarray(timing.wall_ns, dtype=np.float64), 1.0)
    return timing.units_per_call / MIB / (wall / 1e9)
