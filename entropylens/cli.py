"""Command-line interface: ``python -m entropylens <command>``.

Commands:

* ``list``: list the experiment configurations in the configs directory.
* ``run NAME``: run a detection or placement experiment and write a result file.
* ``benchmark``: run the generator and monitoring-overhead benchmarks (E6).
* ``metadata``: print machine and software metadata.

Configuration files are JSON validated against strict schemas; nothing in them is executed, and
names are restricted to ``[a-z0-9_-]``. Result files contain measurements and metadata only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from entropylens.analysis.benchmark import (
    DEFAULT_BUFFER_BYTES,
    DEFAULT_PIPELINE_SAMPLES,
    benchmark_generators,
    benchmark_payload,
    benchmark_pipeline,
    machine_metadata,
    summarize_timing,
)
from entropylens.config import (
    ConfigError,
    DetectionExperimentConfig,
    load_config,
    validate_name,
)
from entropylens.experiments.evaluation import CellResult, TrialRecord, run_cell, summarize
from entropylens.experiments.placement import (
    PlacementResult,
    run_placement_cell,
    summarize_placement,
)
from entropylens.experiments.synthetic_sources import S0
from entropylens.storage.repository import StorageError, to_jsonable, write_result
from entropylens.storage.schemas import (
    RESULT_SCHEMA_VERSION,
    ResultDocument,
    RunKind,
    RunMetadata,
)

DEFAULT_CONFIGS = Path("experiments") / "configs"
DEFAULT_RESULTS = Path("experiments") / "results"
DETECTOR_FIELDS = (
    "evaluations_before",
    "alarms_before",
    "first_alarm_before",
    "evaluations_after",
    "alarms_after",
    "first_alarm_after",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _progress(label: str, stream: TextIO) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            print(f"  {label}: {done}/{total} trials", file=stream, flush=True)

    return report


def _trial_payload(record: TrialRecord) -> dict[str, Any]:
    return {
        "trial": record.trial,
        "spawn_key": list(record.spawn_key),
        "onset": record.onset,
        "length": record.length,
        "faulty_samples": record.faulty_samples,
        "faulty_before_buffer": list(record.faulty_before_buffer),
        "detectors": {
            name: [getattr(outcome, field) for field in DETECTOR_FIELDS]
            for name, outcome in record.detectors.items()
        },
    }


def detection_cell_payload(result: CellResult) -> dict[str, Any]:
    """Result-file payload for one detection cell: settings, summary and compact trials."""
    return {
        "spec": to_jsonable(result.spec),
        "cell_index": result.cell_index,
        "elapsed_seconds": result.elapsed_seconds,
        "critical": list(result.critical),
        "warning": list(result.warning),
        "disabled": dict(result.disabled),
        "alphas": dict(result.alphas),
        "false_alarm_probabilities": dict(result.false_alarm_probabilities),
        "summary": to_jsonable(summarize(result)),
        "detector_fields": list(DETECTOR_FIELDS),
        "trials": [_trial_payload(record) for record in result.trials],
    }


def placement_cell_payload(result: PlacementResult) -> dict[str, Any]:
    """Result-file payload for one placement cell (E3)."""
    return {
        "spec": to_jsonable(result.spec),
        "cell_index": result.cell_index,
        "elapsed_seconds": result.elapsed_seconds,
        "monitor_detectors": list(result.monitor_detectors),
        "block_bits": result.block_bits,
        "family_alpha": result.family_alpha,
        "summary": to_jsonable(summarize_placement(result)),
        "detector_fields": list(DETECTOR_FIELDS),
        "trials": [
            {**_trial_payload(t.record), "p1": to_jsonable(t.p1), "p3": to_jsonable(t.p3)}
            for t in result.trials
        ],
    }


def _write(
    kind: RunKind,
    name: str,
    started: str,
    command: str,
    config: dict[str, Any] | None,
    payload: dict[str, Any],
    out: Path,
) -> Path:
    document = ResultDocument(
        metadata=RunMetadata(
            schema_version=RESULT_SCHEMA_VERSION,
            kind=kind,
            name=name,
            started_utc=started,
            finished_utc=_now(),
            command=command,
            machine=machine_metadata(),
        ),
        config=config,
        payload=payload,
    )
    path, digest = write_result(document, out)
    print(f"wrote {path} (sha256 {digest})")
    return path


def cmd_list(args: argparse.Namespace) -> int:
    """List configurations with their purpose, analysis type and number of cells."""
    configs: Path = args.configs
    if not configs.is_dir():
        print(f"no configs directory at {configs}", file=sys.stderr)
        return 1
    for path in sorted(configs.glob("*.json")):
        try:
            config = load_config(validate_name(path.stem), configs, DetectionExperimentConfig)
        except ConfigError as exc:
            print(f"{path.stem:40s} INVALID: {str(exc).splitlines()[0]}")
            continue
        print(
            f"{config.name:40s} {config.purpose:12s} {config.analysis:10s} "
            f"{len(config.cells)} cells, master seed {config.master_seed}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a configuration and write its result file."""
    config = load_config(args.name, args.configs, DetectionExperimentConfig)
    wanted = set(args.cell or [])
    unknown = wanted - {cell.name for cell in config.cells}
    if unknown:
        print(f"unknown cells: {sorted(unknown)}", file=sys.stderr)
        return 2
    started = _now()
    print(
        f"{config.name}: {config.purpose} run, analysis={config.analysis}, "
        f"master seed {config.master_seed}"
    )
    cells = []
    for index, cell in enumerate(config.cells):  # the index is part of every trial's seed
        if wanted and cell.name not in wanted:
            continue
        progress = _progress(cell.name, sys.stderr) if not args.quiet else None
        if config.analysis == "detection":
            result = run_cell(cell, config.master_seed, index, progress)
            summary = summarize(result)
            cells.append(detection_cell_payload(result))
            lat = summary.latency_median
            print(
                f"  {cell.name:28s} detected {summary.detected:4d}/{summary.trials:<4d} "
                f"[{summary.detection.low:.3f}, {summary.detection.high:.3f}]  "
                f"median latency {'-' if lat is None else f'{lat:.0f}'}  "
                f"pre-onset alarms {summary.pre_onset_alarms}"
            )
        else:
            placement = run_placement_cell(
                cell,
                config.master_seed,
                index,
                progress,
                block_bits=config.output_block_bits,
                family_alpha=config.output_family_alpha,
            )
            ps = summarize_placement(placement)
            cells.append(placement_cell_payload(placement))
            rates = "  ".join(f"{p} {ps.detected[p]}/{ps.trials}" for p in ("P1", "P2", "P3"))
            print(f"  {cell.name:28s} {rates}")
    _write(
        config.analysis,
        config.name,
        started,
        f"run {config.name}",
        config.model_dump(mode="json"),
        {"cells": cells},
        args.results,
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Generator and monitoring-overhead benchmarks."""
    started = _now()
    sizes = (1024, 64 * 1024) if args.quick else DEFAULT_BUFFER_BYTES
    samples = (16 * 1024, 64 * 1024) if args.quick else DEFAULT_PIPELINE_SAMPLES
    repetitions = 5 if args.quick else args.repetitions
    print("benchmarking generators...", file=sys.stderr)
    generators = benchmark_generators(
        buffer_sizes=sizes, repetitions=repetitions, warmup=2, order_seed=args.seed
    )
    print("benchmarking monitoring overhead...", file=sys.stderr)
    pipeline = benchmark_pipeline(S0, samples, repetitions=repetitions, warmup=2, seed=args.seed)
    for g in generators:
        s = summarize_timing(g.timing, seed=args.seed)
        print(
            f"  {g.generator:34s} {g.buffer_bytes:>8d} B  median {s.median_us:10.1f} us  "
            f"{s.throughput_mib_s.estimate:9.2f} MiB/s"
        )
    for p in pipeline:
        print(
            f"  P2/P0 throughput {p.variant:15s} {p.buffer_samples:>8d} samples  "
            f"{p.throughput_ratio.estimate:.3f} [{p.throughput_ratio.low:.3f}, "
            f"{p.throughput_ratio.high:.3f}]"
        )
    payload = benchmark_payload(generators, pipeline, seed=args.seed)
    config = {"quick": args.quick, "repetitions": repetitions, "seed": args.seed, "model": "S0"}
    _write("benchmark", args.name, started, "benchmark", config, payload, args.results)
    return 0


def cmd_metadata(args: argparse.Namespace) -> int:
    """Print machine and software metadata as JSON."""
    print(json.dumps(machine_metadata(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (exposed for tests and documentation)."""
    parser = argparse.ArgumentParser(prog="entropylens", description=__doc__.split("\n")[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS, help="configs directory")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="results directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list experiment configurations").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="run an experiment configuration")
    run.add_argument("name", help="configuration name (file stem in the configs directory)")
    run.add_argument("--cell", action="append", help="run only this cell (repeatable)")
    run.add_argument("--quiet", action="store_true", help="no progress output")
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="run generator and monitoring benchmarks")
    bench.add_argument("--quick", action="store_true", help="few sizes and repetitions (smoke)")
    bench.add_argument("--repetitions", type=int, default=50, help="measured calls per cell")
    bench.add_argument("--seed", type=int, default=0, help="order and bootstrap seed")
    bench.add_argument("--name", type=validate_name, default="benchmark", help="result name")
    bench.set_defaults(func=cmd_benchmark)

    sub.add_parser("metadata", help="print machine metadata").set_defaults(func=cmd_metadata)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (ConfigError, StorageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code
