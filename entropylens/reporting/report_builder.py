"""Structured experiment report built from the experiment history.

Phase 5 produces Markdown, shown in the dashboard and downloadable. Phase 6 adds the HTML export.

The report states measurements with their intervals and the conditions under which they were made.
It never declares a hypothesis supported: that judgement follows the pre-declared decision rules in
``docs/methodology.md`` section 8, applied to evaluation runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from entropylens.storage.history import BenchmarkRow, CellRow, RunRow

LIMITATIONS = (
    "Sources are simulated statistical models; real hardware can fail in ways the fault taxonomy "
    "does not contain.",
    "Detection is relative to the nominal model and its claimed min-entropy, which is known by "
    "construction here but must be estimated for real sources.",
    "Statistical monitoring cannot detect a source that emits predictable but statistically "
    "perfect output (negative control N1, scenario F8b). Passing it is not evidence of security.",
    "Benchmarks describe one machine and a Python implementation; native code would differ. "
    "Throughput is not a security metric.",
    "This is a research prototype, not a certified or NIST-compliant product.",
)


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _number(value: float | None, digits: int = 0) -> str:
    return "" if value is None else f"{value:,.{digits}f}"


def build_markdown_report(
    runs: Sequence[RunRow],
    cells: Sequence[CellRow],
    benchmarks: Sequence[BenchmarkRow],
    *,
    title: str = "EntropyLens experiment report",
) -> str:
    """Markdown report of the given runs, cells and benchmark rows."""
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    parts = [
        f"# {title}",
        f"_Generated {generated}. Measurements only: hypotheses are assessed with the "
        "pre-declared decision rules in docs/methodology.md section 8, not by this report._",
    ]
    development = [r for r in runs if r.purpose != "evaluation" and r.kind != "benchmark"]
    if development:
        parts.append(
            "> **Development runs included.** "
            + ", ".join(r.name for r in development)
            + " are not evaluation results and must not be reported as findings."
        )

    parts.append("## 1. Runs")
    parts.append(
        _table(
            ["file", "kind", "name", "purpose", "started (UTC)", "master seed", "sha256"],
            [
                [
                    r.file_name,
                    r.kind,
                    r.name,
                    r.purpose or "",
                    r.started_utc,
                    r.master_seed or "",
                    r.sha256[:16] + "...",
                ]
                for r in runs
            ],
        )
        if runs
        else "_No runs selected._"
    )

    parts.append("## 2. Method in brief")
    parts.append(
        "Each cell runs independent trials whose seeds derive from the master seed, the cell index "
        "and the trial index. A fault starts at a uniformly drawn onset. Detection means a "
        "critical alarm at or after the onset within the post-onset budget. Rates carry Wilson "
        "intervals (Clopper-Pearson at 0 or n). Thresholds derive from the false-alarm budget "
        "alpha and the model's claimed min-entropy only."
    )

    parts.append("## 3. Detection results")
    if cells:
        parts.append(
            _table(
                [
                    "run",
                    "cell",
                    "pipeline",
                    "model",
                    "fault",
                    "log2 alpha",
                    "trials",
                    "detected",
                    "rate [95% CI]",
                    "median latency",
                    "p95 latency",
                    "pre-onset alarms",
                ],
                [
                    [
                        c.run_name,
                        c.cell_name,
                        c.pipeline,
                        c.model,
                        c.fault,
                        c.alpha_log2,
                        c.trials,
                        c.detected,
                        f"{c.rate:.3f} [{c.rate_low:.3f}, {c.rate_high:.3f}]",
                        _number(c.latency_median),
                        _number(c.latency_p95),
                        c.pre_onset_alarms,
                    ]
                    for c in cells
                ],
            )
        )
        parts.append(
            "Pipelines: P2 is raw-source monitoring; P1 is output tests after SHA-256 "
            "conditioning; P3 is output tests after a DRBG. For a healthy cell, 'detected' "
            "counts false alarms."
        )
    else:
        parts.append("_No detection cells selected._")

    parts.append("## 4. Benchmarks")
    generators = [b for b in benchmarks if b.section == "generator"]
    pipeline = [b for b in benchmarks if b.section == "pipeline"]
    if generators:
        parts.append(
            _table(
                [
                    "generator",
                    "security status",
                    "buffer (bytes)",
                    "median latency (us)",
                    "throughput MiB/s [95% CI]",
                ],
                [
                    [
                        b.subject,
                        b.security_status,
                        b.size,
                        _number(b.median_us, 1),
                        f"{_number(b.throughput_mib_s, 1)} [{_number(b.throughput_low, 1)}, "
                        f"{_number(b.throughput_high, 1)}]",
                    ]
                    for b in generators
                ],
            )
        )
    if pipeline:
        parts.append(
            _table(
                ["variant", "buffer (samples)", "P2/P0 throughput ratio [95% CI]"],
                [
                    [
                        b.subject,
                        b.size,
                        f"{_number(b.ratio, 4)} "
                        f"[{_number(b.ratio_low, 4)}, {_number(b.ratio_high, 4)}]",
                    ]
                    for b in pipeline
                ],
            )
        )
        parts.append(
            "H3's pre-declared target is a ratio of at least 0.5 at buffers of 64 KiB or more."
        )
    if not benchmarks:
        parts.append("_No benchmark runs selected._")

    parts.append("## 5. Limitations")
    parts.append("\n".join(f"- {item}" for item in LIMITATIONS))
    return "\n\n".join(parts) + "\n"
