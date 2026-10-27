"""Experiment history: a SQLite index over the result files, with filtering, comparison and export.

The JSON result files remain the source of truth. The index is rebuilt from them at any time
(:meth:`ExperimentHistory.sync`) and is safe to delete.

Integrity: each file's SHA-256 is recorded at first import. A later sync or :meth:`verify` reports
files whose content changed (``modified``) or disappeared (``missing``). A changed file is never
silently re-imported. This detects accidental edits only; it is not a signature (threat T12).

Security: every query is parameterised. CSV exports neutralise spreadsheet formulas (cells starting
with ``= + - @``, tab or carriage return get a leading apostrophe).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from entropylens.storage.repository import StorageError, read_result
from entropylens.storage.schemas import ResultDocument

HISTORY_FILE = "history.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    purpose TEXT,
    started_utc TEXT NOT NULL,
    finished_utc TEXT NOT NULL,
    master_seed TEXT,
    package_version TEXT,
    command TEXT NOT NULL,
    cells INTEGER NOT NULL,
    indexed_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    cell_index INTEGER NOT NULL,
    cell_name TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    model TEXT NOT NULL,
    fault TEXT NOT NULL,
    alpha_log2 INTEGER,
    trials INTEGER NOT NULL,
    detected INTEGER NOT NULL,
    rate REAL NOT NULL,
    rate_low REAL NOT NULL,
    rate_high REAL NOT NULL,
    pre_onset_alarms INTEGER,
    latency_median REAL,
    latency_p95 REAL,
    exposure_mean REAL,
    PRIMARY KEY (run_id, cell_index, pipeline)
);
CREATE TABLE IF NOT EXISTS benchmarks (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    subject TEXT NOT NULL,
    security_status TEXT,
    size INTEGER NOT NULL,
    median_us REAL,
    p95_us REAL,
    throughput_mib_s REAL,
    throughput_low REAL,
    throughput_high REAL,
    ratio REAL,
    ratio_low REAL,
    ratio_high REAL
);
CREATE INDEX IF NOT EXISTS cells_by_scenario ON cells (model, fault);
CREATE INDEX IF NOT EXISTS benchmarks_by_subject ON benchmarks (subject);
"""


_SELECT_RUNS = (
    "SELECT id, file_name, sha256, kind, name, purpose, started_utc, finished_utc,"
    " master_seed, package_version, command, cells FROM runs"
)
_SELECT_CELLS = (
    "SELECT c.run_id, r.name, r.started_utc, c.cell_index, c.cell_name, c.pipeline, c.model,"
    " c.fault, c.alpha_log2, c.trials, c.detected, c.rate, c.rate_low, c.rate_high,"
    " c.pre_onset_alarms, c.latency_median, c.latency_p95, c.exposure_mean"
    " FROM cells c JOIN runs r ON r.id = c.run_id"
)
_SELECT_BENCHMARKS = (
    "SELECT b.run_id, r.started_utc, b.section, b.subject, b.security_status, b.size,"
    " b.median_us, b.p95_us, b.throughput_mib_s, b.throughput_low, b.throughput_high,"
    " b.ratio, b.ratio_low, b.ratio_high FROM benchmarks b JOIN runs r ON r.id = b.run_id"
)


def _where(clauses: Sequence[str]) -> str:
    """A WHERE clause joined from fixed column names and ``?`` placeholders.

    Filter clauses are assembled in this module only, from literal column names; every
    user-supplied value is passed separately as a query parameter.
    """
    return " WHERE " + " AND ".join(clauses) if clauses else ""


@dataclass(frozen=True, slots=True)
class RunRow:
    """One indexed result file."""

    id: int
    file_name: str
    sha256: str
    kind: str
    name: str
    purpose: str | None
    started_utc: str
    finished_utc: str
    master_seed: str | None
    package_version: str | None
    command: str
    cells: int


@dataclass(frozen=True, slots=True)
class CellRow:
    """Summary metrics of one cell (one row per pipeline for placement runs)."""

    run_id: int
    run_name: str
    started_utc: str
    cell_index: int
    cell_name: str
    pipeline: str
    model: str
    fault: str
    alpha_log2: int | None
    trials: int
    detected: int
    rate: float
    rate_low: float
    rate_high: float
    pre_onset_alarms: int | None
    latency_median: float | None
    latency_p95: float | None
    exposure_mean: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """One benchmark cell: a generator at a buffer size, or a pipeline-overhead variant."""

    run_id: int
    started_utc: str
    section: str
    subject: str
    security_status: str | None
    size: int
    median_us: float | None
    p95_us: float | None
    throughput_mib_s: float | None
    throughput_low: float | None
    throughput_high: float | None
    ratio: float | None
    ratio_low: float | None
    ratio_high: float | None


@dataclass(slots=True)
class SyncReport:
    """Outcome of indexing a results directory."""

    added: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)


def _fault_label(spec: dict[str, Any]) -> str:
    fault = spec.get("fault")
    if not fault:
        return "healthy"
    parameters = ", ".join(f"{k}={v}" for k, v in sorted(fault.items()) if k != "fault")
    return f"{fault['fault']}({parameters})" if parameters else str(fault["fault"])


class ExperimentHistory:
    """SQLite index of result files in one results directory."""

    def __init__(self, database: Path) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            yield db
            db.commit()
        finally:
            db.close()

    # --- Indexing ---

    def sync(self, results_dir: Path) -> SyncReport:
        """Index new result files; report unchanged, modified, missing and invalid ones."""
        report = SyncReport()
        present = {p.name: p for p in sorted(results_dir.glob("*.json"))}
        with self._connect() as db:
            known = dict(db.execute("SELECT file_name, sha256 FROM runs").fetchall())
            for name, path in present.items():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if name in known:
                    target = report.unchanged if known[name] == digest else report.modified
                    target.append(name)
                    continue
                try:
                    document = read_result(path, results_dir)
                except (StorageError, ValueError, OSError) as exc:
                    report.invalid[name] = str(exc).splitlines()[0]
                    continue
                self._insert(db, name, digest, document)
                report.added.append(name)
            report.missing = sorted(set(known) - set(present))
        return report

    def verify(self, results_dir: Path) -> dict[str, str]:
        """Status of every indexed file: ``ok``, ``modified`` or ``missing``."""
        status = {}
        for run in self.runs():
            path = results_dir / run.file_name
            if not path.is_file():
                status[run.file_name] = "missing"
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                status[run.file_name] = "ok" if digest == run.sha256 else "modified"
        return status

    def _insert(
        self, db: sqlite3.Connection, file_name: str, digest: str, document: ResultDocument
    ) -> None:
        meta, config, payload = document.metadata, document.config or {}, document.payload
        cells = payload.get("cells", [])
        cursor = db.execute(
            "INSERT INTO runs (file_name, sha256, kind, name, purpose, started_utc, finished_utc,"
            " master_seed, package_version, command, cells, indexed_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_name,
                digest,
                meta.kind,
                meta.name,
                config.get("purpose"),
                meta.started_utc,
                meta.finished_utc,
                None if config.get("master_seed") is None else str(config["master_seed"]),
                str(meta.machine.get("entropylens")),
                meta.command,
                len(cells) if meta.kind != "benchmark" else len(payload.get("generators", [])),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        run_id = cursor.lastrowid
        if meta.kind == "detection":
            for cell in cells:
                self._insert_cell(db, run_id, cell, "P2", cell["summary"])
        elif meta.kind == "placement":
            for cell in cells:
                summary = cell["summary"]
                for pipeline, interval in summary["detection"].items():
                    self._insert_cell(
                        db,
                        run_id,
                        cell,
                        pipeline,
                        {
                            "trials": summary["trials"],
                            "detected": summary["detected"][pipeline],
                            "detection": interval,
                            "pre_onset_alarms": summary["false_alarms_before_onset"][pipeline],
                            "latency_median": summary["median_latency"][pipeline],
                        },
                    )
        elif meta.kind == "benchmark":
            self._insert_benchmarks(db, run_id, payload)

    @staticmethod
    def _insert_cell(
        db: sqlite3.Connection,
        run_id: int | None,
        cell: dict[str, Any],
        pipeline: str,
        summary: dict[str, Any],
    ) -> None:
        spec = cell["spec"]
        detection = summary["detection"]
        db.execute(
            "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                cell["cell_index"],
                spec["name"],
                pipeline,
                spec["model"],
                _fault_label(spec),
                spec.get("monitor", {}).get("alpha_log2"),
                summary["trials"],
                summary["detected"],
                detection["estimate"],
                detection["low"],
                detection["high"],
                summary.get("pre_onset_alarms"),
                summary.get("latency_median"),
                summary.get("latency_p95"),
                summary.get("exposure_mean_samples"),
            ),
        )

    @staticmethod
    def _insert_benchmarks(
        db: sqlite3.Connection, run_id: int | None, payload: dict[str, Any]
    ) -> None:
        for g in payload.get("generators", []):
            s = g["summary"]
            t = s["throughput_mib_s"]
            db.execute(
                "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    "generator",
                    g["generator"],
                    g["security_status"],
                    g["buffer_bytes"],
                    s["median_us"],
                    s["p95_us"],
                    t["estimate"],
                    t["low"],
                    t["high"],
                    None,
                    None,
                    None,
                ),
            )
        for p in payload.get("pipeline", []):
            r = p["throughput_ratio"]
            s = p["p2_summary"]
            db.execute(
                "INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    "pipeline",
                    p["variant"],
                    None,
                    p["buffer_samples"],
                    s["median_us"],
                    s["p95_us"],
                    None,
                    None,
                    None,
                    r["estimate"],
                    r["low"],
                    r["high"],
                ),
            )

    # --- Queries ---

    def runs(
        self,
        *,
        kinds: Sequence[str] | None = None,
        name_contains: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[RunRow]:
        """Indexed runs, newest first. ``since``/``until`` compare ISO 8601 timestamps."""
        clauses, params = self._run_filters(kinds, name_contains, since, until)
        query = _SELECT_RUNS + _where(clauses) + " ORDER BY started_utc DESC, id DESC"
        with self._connect() as db:
            return [RunRow(*row) for row in db.execute(query, params).fetchall()]

    @staticmethod
    def _run_filters(
        kinds: Sequence[str] | None,
        name_contains: str | None,
        since: str | None,
        until: str | None,
        prefix: str = "",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if kinds:
            clauses.append(f"{prefix}kind IN ({', '.join('?' * len(kinds))})")
            params.extend(kinds)
        if name_contains:
            clauses.append(f"{prefix}name LIKE ? ESCAPE '\\'")
            escaped = name_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        if since:
            clauses.append(f"{prefix}started_utc >= ?")
            params.append(since)
        if until:
            clauses.append(f"{prefix}started_utc <= ?")
            params.append(until)
        return clauses, params

    def cells(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        models: Sequence[str] | None = None,
        faults: Sequence[str] | None = None,
        pipelines: Sequence[str] | None = None,
    ) -> list[CellRow]:
        """Cell summaries, filterable by run, model, fault label and pipeline."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, values in (
            ("c.run_id", run_ids),
            ("c.model", models),
            ("c.fault", faults),
            ("c.pipeline", pipelines),
        ):
            if values:
                clauses.append(f"{column} IN ({', '.join('?' * len(values))})")
                params.extend(values)
        query = (
            _SELECT_CELLS
            + _where(clauses)
            + " ORDER BY r.started_utc DESC, c.cell_index, c.pipeline"
        )
        with self._connect() as db:
            return [CellRow(*row) for row in db.execute(query, params).fetchall()]

    def benchmarks(
        self, *, run_ids: Sequence[int] | None = None, subjects: Sequence[str] | None = None
    ) -> list[BenchmarkRow]:
        """Benchmark rows, filterable by run and by generator or pipeline variant."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, values in (("b.run_id", run_ids), ("b.subject", subjects)):
            if values:
                clauses.append(f"{column} IN ({', '.join('?' * len(values))})")
                params.extend(values)
        query = (
            _SELECT_BENCHMARKS
            + _where(clauses)
            + " ORDER BY r.started_utc DESC, b.section, b.subject, b.size"
        )
        with self._connect() as db:
            return [BenchmarkRow(*row) for row in db.execute(query, params).fetchall()]

    def distinct(self, column: str) -> list[str]:
        """Distinct values for filter controls (``model``, ``fault``, ``pipeline``, ``subject``)."""
        tables = {"model": "cells", "fault": "cells", "pipeline": "cells", "subject": "benchmarks"}
        if column not in tables:  # column names cannot be parameterised, so allow-list them
            raise ValueError(f"unsupported column {column!r}")
        query = f"SELECT DISTINCT {column} FROM {tables[column]} ORDER BY {column}"  # noqa: S608
        with self._connect() as db:
            return [row[0] for row in db.execute(query).fetchall()]

    def compare(self, run_a: int, run_b: int) -> list[dict[str, Any]]:
        """Side-by-side detection rates of two runs, matched by cell name and pipeline."""
        a = {(c.cell_name, c.pipeline): c for c in self.cells(run_ids=[run_a])}
        b = {(c.cell_name, c.pipeline): c for c in self.cells(run_ids=[run_b])}
        rows = []
        for key in sorted(set(a) | set(b)):
            left, right = a.get(key), b.get(key)
            rows.append(
                {
                    "cell": key[0],
                    "pipeline": key[1],
                    "rate_a": None if left is None else left.rate,
                    "ci_a": None
                    if left is None
                    else f"[{left.rate_low:.3f}, {left.rate_high:.3f}]",
                    "rate_b": None if right is None else right.rate,
                    "ci_b": None
                    if right is None
                    else f"[{right.rate_low:.3f}, {right.rate_high:.3f}]",
                    "difference": None
                    if left is None or right is None
                    else round(right.rate - left.rate, 6),
                }
            )
        return rows


# --- Export ---

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [row if isinstance(row, dict) else asdict(row) for row in rows]


def to_csv(rows: Sequence[Any]) -> str:
    """CSV text for dataclass rows or dicts, with spreadsheet-formula neutralisation."""
    records = _rows(rows)
    if not records:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                key: f"'{value}"
                if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES)
                else value
                for key, value in record.items()
            }
        )
    return buffer.getvalue()


def to_json(rows: Sequence[Any]) -> str:
    """JSON array text for dataclass rows or dicts."""
    return json.dumps(_rows(rows), indent=2, allow_nan=False)
