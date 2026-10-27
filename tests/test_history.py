"""Tests for the SQLite experiment history: indexing, integrity, filters, comparison, export."""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from entropylens.storage.history import ExperimentHistory, to_csv, to_json


@pytest.fixture
def workspace(results_dir: Path, tmp_path: Path) -> tuple[ExperimentHistory, Path]:
    """A private copy of the shared result files and a fresh index over it."""
    copy = tmp_path / "results"
    shutil.copytree(results_dir, copy, ignore=shutil.ignore_patterns("*.sqlite3"))
    return ExperimentHistory(copy / "history.sqlite3"), copy


def test_sync_indexes_each_file_once(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    first = index.sync(directory)
    assert len(first.added) == 3
    assert not first.invalid
    second = index.sync(directory)
    assert second.added == []
    assert sorted(second.unchanged) == sorted(first.added)


def test_changed_and_missing_files_are_reported_not_reimported(
    workspace: tuple[ExperimentHistory, Path],
) -> None:
    index, directory = workspace
    index.sync(directory)
    [tiny] = directory.glob("tiny-*.json")
    [bench] = directory.glob("bench-*.json")
    tiny.write_text(tiny.read_text(encoding="utf-8").replace('"f1"', '"f9"'), encoding="utf-8")
    bench.unlink()
    report = index.sync(directory)
    assert report.modified == [tiny.name]
    assert report.missing == [bench.name]
    assert index.verify(directory) == {
        tiny.name: "modified",
        bench.name: "missing",
        next(directory.glob("place-*.json")).name: "ok",
    }
    assert {c.cell_name for c in index.cells()} >= {"f1"}  # the original record is kept


def test_invalid_files_are_skipped_with_a_reason(
    workspace: tuple[ExperimentHistory, Path],
) -> None:
    index, directory = workspace
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    report = index.sync(directory)
    assert "broken.json" in report.invalid
    assert len(report.added) == 3


def test_run_filters(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    index.sync(directory)
    assert {r.kind for r in index.runs()} == {"detection", "placement", "benchmark"}
    assert [r.name for r in index.runs(kinds=["placement"])] == ["place"]
    assert [r.name for r in index.runs(name_contains="tin")] == ["tiny"]
    assert index.runs(name_contains="%") == []  # LIKE wildcards are escaped
    assert index.runs(since="2999-01-01") == []
    assert len(index.runs(until="2999-01-01")) == 3
    assert index.runs()[0].purpose in ("development", None)


def test_queries_are_parameterised(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    index.sync(directory)
    assert index.runs(name_contains="'; DROP TABLE runs; --") == []
    assert len(index.runs()) == 3
    with pytest.raises(ValueError, match="unsupported column"):
        index.distinct("name; DROP TABLE runs")


def test_cell_rows_for_detection_and_placement(
    workspace: tuple[ExperimentHistory, Path],
) -> None:
    index, directory = workspace
    index.sync(directory)
    detection = index.cells(pipelines=["P2"])
    names = sorted((c.run_name, c.cell_name) for c in detection)
    assert names == [("place", "f1"), ("tiny", "f1"), ("tiny", "healthy")]
    tiny_f1 = next(c for c in detection if c.run_name == "tiny" and c.cell_name == "f1")
    assert (tiny_f1.detected, tiny_f1.trials, tiny_f1.fault) == (2, 2, "F1(value=0)")
    assert tiny_f1.rate_low <= tiny_f1.rate <= tiny_f1.rate_high
    placement = index.cells(run_ids=[r.id for r in index.runs(kinds=["placement"])])
    assert sorted(c.pipeline for c in placement) == ["P1", "P2", "P3"]
    assert index.cells(faults=["healthy"])[0].cell_name == "healthy"
    assert set(index.distinct("model")) == {"S0"}


def test_benchmark_rows(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    index.sync(directory)
    rows = index.benchmarks()
    sections = {r.section for r in rows}
    assert sections == {"generator", "pipeline"}
    secure = [r for r in rows if r.subject == "os-csprng"]
    assert secure
    assert all(r.security_status == "production" for r in secure)
    ratios = [r for r in rows if r.section == "pipeline"]
    assert all(r.ratio is not None and r.ratio_low <= r.ratio <= r.ratio_high for r in ratios)  # type: ignore[operator]
    assert index.benchmarks(subjects=["end-to-end"])


def test_compare_two_runs(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    index.sync(directory)
    tiny = index.runs(name_contains="tiny")[0]
    place = index.runs(name_contains="place")[0]
    rows = {(r["cell"], r["pipeline"]): r for r in index.compare(tiny.id, place.id)}
    assert rows[("f1", "P2")]["difference"] == 0.0
    assert rows[("healthy", "P2")]["rate_b"] is None
    assert rows[("f1", "P1")]["rate_a"] is None


def test_csv_export_neutralises_formulas() -> None:
    text = to_csv([{"name": "=HYPERLINK(1)", "value": -1.5, "note": "+cmd", "ok": "plain"}])
    [row] = list(csv.DictReader(io.StringIO(text)))
    assert row == {"name": "'=HYPERLINK(1)", "value": "-1.5", "note": "'+cmd", "ok": "plain"}
    assert to_csv([]) == ""


def test_json_export_of_dataclass_rows(workspace: tuple[ExperimentHistory, Path]) -> None:
    index, directory = workspace
    index.sync(directory)
    exported = json.loads(to_json(index.runs()))
    assert len(exported) == 3
    assert {"sha256", "master_seed", "started_utc"} <= set(exported[0])
