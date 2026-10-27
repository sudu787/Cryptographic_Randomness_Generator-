"""Tests for the Markdown report builder."""

from __future__ import annotations

import shutil
from pathlib import Path

from entropylens.reporting.report_builder import LIMITATIONS, build_markdown_report
from entropylens.storage.history import ExperimentHistory


def _index(results_dir: Path, tmp_path: Path) -> ExperimentHistory:
    copy = tmp_path / "results"
    shutil.copytree(results_dir, copy, ignore=shutil.ignore_patterns("*.sqlite3"))
    index = ExperimentHistory(copy / "history.sqlite3")
    index.sync(copy)
    return index


def test_report_contains_every_section_and_the_measurements(
    results_dir: Path, tmp_path: Path
) -> None:
    index = _index(results_dir, tmp_path)
    runs = index.runs()
    ids = [r.id for r in runs]
    report = build_markdown_report(runs, index.cells(run_ids=ids), index.benchmarks(run_ids=ids))
    for heading in (
        "## 1. Runs",
        "## 2. Method in brief",
        "## 3. Detection results",
        "## 4. Benchmarks",
        "## 5. Limitations",
    ):
        assert heading in report
    assert "2/2" not in report  # counts appear in their own column, not as fractions
    assert "| tiny | f1 | P2 | S0 | F1(value=0) |" in report
    assert "os-csprng" in report
    assert all(item in report for item in LIMITATIONS)


def test_report_flags_development_runs_and_never_claims_a_result(
    results_dir: Path, tmp_path: Path
) -> None:
    index = _index(results_dir, tmp_path)
    runs = index.runs(kinds=["detection"])
    report = build_markdown_report(runs, index.cells(run_ids=[runs[0].id]), [])
    assert "Development runs included" in report
    lowered = report.lower()
    for claim in ("proven", "hypothesis is supported", "confirms that", "secure source"):
        assert claim not in lowered


def test_empty_report_is_still_well_formed() -> None:
    report = build_markdown_report([], [], [])
    assert "_No runs selected._" in report
    assert "_No detection cells selected._" in report
    assert "_No benchmark runs selected._" in report


def test_table_cells_escape_pipes() -> None:
    from entropylens.reporting.report_builder import _table

    assert "a\\|b" in _table(["x"], [["a|b"]])
