"""Experiment History: indexed result files, filters, comparison, integrity and exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd
import streamlit as st

from entropylens.dashboard import components as ui
from entropylens.dashboard import state
from entropylens.storage.history import ExperimentHistory, RunRow, to_csv, to_json


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Experiment History",
        "Every result file in the results directory, indexed in SQLite with its SHA-256. "
        "Filter, compare and export without losing reproducibility metadata.",
    )
    index, report = state.synced_history()
    _sync_messages(report.added, report.modified, report.missing, report.invalid)
    _browse(index)
    ui.limitations(
        "Runs whose purpose is 'development' are smoke or exploratory runs, not evaluation "
        "results.",
        "The SHA-256 integrity check detects accidental changes to result files; it is not a "
        "signature and cannot prove a file was not deliberately altered.",
    )


def _browse(index: ExperimentHistory) -> None:
    st.subheader("Filters")
    row = st.columns(4)
    kinds = row[0].multiselect("Run kind", ["detection", "placement", "benchmark"], key="hi_kinds")
    name = row[1].text_input("Name contains", max_chars=64, key="hi_name")
    since = row[2].date_input("Started on or after", value=None, key="hi_since")
    until = row[3].date_input("Started on or before", value=None, key="hi_until")
    row = st.columns(3)
    models = row[0].multiselect("Source model", index.distinct("model"), key="hi_models")
    faults = row[1].multiselect("Scenario", index.distinct("fault"), key="hi_faults")
    subjects = row[2].multiselect(
        "Generator / overhead variant", index.distinct("subject"), key="hi_subjects"
    )

    runs = index.runs(
        kinds=kinds or None,
        name_contains=name or None,
        since=since.isoformat() if since else None,
        until=f"{until.isoformat()}T23:59:59" if until else None,
    )
    run_ids = [r.id for r in runs]
    cells = (
        index.cells(run_ids=run_ids, models=models or None, faults=faults or None)
        if run_ids
        else []
    )
    benchmarks = index.benchmarks(run_ids=run_ids, subjects=subjects or None) if run_ids else []

    st.subheader(f"Runs ({len(runs)})")
    if not runs:
        st.info(
            "No result files match. Create some with `python -m entropylens run smoke` or "
            "`python -m entropylens benchmark --quick`.",
            icon=":material/history:",
        )
        return
    st.dataframe(pd.DataFrame([asdict(r) for r in runs]), hide_index=True, width="stretch")
    _downloads("runs", runs)

    st.subheader(f"Cell results ({len(cells)})")
    if cells:
        frame = pd.DataFrame([asdict(c) for c in cells])
        st.dataframe(frame, hide_index=True, width="stretch")
        _downloads("cells", cells)
    else:
        st.caption("No detection or placement cells in the selection.")

    st.subheader(f"Benchmarks ({len(benchmarks)})")
    if benchmarks:
        st.dataframe(
            pd.DataFrame([asdict(b) for b in benchmarks]), hide_index=True, width="stretch"
        )
        _downloads("benchmarks", benchmarks)
    else:
        st.caption("No benchmark rows in the selection.")

    _compare(runs, index)
    _integrity(index)


def _sync_messages(
    added: list[str], modified: list[str], missing: list[str], invalid: dict[str, str]
) -> None:
    if added:
        st.toast(f"Indexed {len(added)} new result file(s).", icon=":material/add:")
    if modified:
        st.error(
            "Changed since indexing (not re-imported): " + ", ".join(modified),
            icon=":material/gpp_bad:",
        )
    if missing:
        st.warning("Indexed but no longer present: " + ", ".join(missing), icon=":material/help:")
    for name, reason in invalid.items():
        st.warning(f"Skipped invalid file {name}: {reason}", icon=":material/block:")


def _downloads(label: str, rows: Sequence[object]) -> None:
    left, right, _ = st.columns([1, 1, 4])
    left.download_button(
        f"{label}.csv",
        to_csv(rows),
        file_name=f"entropylens-{label}.csv",
        mime="text/csv",
        key=f"hi_csv_{label}",
        icon=":material/download:",
    )
    right.download_button(
        f"{label}.json",
        to_json(rows),
        file_name=f"entropylens-{label}.json",
        mime="application/json",
        key=f"hi_json_{label}",
        icon=":material/download:",
    )


def _compare(runs: list[RunRow], index: ExperimentHistory) -> None:
    comparable = [r for r in runs if r.kind in ("detection", "placement")]
    if len(comparable) < 2:
        return
    st.subheader("Compare two runs")
    labels = {r.id: f"{r.name} ({r.started_utc})" for r in comparable}
    left, right = st.columns(2)
    run_a = left.selectbox("Run A", list(labels), format_func=labels.__getitem__, key="hi_a")
    run_b = right.selectbox(
        "Run B", list(labels), index=1, format_func=labels.__getitem__, key="hi_b"
    )
    rows = index.compare(run_a, run_b)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Cells are matched by name and pipeline. The difference is rate B minus rate A.")


def _integrity(index: ExperimentHistory) -> None:
    with st.expander("Integrity check", icon=":material/verified:"):
        st.caption(
            "Recomputes each file's SHA-256 and compares it with the value recorded at import. "
            "This detects accidental changes; it is not a signature."
        )
        if st.button("Verify all files", key="hi_verify"):
            status = index.verify(state.results_dir())
            st.dataframe(
                pd.DataFrame({"file": list(status), "status": list(status.values())}),
                hide_index=True,
                width="stretch",
            )
