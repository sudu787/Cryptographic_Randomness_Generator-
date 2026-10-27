"""Research Report: a structured report of selected runs, with Markdown download."""

from __future__ import annotations

import streamlit as st

from entropylens.dashboard import components as ui
from entropylens.dashboard import state
from entropylens.reporting.report_builder import build_markdown_report


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Research Report",
        "Methodology, configuration, measurements with intervals, and limitations for the runs "
        "you select. Hypotheses are judged by the pre-declared rules, never automatically.",
    )
    index, _ = state.synced_history()
    runs = index.runs()
    if not runs:
        st.info("No result files indexed yet. Run an experiment first.", icon=":material/article:")
        return
    labels = {r.id: f"{r.name} ({r.kind}, {r.purpose or 'n/a'}, {r.started_utc})" for r in runs}
    chosen = st.multiselect(
        "Runs to include",
        list(labels),
        default=[runs[0].id],
        format_func=labels.__getitem__,
        key="rp_runs",
    )
    selected = [r for r in runs if r.id in chosen]
    ids = [r.id for r in selected]
    report = build_markdown_report(
        selected,
        index.cells(run_ids=ids) if ids else [],
        index.benchmarks(run_ids=ids) if ids else [],
    )
    st.download_button(
        "Download report (Markdown)",
        report,
        file_name="entropylens-report.md",
        mime="text/markdown",
        key="rp_download",
        icon=":material/download:",
    )
    st.caption("HTML and PDF export arrive in Phase 6.")
    with st.container(border=True):
        st.markdown(report)
