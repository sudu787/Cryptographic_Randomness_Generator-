"""Overview: the project, its generators, and the latest state of this session and the history."""

from __future__ import annotations

import streamlit as st

from entropylens.dashboard import components as ui
from entropylens.dashboard import state
from entropylens.dashboard.catalogue import GENERATORS

RESEARCH_QUESTION = (
    "Can a lightweight, explainable entropy-source monitoring layer detect injected entropy-source "
    "failures with an acceptable false-alarm rate and performance overhead, while clearly "
    "distinguishing observable statistical anomalies from cryptographic security guarantees?"
)


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "EntropyLens",
        "An auditable cryptographic randomness generator and entropy-health research platform.",
    )
    st.markdown(f"> {RESEARCH_QUESTION}")
    st.markdown(
        "Security-relevant randomness comes only from the operating system's CSPRNG. The research "
        "path injects labelled faults into **simulated** noise sources, monitors the raw samples "
        "with SP 800-90B-style health tests and supplementary detectors, and measures detection "
        "rate, latency, false alarms and overhead with uncertainty intervals."
    )

    index, _ = state.synced_history()
    runs = index.runs()
    detection_runs = [r for r in runs if r.kind in ("detection", "placement")]
    benchmark_runs = [r for r in runs if r.kind == "benchmark"]
    status, samples = state.monitor_status()

    tiles = st.columns(4)
    tiles[0].metric("Indexed runs", len(runs))
    latest = detection_runs[0] if detection_runs else None
    tiles[1].metric(
        "Latest experiment",
        latest.name if latest else "none yet",
        help=None if latest is None else f"{latest.kind}, {latest.purpose}, {latest.started_utc}",
    )
    tiles[2].metric("Samples monitored this session", f"{samples:,}")
    with tiles[3]:
        st.caption("Source health (this session)")
        ui.status_badge(status)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Generator modes")
        for choice in GENERATORS:
            info = choice.factory(1).info
            with st.container(border=True):
                ui.security_badge(info)
                st.markdown(f"**{choice.label}**")
                st.caption(info.backend)
    with right:
        st.subheader("Latest experiment")
        if latest is None:
            st.info("No experiment results indexed yet. Try `python -m entropylens run smoke`.")
        else:
            cells = index.cells(run_ids=[latest.id])
            if latest.purpose != "evaluation":
                st.warning(
                    f"'{latest.name}' is a {latest.purpose} run, not an evaluation result.",
                    icon=":material/construction:",
                )
            for cell in cells[:8]:
                st.markdown(
                    f"- `{cell.cell_name}` ({cell.pipeline}): {cell.detected}/{cell.trials} "
                    f"detected, rate {ui.interval_text(cell.rate, cell.rate_low, cell.rate_high)}"
                )
        st.subheader("Latest benchmark")
        if not benchmark_runs:
            st.info("No benchmark indexed yet. Try `python -m entropylens benchmark --quick`.")
        else:
            rows = index.benchmarks(run_ids=[benchmark_runs[0].id])
            for row in [r for r in rows if r.section == "generator"][:6]:
                st.markdown(
                    f"- `{row.subject}` at {row.size:,} B: {row.throughput_mib_s or 0:,.1f} MiB/s"
                )
            ratios = [r for r in rows if r.section == "pipeline" and r.subject == "end-to-end"]
            if ratios:
                worst = min(ratios, key=lambda r: r.ratio or 0)
                st.caption(
                    f"Monitoring overhead: P2/P0 throughput ratio as low as {worst.ratio:.3f} "
                    f"(buffer {worst.size:,} samples); the H3 target is 0.5."
                )
        st.subheader("Recent alerts (this session)")
        alerts = state.recent_alerts()
        if not alerts:
            st.caption("No alerts yet. Run the Fault Injection Lab or the Health Monitor.")
        for alert in alerts[:3]:
            ui.alert_card(alert, compact=True)

    ui.limitations(
        "This is a research prototype, not a certified or NIST-compliant security product.",
        "Monitoring results describe simulated sources under stated models; they are never "
        "evidence that a randomness source is secure.",
    )
