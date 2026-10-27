"""Entropy and Health Monitor: the detectors, their thresholds, and a monitored stream over time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from entropylens.dashboard import components as ui
from entropylens.dashboard import plots, state
from entropylens.dashboard.controls import fault_control, model_control, monitor_control
from entropylens.dashboard.simulation import MonitorRun, lab_spec, run_lab_trace
from entropylens.experiments.evaluation import CellSpec
from entropylens.experiments.synthetic_sources import MODELS
from entropylens.monitoring.detectors import RepetitionCountTest
from entropylens.monitoring.source_health import build_detectors


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Entropy & Health Monitor",
        "Continuous health tests on the raw samples of a simulated noise source, before any "
        "conditioning. Every threshold follows from the false-alarm budget and the model.",
    )
    left, middle, right = st.columns(3, gap="large")
    with left:
        model = model_control("hm")
        fault = fault_control("hm", model)
    with middle:
        settings = monitor_control("hm")
    with right:
        buffers = st.number_input("Buffers to stream", 4, 128, 32, key="hm_buffers")
        buffer = st.select_slider(
            "Buffer size (samples)", [4096, 16384, 65536], value=16384, key="hm_buffer"
        )
        seed = st.number_input("Seed (development)", 0, 2**31 - 1, 7, key="hm_seed")
    length = int(buffers) * int(buffer)
    spec = lab_spec(
        model,
        fault,
        warmup=max(4096, length // 2),
        post_onset=max(1, length - max(4096, length // 2)),
        buffer=int(buffer),
        settings=settings,
    )

    _thresholds(spec)

    if st.button("Stream through the monitor", type="primary", key="hm_run"):
        run = run_lab_trace(spec, int(seed))
        st.session_state["hm_last"] = (spec, run)
        state.record_monitor_result(run.statuses[-1], run.trace.samples.size, run.alerts[:3])
    if "hm_last" in st.session_state:
        _stream(*st.session_state["hm_last"])
    else:
        st.info(
            "Configure the source and press **Stream through the monitor**.",
            icon=":material/monitor_heart:",
        )

    st.subheader("Source failures versus output anomalies")
    st.markdown(
        "- **Source health tests** (this page) examine raw noise samples against the model the "
        "source was assessed under. A failure means the source no longer behaves as claimed, so "
        "the gate closes before any output is derived from it.\n"
        "- **Output-level statistics** (Statistical Analysis page) examine bytes after "
        "conditioning. Conditioning hides most source defects, so a clean output test says "
        "little about the source.\n"
        "- **Availability failures** (exceptions, short reads) are reported by the source itself "
        "and are critical immediately."
    )
    ui.limitations(
        "A healthy status means none of the targeted failure modes was seen at the configured "
        "false-alarm budget. It does not show the claimed entropy is present.",
        "A source controlled by an adversary can emit data that passes every health test.",
        "Warnings come from exploratory detectors with approximate null distributions; some are "
        "expected by chance.",
    )


def _thresholds(spec: CellSpec) -> None:
    detectors, disabled = build_detectors(MODELS[spec.model], spec.monitor)
    rows = [
        {
            "detector": d.spec.title,
            "level": d.spec.level.value,
            "alpha per evaluation": d.spec.alpha,
            "exact false-alarm probability": d.spec.false_alarm_probability,
            "threshold and derivation": d.spec.threshold,
            "null model": d.spec.null_model,
            "targets": ", ".join(d.spec.targets),
        }
        for d in detectors
    ]
    with st.expander("Detectors and threshold derivations", expanded=False, icon=":material/rule:"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if disabled:
            st.caption(
                "Not applied: "
                + "; ".join(f"{name} ({reason})" for name, reason in disabled.items())
            )
        st.caption(
            "The APT's exact false-alarm probability can exceed alpha: the standard's cutoff "
            "formula does not count the window's first sample (methodology section 4.1)."
        )


def _stream(spec: CellSpec, run: MonitorRun) -> None:
    model = MODELS[spec.model]
    tiles = st.columns(3)
    tiles[0].metric("Samples monitored", f"{run.trace.samples.size:,}")
    tiles[1].metric("Alerts raised", f"{len(run.alerts):,}")
    with tiles[2]:
        st.caption("Status after the last buffer")
        ui.status_badge(run.statuses[-1])

    ui.chart(plots.status_timeline(run), key="hm_status")
    ui.chart(plots.signal_figure(run, model.kind == "binary", model.p_one), key="hm_signal")
    rct = next((d for d in build_detectors(model, spec.monitor)[0] if d.spec.name == "rct"), None)
    cutoff = rct.cutoff if isinstance(rct, RepetitionCountTest) else None
    ui.chart(plots.repetition_figure(run, cutoff), key="hm_repetition")

    st.subheader("Event history")
    if not run.alerts:
        st.success("No alerts in this stream.", icon=":material/check_circle:")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "sample": a.sample_index,
                    "detector": a.detector,
                    "outcome": a.outcome.value,
                    "explanation": a.message,
                }
                for a in run.alerts
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    for alert in run.alerts[:3]:
        ui.alert_card(alert)
