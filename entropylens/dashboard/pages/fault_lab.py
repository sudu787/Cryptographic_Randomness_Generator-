"""Fault Injection Lab: the demonstration centrepiece.

Make a simulated source fail on purpose; see which detector notices, after how many samples, why,
and how much faulty data would have escaped without monitoring. Then repeat the trial to turn the
anecdote into a detection rate with an interval.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from entropylens.dashboard import components as ui
from entropylens.dashboard import plots, state
from entropylens.dashboard.controls import fault_control, model_control, monitor_control
from entropylens.dashboard.simulation import (
    INTERACTIVE_BLOCK_BITS,
    MonitorRun,
    PipelineComparison,
    compare_pipelines,
    lab_spec,
    run_lab_trace,
)
from entropylens.experiments.conditioning import CONDITIONING_INPUT_SAMPLES, DIGEST_BYTES
from entropylens.experiments.evaluation import AVAILABILITY, CellSpec, run_cell, summarize
from entropylens.experiments.synthetic_sources import MODELS
from entropylens.monitoring.alerts import DetectorLevel
from entropylens.monitoring.detectors import RepetitionCountTest
from entropylens.monitoring.source_health import build_detectors


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Fault Injection Lab",
        "Inject a labelled fault into a simulated source. See which detector fires, how fast, why, "
        "and what would have escaped without monitoring.",
    )
    spec, seed = _controls()
    single, repeated = st.tabs(["One trace, explained", "Repeated trials"])
    with single:
        if st.button("Inject fault and run", type="primary", key="fl_run"):
            run = run_lab_trace(spec, seed)
            st.session_state["fl_last"] = (spec, run, compare_pipelines(spec, run.trace))
            state.record_monitor_result(run.statuses[-1], run.trace.samples.size, run.alerts[:3])
        if "fl_last" in st.session_state:
            _single_result(*st.session_state["fl_last"])
        else:
            st.info(
                "Choose a scenario and press **Inject fault and run**.", icon=":material/science:"
            )
    with repeated:
        _repeated_trials(spec, seed)
    ui.limitations(
        "Faults are simulated models of failures; real hardware fails in ways this taxonomy may "
        "not contain.",
        "Detection here is relative to the nominal model. A source producing statistically perfect "
        "but predictable output (scenario N1 or F8b) is not detected, and passing the monitor is "
        "not evidence of security.",
        "Interactive runs use the seed typed above, a development seed. Reported results come from "
        "the pre-registered evaluation configurations only.",
    )


def _controls() -> tuple[CellSpec, int]:
    left, middle, right = st.columns(3, gap="large")
    with left:
        model = model_control("fl")
        fault = fault_control("fl", model)
    with middle:
        warmup = st.number_input(
            "Healthy samples before the fault", 8192, 1 << 20, 1 << 16, step=4096, key="fl_warmup"
        )
        post = st.number_input(
            "Samples after the onset", 4096, 1 << 21, 1 << 18, step=4096, key="fl_post"
        )
        buffer = st.select_slider(
            "Monitor buffer (samples)", [4096, 16384, 65536], value=16384, key="fl_buffer"
        )
    with right:
        settings = monitor_control("fl")
        seed = st.number_input("Seed (development)", 0, 2**31 - 1, 1, key="fl_seed")
    spec = lab_spec(
        model,
        fault,
        warmup=int(warmup),
        post_onset=int(post),
        buffer=int(buffer),
        settings=settings,
    )
    return spec, int(seed)


def _single_result(spec: CellSpec, run: MonitorRun, comparison: PipelineComparison) -> None:
    model = MODELS[spec.model]
    onset = run.trace.onset
    first = run.record.first_alarm(
        [n for n in run.record.detectors if n not in ("autocorrelation", "byte-chi-square")],
        after_onset=True,
    )
    tiles = st.columns(4)
    tiles[0].metric("Fault onset (sample)", f"{onset:,}")
    tiles[1].metric(
        "First critical alarm",
        "none" if first is None else f"{first:,}",
        help="Earliest alarm at or after the onset among critical detectors.",
    )
    tiles[2].metric(
        "Detection latency (samples)", "not detected" if first is None else f"{first - onset:,}"
    )
    with tiles[3]:
        st.caption("Final source status")
        ui.status_badge(run.statuses[-1])

    ui.chart(plots.signal_figure(run, model.kind == "binary", model.p_one), key="fl_signal")
    rct = next((d for d in build_detectors(model, spec.monitor)[0] if d.spec.name == "rct"), None)
    cutoff = rct.cutoff if isinstance(rct, RepetitionCountTest) else None
    ui.chart(plots.repetition_figure(run, cutoff), key="fl_repetition")

    rows = []
    levels = {d.spec.name: d.spec.level for d in build_detectors(model, spec.monitor)[0]}
    for name, outcome in run.record.detectors.items():
        after = outcome.first_alarm_after
        rows.append(
            {
                "detector": name,
                "level": levels.get(name, DetectorLevel.CRITICAL).value,
                "first alarm after onset": after,
                "latency (samples)": None if after is None else after - onset,
                "alarms before onset": outcome.alarms_before,
                "evaluations before onset": outcome.evaluations_before,
            }
        )
    st.subheader("Which detector saw it")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if run.disabled:
        st.caption(
            "Not applied to this model: "
            + "; ".join(f"{name} ({reason})" for name, reason in run.disabled.items())
        )

    st.subheader("Why the monitor raised the alarm")
    if run.failure is not None:
        ui.alert_card(run.failure)
    else:
        st.warning(
            "No critical alarm. The monitor did not detect this fault within the trace. That is "
            "itself a result: see which statistics the fault leaves unchanged.",
            icon=":material/visibility_off:",
        )

    st.subheader("Monitoring on versus off, same trace")
    _comparison(comparison, onset)


def _comparison(comparison: PipelineComparison, onset: int) -> None:
    def conditioned(samples: int | None) -> str:
        blocks = (samples or 0) // CONDITIONING_INPUT_SAMPLES
        return f"{blocks * DIGEST_BYTES:,} B"

    off, on = st.columns(2, gap="large")
    with off, st.container(border=True):
        st.markdown("**:material/visibility_off: Without monitoring (P0)**")
        exposed = comparison.unmonitored.released_faulty_samples
        st.metric("Faulty raw samples released", f"{exposed or 0:,}")
        st.metric("Conditioned output derived from them", conditioned(exposed))
        outcome = comparison.output_tests
        if outcome.first_rejection_after is not None:
            verdict = (
                f"flagged the block ending at sample {outcome.first_rejection_after:,} "
                f"({', '.join(outcome.rejected_tests)})"
            )
        else:
            verdict = (
                f"passed all {outcome.blocks_tested} blocks of {INTERACTIVE_BLOCK_BITS:,} bits"
            )
        st.caption(f"Output tests on the conditioned output (P1): {verdict}.")
    with on, st.container(border=True):
        st.markdown("**:material/visibility: With raw-source monitoring (P2)**")
        st.metric(
            "Faulty raw samples released", f"{comparison.monitored.released_faulty_samples or 0:,}"
        )
        st.metric("Raw samples withheld by the gate", f"{comparison.monitored.withheld_samples:,}")
        st.caption(
            "The gate closes for the buffer containing the first critical alarm and stays closed "
            "(latched) until an explicit reset."
        )
    if onset:
        st.caption(
            "Conditioning (SHA-256 over 512-sample blocks) makes faulty input look random, which "
            "is why output tests often pass. Health tests belong on the raw samples."
        )


def _repeated_trials(spec: CellSpec, seed: int) -> None:
    trials = st.number_input("Trials", 5, 200, 20, step=5, key="fl_trials")
    if st.button("Run repeated trials", key="fl_repeat"):
        cell = spec.model_copy(update={"trials": int(trials)})
        with st.spinner(f"Running {trials} trials..."):
            st.session_state["fl_repeat_result"] = run_cell(cell, seed, 0)
    result = st.session_state.get("fl_repeat_result")
    if result is None:
        st.info("Repeat the same scenario with independent seeds to estimate the detection rate.")
        return
    subsets = {"full monitor": list(result.critical), "RCT + APT only": ["rct", "apt"]}
    subsets.update({name: [name] for name in result.critical})
    rows = []
    for label, names in subsets.items():
        names = [n for n in names if n in result.critical]
        if not names:
            continue
        s = summarize(result, names)
        rows.append(
            {
                "detectors": label,
                "detected": f"{s.detected}/{s.trials}",
                "rate [95% CI]": ui.interval_text(
                    s.detection.estimate, s.detection.low, s.detection.high
                ),
                "median latency": s.latency_median,
                "p95 latency": s.latency_p95,
                "pre-onset false alarms": s.pre_onset_alarms,
                "mean exposure (samples)": round(s.exposure_mean_samples),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"{result.spec.trials} trials of "
        f"{result.spec.fault.fault if result.spec.fault else 'healthy'} "
        f"on {result.spec.model}, development seed {result.master_seed}. Intervals are "
        "Wilson (Clopper-Pearson at 0 or n). The ablation rows reuse the same trials. "
        f"'{AVAILABILITY}' is included in the full monitor."
    )
