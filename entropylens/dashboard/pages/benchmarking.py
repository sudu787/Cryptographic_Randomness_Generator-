"""Benchmarking: generator throughput and latency, and the cost of monitoring (E6, H3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entropylens.analysis import benchmark as bm
from entropylens.dashboard import components as ui
from entropylens.dashboard import state, theme
from entropylens.experiments.synthetic_sources import S0
from entropylens.storage.repository import write_result
from entropylens.storage.schemas import RESULT_SCHEMA_VERSION, ResultDocument, RunMetadata

SIZES = [64, 1024, 16 * 1024, 64 * 1024, 1 << 20]


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Benchmarking",
        "Throughput and latency of each generator, and the overhead of monitoring, measured with "
        "warm-up, repetition and recorded machine metadata.",
    )
    factories = bm.default_generators()
    left, right = st.columns(2, gap="large")
    with left:
        chosen = st.multiselect(
            "Generators", list(factories), default=list(factories), key="bm_generators"
        )
        sizes = st.multiselect(
            "Buffer sizes (bytes)", SIZES, default=[1024, 64 * 1024], key="bm_sizes"
        )
    with right:
        repetitions = st.slider("Measured repetitions per cell", 3, 50, 10, key="bm_reps")
        overhead = st.toggle("Also measure monitoring overhead (P2 vs P0)", True, key="bm_overhead")
        seed = st.number_input("Order and bootstrap seed", 0, 2**31 - 1, 0, key="bm_seed")

    if st.button("Run benchmark", type="primary", key="bm_run", disabled=not (chosen and sizes)):
        with st.spinner("Measuring..."):
            started = state_now()
            generators = bm.benchmark_generators(
                {k: factories[k] for k in chosen},
                sorted(sizes),
                repetitions=int(repetitions),
                warmup=2,
                order_seed=int(seed),
            )
            pipeline = (
                bm.benchmark_pipeline(
                    S0,
                    [s for s in sorted(sizes) if s >= 1024] or [1024],
                    repetitions=int(repetitions),
                    warmup=2,
                    seed=int(seed),
                )
                if overhead
                else []
            )
        st.session_state["bm_last"] = (started, int(seed), int(repetitions), generators, pipeline)
    last = st.session_state.get("bm_last")
    if last is None:
        st.info(
            "Choose generators and sizes, then press **Run benchmark**.", icon=":material/speed:"
        )
    else:
        _show(*last)
    ui.limitations(
        "Throughput is not a security metric. The fastest generator is not the most secure.",
        "APIs differ: secrets.token_bytes makes a system call per request, ChaCha20 runs in "
        "OpenSSL, and the LCG runs in Python or NumPy. Results describe this machine and stack.",
        "Interactive runs use few repetitions; the reported E6 results come from the full CLI run.",
    )


def state_now() -> str:
    """Current UTC timestamp (ISO 8601) for result metadata."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _show(
    started: str,
    seed: int,
    repetitions: int,
    generators: list[bm.GeneratorBenchmark],
    pipeline: list[bm.PipelineBenchmark],
) -> None:
    rows = []
    for g in generators:
        s = bm.summarize_timing(g.timing, seed=seed)
        rows.append(
            {
                "generator": g.generator,
                "security status": g.security_status,
                "buffer (bytes)": g.buffer_bytes,
                "median latency (us)": round(s.median_us, 2),
                "p95 latency (us)": round(s.p95_us, 2),
                "IQR (us)": round(s.iqr_us, 2),
                "CPU share": round(s.cpu_fraction, 3),
                "throughput MiB/s": round(s.throughput_mib_s.estimate, 2),
                "throughput 95% CI": (
                    f"[{s.throughput_mib_s.low:.2f}, {s.throughput_mib_s.high:.2f}]"
                ),
                "peak heap (bytes)": g.peak_memory_bytes,
            }
        )
    frame = pd.DataFrame(rows)
    fig = theme.figure(
        "Generator throughput by buffer size", "Buffer size (bytes, log scale)", "Median MiB/s"
    )
    for name in dict.fromkeys(frame["generator"]):
        part = frame[frame["generator"] == name]
        fig.add_trace(
            go.Scatter(
                x=part["buffer (bytes)"],
                y=part["throughput MiB/s"],
                mode="lines+markers",
                name=name,
                line={"color": theme.GENERATOR_COLOURS.get(name, theme.INK)},
                hovertemplate=f"{name}<br>%{{x:,}} B: %{{y:,.1f}} MiB/s<extra></extra>",
            )
        )
    fig.update_xaxes(type="log")
    ui.chart(fig, key="bm_throughput")
    st.dataframe(frame, hide_index=True, width="stretch")

    if pipeline:
        st.subheader("Monitoring overhead (H3: monitored throughput at least half of unmonitored)")
        ratio_rows = [
            {
                "variant": p.variant,
                "buffer (samples)": p.buffer_samples,
                "P2/P0 throughput ratio": round(p.throughput_ratio.estimate, 4),
                "95% CI": f"[{p.throughput_ratio.low:.4f}, {p.throughput_ratio.high:.4f}]",
            }
            for p in pipeline
        ]
        ratios = pd.DataFrame(ratio_rows)
        fig = theme.figure(
            "Throughput with monitoring relative to without",
            "Buffer size (samples, log scale)",
            "P2 / P0 throughput",
        )
        for index, variant in enumerate(dict.fromkeys(ratios["variant"])):
            part = [p for p in pipeline if p.variant == variant]
            fig.add_trace(
                go.Scatter(
                    x=[p.buffer_samples for p in part],
                    y=[p.throughput_ratio.estimate for p in part],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            p.throughput_ratio.high - p.throughput_ratio.estimate for p in part
                        ],
                        "arrayminus": [
                            p.throughput_ratio.estimate - p.throughput_ratio.low for p in part
                        ],
                        "thickness": 1,
                    },
                    mode="lines+markers",
                    name=variant,
                    line={"color": theme.CATEGORICAL[index]},
                )
            )
        fig.add_hline(y=0.5, line_color=theme.MUTED, line_width=1)
        fig.add_annotation(
            x=1,
            xref="paper",
            y=0.5,
            text="H3 target 0.5",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": theme.INK_SECONDARY, "size": 11},
        )
        fig.update_xaxes(type="log")
        ui.chart(fig, key="bm_ratio")
        st.dataframe(ratios, hide_index=True, width="stretch")

    with st.expander("Experiment conditions and machine metadata", icon=":material/memory:"):
        metadata = bm.machine_metadata()
        st.dataframe(
            pd.DataFrame({"field": list(metadata), "value": [str(v) for v in metadata.values()]}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"{repetitions} measured repetitions per cell after 2 warm-up calls; seed {seed}."
        )

    if st.button("Save this run to the experiment history", key="bm_save"):
        payload = bm.benchmark_payload(
            generators,
            pipeline,
            seed=seed,
            extra_notes=["Interactive dashboard run with few repetitions."],
        )
        document = ResultDocument(
            metadata=RunMetadata(
                schema_version=RESULT_SCHEMA_VERSION,
                kind="benchmark",
                name="dashboard-benchmark",
                started_utc=started,
                finished_utc=state_now(),
                command="dashboard benchmark",
                machine=bm.machine_metadata(),
            ),
            config={"repetitions": repetitions, "seed": seed, "source": "dashboard"},
            payload=payload,
        )
        path, digest = write_result(document, state.results_dir())
        state.history().sync(state.results_dir())
        st.success(f"Saved {path.name} (sha256 {digest[:16]}...).", icon=":material/save:")
