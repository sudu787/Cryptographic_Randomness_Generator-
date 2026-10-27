"""Statistical Analysis: the SP 800-22 subset, entropy metrics and diagnostics, and their limits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entropylens.analysis import entropy_metrics as em
from entropylens.analysis import statistical_tests as st_tests
from entropylens.dashboard import components as ui
from entropylens.dashboard import theme
from entropylens.dashboard.catalogue import BY_KEY, GENERATORS
from entropylens.dashboard.controls import fault_control
from entropylens.experiments.conditioning import sha256_condition
from entropylens.experiments.fault_injection import FaultBase, inject
from entropylens.experiments.synthetic_sources import S0

SOURCES = {
    **{g.key: g.label for g in GENERATORS},
    "conditioned": "SHA-256-conditioned output of a simulated source (optionally faulty)",
}


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Statistical Analysis",
        "Output-level tests (SP 800-22 subset) and descriptive metrics. Useful for spotting gross "
        "defects; unable to show that data is unpredictable.",
    )
    left, right = st.columns([2, 1], gap="large")
    with left:
        source = st.selectbox(
            "Data source", list(SOURCES), format_func=SOURCES.__getitem__, key="sa_source"
        )
        fault = fault_control("sa", "S0") if source == "conditioned" else None
    with right:
        bits = st.select_slider(
            "Sample size (bits)",
            options=[1 << 13, 1 << 15, 1 << 17, 1 << 19, 1 << 20],
            value=1 << 17,
            format_func=lambda b: f"{b:,}",
            key="sa_bits",
        )
        family_alpha = st.select_slider(
            "Family-wise significance level", [0.001, 0.01, 0.05], value=0.01, key="sa_alpha"
        )
        seed = st.number_input("Seed (deterministic sources)", 0, 2**31 - 1, 3, key="sa_seed")
    selected = st.multiselect(
        "Tests",
        list(st_tests.TESTS),
        default=list(st_tests.TESTS),
        format_func=lambda name: st_tests.TESTS[name].title,
        key="sa_tests",
    )

    if st.button("Run analysis", type="primary", key="sa_run"):
        data = _data(source, fault, int(bits) // 8, int(seed))
        if len(data) < 1024:
            st.warning(
                "Too little data to analyse: a source outage delivers no samples after the onset.",
                icon=":material/block:",
            )
        else:
            st.session_state["sa_last"] = _analyse(data, selected, float(family_alpha))
    last = st.session_state.get("sa_last")
    if last is None:
        st.info("Choose a source and press **Run analysis**.", icon=":material/query_stats:")
    else:
        _show(last)
    ui.limitations(
        "A p-value above the threshold is not evidence of randomness; it only means this test "
        "found no deviation of the kind it looks for, at this sample size.",
        "Passing every test does not show unpredictability. A generator with a published key "
        "(try 'ChaCha20, published key') passes.",
        "Plug-in entropy and compression ratios describe this sample, not the source, and are "
        "heuristic.",
    )


def _data(source: str, fault: FaultBase | None, n_bytes: int, seed: int) -> bytes:
    if source != "conditioned":
        return BY_KEY[source].factory(seed).generate(n_bytes)
    raw_samples = n_bytes * 8 * 2  # 512 raw samples -> 256 output bits
    rng = np.random.default_rng(seed)
    healthy = S0.generate(rng, raw_samples + 4096)
    if fault is not None:
        healthy = inject(healthy, "binary", fault, 4096, rng, lambda n: S0.generate(rng, n)).samples
    return sha256_condition(healthy[4096:], "binary")[:n_bytes]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything the page shows for one analysed sample (no raw data)."""

    rows: list[dict[str, object]]
    rejected: list[str]
    family_alpha: float
    n_bits: int
    histogram: npt.NDArray[np.int64]
    autocorrelation: npt.NDArray[np.float64]
    entropy: float
    pattern: float
    mcv: em.MinEntropyEstimate
    compression: em.CompressionHeuristic


def _analyse(data: bytes, selected: list[str], family_alpha: float) -> AnalysisResult:
    battery = st_tests.run_battery(data, family_alpha)
    rows: list[dict[str, object]] = [
        {
            "test": st_tests.TESTS[r.name].title,
            "statistic": round(r.statistic, 4),
            "p-value": float(f"{r.p_value:.3g}"),
            "Holm-adjusted p": None if adjusted is None else float(f"{adjusted:.3g}"),
            "applicable": r.applicable,
            "decision": "reject H0"
            if r.name in battery.rejected
            else ("no rejection" if r.applicable else "not interpreted"),
            "note": r.note,
        }
        for r, adjusted in zip(battery.results, battery.adjusted_p_values, strict=True)
        if r.name in selected
    ]
    bits = st_tests.as_bits(data)
    return AnalysisResult(
        rows=rows,
        rejected=[st_tests.TESTS[n].title for n in battery.rejected if n in selected],
        family_alpha=family_alpha,
        n_bits=int(bits.size),
        histogram=em.byte_histogram(data),
        autocorrelation=em.autocorrelation(bits, 64),
        entropy=em.shannon_entropy(data),
        pattern=em.pattern_entropy(bits, 8),
        mcv=em.most_common_value_estimate(data),
        compression=em.compression_ratio(data),
    )


def _show(last: AnalysisResult) -> None:
    st.dataframe(pd.DataFrame(last.rows), hide_index=True, width="stretch")
    if last.rejected:
        st.error(
            f"At family-wise level {last.family_alpha}, H0 (independent, unbiased bits) is "
            f"rejected by: {', '.join(last.rejected)}. The data shows a detectable statistical "
            "defect.",
            icon=":material/error:",
        )
    else:
        st.success(
            f"No test rejected H0 at family-wise level {last.family_alpha}. This means only that "
            "these tests found nothing at this sample size; it does not show the data is random "
            "or unpredictable.",
            icon=":material/check:",
        )

    tiles = st.columns(4)
    tiles[0].metric("Byte entropy (plug-in)", f"{last.entropy:.4f} bits/byte")
    tiles[1].metric("8-bit pattern entropy", f"{last.pattern:.4f} bits")
    tiles[2].metric(
        "MCV min-entropy estimate",
        f"{last.mcv.min_entropy:.3f} bits/byte",
        help="SP 800-90B 6.3.1 estimator applied to output bytes, for comparison only.",
    )
    tiles[3].metric(
        "LZMA size ratio (heuristic)",
        f"{last.compression.lzma_ratio:.3f}",
        help="Below 1 means the compressor found redundancy. Near 1 shows only that it found none.",
    )

    fig = theme.figure("Byte value frequencies", "Byte value", "Count", height=300)
    fig.add_trace(
        go.Bar(
            x=np.arange(256),
            y=last.histogram,
            name="count",
            marker={"color": theme.CATEGORICAL[0]},
            hovertemplate="byte %{x}<br>count %{y}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.05, showlegend=False)
    ui.chart(fig, key="sa_histogram")

    r = last.autocorrelation
    lags = np.arange(1, r.size + 1)
    band = 1.96 / math.sqrt(last.n_bits)
    fig = theme.figure("Autocorrelation of the bit sequence", "Lag (bits)", "r_k", height=300)
    fig.add_hrect(y0=-band, y1=band, fillcolor=theme.MUTED, opacity=0.15, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=r,
            mode="markers+lines",
            name="r_k",
            line={"color": theme.CATEGORICAL[0]},
            marker={"size": 8},
            hovertemplate="lag %{x}<br>r = %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    ui.chart(fig, key="sa_autocorrelation")
    st.caption(
        f"Shaded band: pointwise 95% range for independent bits (+/- {band:.4f}). With 64 lags, "
        "a few points outside the band are expected by chance; this plot is exploratory."
    )
    ui.table_view(pd.DataFrame({"lag": lags, "r_k": r}), "Autocorrelation table")
