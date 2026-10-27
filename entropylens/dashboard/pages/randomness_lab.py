"""Randomness Lab: generate bytes from each generator, and break the insecure ones."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entropylens.analysis.entropy_metrics import byte_histogram, shannon_entropy
from entropylens.dashboard import components as ui
from entropylens.dashboard import theme
from entropylens.dashboard.catalogue import BY_KEY, GENERATORS
from entropylens.security.educational_lcg import (
    DRAND48_HIGH32,
    NUMERICAL_RECIPES_32,
    EducationalLCG,
    LCGParameters,
    predict_outputs,
    recover_parameters,
    recover_seed,
    recover_truncated_state,
)

MAX_BYTES = 1 << 20
PREVIEW_BYTES = 32


def render() -> None:
    """Draw the page."""
    ui.page_header(
        "Randomness Lab",
        "Generate bytes from the production source and the labelled baselines, then see why "
        "deterministic generators with guessable seeds fail.",
    )
    generate, attack = st.tabs(["Generate", "Break the LCG"])
    with generate:
        _generate()
    with attack:
        _attacks()
    ui.limitations(
        "Throughput and an even byte histogram say nothing about unpredictability: the ChaCha20 "
        "stream under a published key looks perfect and is fully predictable.",
        "Bytes shown or downloaded here are demonstration data. Never use them as keys.",
    )


def _generate() -> None:
    left, right = st.columns([2, 1], gap="large")
    with left:
        key = st.selectbox(
            "Generator",
            [g.key for g in GENERATORS],
            format_func=lambda k: BY_KEY[k].label,
            key="rl_generator",
        )
    choice = BY_KEY[key]
    with right:
        seed = st.number_input(
            "Seed (deterministic generators)",
            0,
            2**53 - 1,
            20261027,
            disabled=not choice.seeded,
            key="rl_seed",
        )
    size = st.select_slider(
        "Output length (bytes)",
        options=[256, 1024, 4096, 16384, 65536, 262144, MAX_BYTES],
        value=4096,
        key="rl_size",
    )
    generator = choice.factory(int(seed))
    ui.security_badge(generator.info)
    st.caption(generator.info.backend)

    if st.button("Generate", type="primary", key="rl_generate"):
        start = time.perf_counter_ns()
        data = generator.generate(int(size))
        elapsed = max(1, time.perf_counter_ns() - start)
        # Only non-secret demonstration output is kept for display and download.
        st.session_state["rl_last"] = {
            "key": key,
            "size": len(data),
            "elapsed_ns": elapsed,
            "entropy": shannon_entropy(data),
            "histogram": byte_histogram(data),
            "bytes": data if choice.reveal_output else None,
        }
    last = st.session_state.get("rl_last")
    if not last:
        st.info("Press **Generate**.", icon=":material/casino:")
        return

    tiles = st.columns(3)
    tiles[0].metric("Generation latency", f"{last['elapsed_ns'] / 1e3:,.1f} us")
    tiles[1].metric(
        "Throughput (this call)",
        f"{last['size'] / (1 << 20) / (last['elapsed_ns'] / 1e9):,.1f} MiB/s",
    )
    tiles[2].metric(
        "Byte entropy (plug-in)",
        f"{last['entropy']:.4f} bits/byte",
        help="Shannon entropy of this sample's byte frequencies. Descriptive only.",
    )
    histogram = last["histogram"]
    fig = theme.figure("Byte value frequencies", "Byte value", "Count", height=300)
    fig.add_trace(
        go.Bar(
            x=np.arange(256),
            y=histogram,
            name="count",
            marker={"color": theme.CATEGORICAL[0]},
            hovertemplate="byte %{x}<br>count %{y}<extra></extra>",
        )
    )
    fig.add_hline(y=last["size"] / 256, line_color=theme.MUTED, line_width=1)
    fig.update_layout(bargap=0.05, showlegend=False)
    ui.chart(fig, key="rl_histogram")
    ui.table_view(pd.DataFrame({"byte": np.arange(256), "count": histogram}))

    data = last["bytes"]
    if data is None:
        st.info(
            "Output preview and download are withheld for this generator: its output is treated "
            "as secret. Only the statistics above are shown.",
            icon=":material/lock:",
        )
        return
    st.warning(
        "Demonstration bytes from a deterministic, labelled generator. Never use them as keys.",
        icon=":material/warning:",
    )
    st.code(data[:PREVIEW_BYTES].hex(" "), language=None)
    st.download_button(
        "Download demonstration data",
        data=data,
        file_name=f"entropylens-demo-{last['key']}.bin",
        mime="application/octet-stream",
        key="rl_download",
    )


def _attacks() -> None:
    st.markdown(
        "An LCG's next state is a fixed function of its current state, and its whole state fits "
        "in one number. Each attack below runs in milliseconds."
    )
    if st.button("Seed a victim LCG from the clock", key="rl_victim"):
        st.session_state["rl_victim_seed"] = EducationalLCG.seeded_from_clock().seed
    victim_seed = st.session_state.get("rl_victim_seed")
    if victim_seed is None:
        st.info("Seed a victim first. It uses the current Unix time in seconds, a common mistake.")
        return

    st.subheader("1. Predict from three outputs")
    secret = LCGParameters("victim", 0x2545F491, 0x6B8B4567, 1 << 32, 0, 32, "unknown to attacker")
    victim = EducationalLCG(victim_seed % (1 << 32), secret)
    observed = victim.outputs(3)
    start = time.perf_counter()
    candidates = recover_parameters(observed, 1 << 32)
    elapsed = time.perf_counter() - start
    st.markdown(
        f"Observed outputs: `{observed}`. Recovered (a, c): "
        f"`{[(hex(a), hex(c)) for a, c in candidates]}` in {elapsed * 1e3:.2f} ms."
    )
    if len(candidates) == 1:
        attacker = LCGParameters("recovered", *candidates[0], 1 << 32, 0, 32, "attack")
        predicted = predict_outputs(attacker, observed[-1], 5)
        actual = victim.outputs(5)
        st.dataframe(
            pd.DataFrame(
                {
                    "predicted": predicted,
                    "actual": actual,
                    "match": [p == a for p, a in zip(predicted, actual, strict=True)],
                }
            ),
            hide_index=True,
        )

    st.subheader("2. Recover the clock seed")
    clock_victim = EducationalLCG(victim_seed, NUMERICAL_RECIPES_32)
    clock_outputs = clock_victim.outputs(2)
    now = int(time.time())
    start = time.perf_counter()
    found = recover_seed(NUMERICAL_RECIPES_32, clock_outputs, range(now - 3600, now + 1))
    elapsed = time.perf_counter() - start
    st.markdown(
        f"Trying every second of the last hour (3,601 candidates) took {elapsed * 1e3:.1f} ms and "
        f"found seed(s) `{found}`; the victim used `{victim_seed}`."
    )

    st.subheader("3. Truncated output does not help")
    truncated = EducationalLCG((victim_seed * 65537) % (1 << 48), DRAND48_HIGH32)
    shown = truncated.outputs(2)
    start = time.perf_counter()
    states = recover_truncated_state(DRAND48_HIGH32, shown)
    elapsed = time.perf_counter() - start
    if len(states) == 1:
        predicted = predict_outputs(DRAND48_HIGH32, states[0], 3)
        st.markdown(
            f"Two 32-bit outputs `{shown}` hide 16 state bits; 65,536 guesses took "
            f"{elapsed * 1e3:.1f} ms. Next outputs predicted `{predicted}`, actual "
            f"`{truncated.outputs(3)}`."
        )
