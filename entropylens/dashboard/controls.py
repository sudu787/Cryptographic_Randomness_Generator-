"""Shared input controls: source model, fault scenario and monitor settings.

All inputs are bounded widgets; nothing typed by the user is executed or rendered as markup.
"""

from __future__ import annotations

from typing import Literal, cast

import streamlit as st

from entropylens.experiments import fault_injection as fi
from entropylens.experiments.synthetic_sources import MODELS, ModelName
from entropylens.monitoring.source_health import MonitorSettings

SCENARIOS: dict[str, str] = {
    "none": "Healthy (no fault)",
    "F1": "F1: constant output (stuck)",
    "F2": "F2: stuck bit (byte sources)",
    "F3": "F3: bias",
    "F4": "F4: short repeating pattern",
    "F5": "F5: repeated block (buffer replay)",
    "F6": "F6: lag correlation (sticky samples)",
    "F7": "F7: abrupt change of distribution",
    "F8a": "F8a: replay from an earlier point (snapshot restore)",
    "F8b": "F8b: low-entropy deterministic substitution",
    "F9": "F9: source outage",
    "F10": "F10: intermittent bursts",
    "N1": "N1: negative control (ChaCha20, published key)",
}
BINARY_ONLY = {"F3", "F6", "F7"}


def model_control(key: str) -> ModelName:
    """Select a nominal source model."""
    labels = {name: f"{name}: {model.description}" for name, model in MODELS.items()}
    choice = st.selectbox(
        "Source model", list(labels), format_func=labels.__getitem__, key=f"{key}_model"
    )
    return cast(ModelName, choice)


def fault_control(key: str, model: ModelName, *, allow_none: bool = True) -> fi.FaultBase | None:
    """Select a fault scenario and its intensity; returns ``None`` for a healthy source."""
    kind = MODELS[model].kind
    options = [
        s
        for s in SCENARIOS
        if (allow_none or s != "none")
        and not (kind == "byte" and s in BINARY_ONLY)
        and not (kind == "binary" and s == "F2")
    ]
    scenario = st.selectbox(
        "Fault scenario", options, format_func=SCENARIOS.__getitem__, key=f"{key}_scenario"
    )
    return _parameters(scenario, kind, key)


def _parameters(scenario: str, kind: str, key: str) -> fi.FaultBase | None:
    if scenario == "none":
        return None
    if scenario == "F1":
        top = 1 if kind == "binary" else 255
        return fi.ConstantFault(
            value=int(st.number_input("Stuck value", 0, top, 0, key=f"{key}_value"))
        )
    if scenario == "F2":
        return fi.StuckBitFault(
            bit=int(st.number_input("Bit position", 0, 7, 0, key=f"{key}_bit")),
            value=int(st.number_input("Stuck at", 0, 1, 0, key=f"{key}_bitvalue")),
        )
    if scenario == "F3":
        return fi.BiasFault(
            p_one=st.slider("P(1) after the fault", 0.40, 0.90, 0.65, 0.01, key=f"{key}_p")
        )
    if scenario == "F4":
        return fi.PatternFault(
            period=int(
                st.number_input("Pattern period (samples)", 2, 4096, 16, key=f"{key}_period")
            )
        )
    if scenario == "F5":
        return fi.BlockReplayFault(
            block_samples=int(
                st.number_input("Replayed block (samples)", 1, 65536, 512, key=f"{key}_block")
            ),
            distance=int(
                st.number_input("Replay distance (samples)", 1, 4096, 2048, key=f"{key}_dist")
            ),
        )
    if scenario == "F6":
        return fi.CorrelationFault(
            stay_probability=st.slider(
                "P(next = current)", 0.51, 0.95, 0.70, 0.01, key=f"{key}_stay"
            )
        )
    if scenario == "F7":
        return fi.ChangePointFault(
            p01=st.slider("P(1 | previous 0)", 0.05, 0.95, 0.60, 0.01, key=f"{key}_p01"),
            p11=st.slider("P(1 | previous 1)", 0.05, 0.95, 0.60, 0.01, key=f"{key}_p11"),
        )
    if scenario == "F8a":
        return fi.ReplayFromStartFault(
            replay_from=int(st.number_input("Replay from sample", 0, 4000, 0, key=f"{key}_from"))
        )
    if scenario == "F8b":
        return fi.LowEntropySubstitutionFault(
            seed_bits=int(st.number_input("Seed bits", 1, 32, 16, key=f"{key}_seedbits"))
        )
    if scenario == "F9":
        mode = st.selectbox("Failure mode", ["exception", "short-read"], key=f"{key}_mode")
        return fi.OutageFault(mode=cast(Literal["exception", "short-read"], mode))
    if scenario == "F10":
        return fi.IntermittentFault(
            base=fi.ConstantFault(),
            mean_on=float(st.number_input("Mean burst length", 1, 4096, 64, key=f"{key}_on")),
            mean_off=float(st.number_input("Mean gap length", 1, 65536, 4096, key=f"{key}_off")),
        )
    return fi.NegativeControlFault()


def monitor_control(key: str) -> MonitorSettings:
    """False-alarm budget and history length; thresholds follow from these and the model only."""
    alpha_log2 = st.slider(
        "False-alarm budget per evaluation, log2(alpha)",
        -40,
        -10,
        -30,
        help="SP 800-90B recommends 2^-20 to 2^-40. Thresholds are derived from this value and "
        "the model's claimed min-entropy; they are never tuned on fault data.",
        key=f"{key}_alpha",
    )
    history = st.select_slider(
        "Repeated-block history (samples)",
        options=[1 << 12, 1 << 14, 1 << 16, 1 << 18],
        value=1 << 16,
        key=f"{key}_history",
    )
    return MonitorSettings(alpha_log2=int(alpha_log2), history_samples=int(history))
