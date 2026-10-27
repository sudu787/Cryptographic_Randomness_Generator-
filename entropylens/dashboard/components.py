"""Reusable dashboard pieces: headers, limitation boxes, alert cards, badges, charts, table views.

Everything renders through Streamlit's own widgets and Markdown. No raw HTML is rendered, so no
input can inject markup.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entropylens.dashboard.theme import OUTCOME_BADGES, STATUS_BADGES
from entropylens.monitoring.alerts import Alert, SourceStatus
from entropylens.security.base import GeneratorInfo

SECURITY_BADGES = {
    "production": ("green", ":material/verified_user:", "Production: OS CSPRNG"),
    "educational-insecure": ("red", ":material/dangerous:", "INSECURE: educational only"),
    "reference-deterministic": ("violet", ":material/key:", "Deterministic reference"),
}


def page_header(title: str, purpose: str) -> None:
    """Page title with a one-line statement of what the page is for."""
    st.title(title)
    st.caption(purpose)


def limitations(*points: str) -> None:
    """A clearly marked box stating what the page's results do **not** show."""
    with st.container(border=True):
        st.markdown("**:material/report: What this does not show**")
        for point in points:
            st.markdown(f"- {point}")


def status_badge(status: SourceStatus | None) -> None:
    """Source status as a coloured badge with an icon and a label."""
    if status is None:
        st.badge("No monitor run in this session", icon=":material/remove:", color="gray")
        return
    colour, icon, label = STATUS_BADGES[status]
    st.badge(label, icon=icon, color=colour)  # type: ignore[arg-type]


def security_badge(info: GeneratorInfo) -> None:
    """Security status of a generator, impossible to miss."""
    colour, icon, label = SECURITY_BADGES[info.security_status]
    st.badge(label, icon=icon, color=colour)  # type: ignore[arg-type]


def alert_card(alert: Alert, *, compact: bool = False) -> None:
    """Explain one alert: what fired, the evidence, and what it cannot establish."""
    colour, icon, label = OUTCOME_BADGES[alert.outcome]
    with st.container(border=True):
        st.badge(f"{label}: {alert.detector}", icon=icon, color=colour)  # type: ignore[arg-type]
        where = f" at sample {alert.sample_index:,}" if alert.sample_index >= 0 else ""
        st.markdown(f"**Why it fired{where}.** {alert.message}")
        if not compact and alert.evidence:
            st.dataframe(
                pd.DataFrame(
                    {
                        "evidence": list(alert.evidence),
                        "value": [str(v) for v in alert.evidence.values()],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        if alert.limitation:
            st.caption(f":material/report: Cannot establish: {alert.limitation}")


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the dashboard's own template (not Streamlit's)."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displaylogo": False})


def table_view(rows: Sequence[dict[str, Any]] | pd.DataFrame, label: str = "Table view") -> None:
    """Every chart has a table twin, so no value depends on colour or hover alone."""
    with st.expander(label, icon=":material/table:"):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        st.dataframe(frame, hide_index=True, width="stretch")


def interval_text(estimate: float, low: float, high: float, digits: int = 3) -> str:
    """``0.950 [0.912, 0.975]``."""
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
