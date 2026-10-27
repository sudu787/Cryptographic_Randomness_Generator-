"""Visual language of the dashboard: palette, chart template and status mapping.

Colours are the reference data-viz palette in its dark-mode steps, used unchanged. Its adjacent-pair
colour-vision checks are documented as passing for this order on the dark surface ``#1a1a19``. The
validator script was not re-run here (no Node.js on this machine), so nothing was re-stepped.

Rules applied throughout: categorical colours follow the *entity* (each detector and each
generator has a fixed slot), never its rank; status colours are reserved for health states and
always appear with an icon and a label; one y-axis per chart; thin marks and hairline grids.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from entropylens.monitoring.alerts import Outcome, SourceStatus

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#f0efec"
INK_SECONDARY = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CATEGORICAL = (
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
)

STATUS_COLOURS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Fixed colour per detector, so a detector keeps its colour whichever subset is shown.
DETECTOR_COLOURS = {
    "rct": CATEGORICAL[0],
    "apt": CATEGORICAL[1],
    "repeated-block": CATEGORICAL[2],
    "frequency": CATEGORICAL[3],
    "runs": CATEGORICAL[4],
    "lag-agreement": CATEGORICAL[5],
    "availability": CATEGORICAL[6],
    "autocorrelation": CATEGORICAL[7],
    "byte-chi-square": CATEGORICAL[7],  # never shown together with autocorrelation
}

#: Fixed colour per generator identifier.
GENERATOR_COLOURS = {
    "os-csprng": CATEGORICAL[0],
    "lcg-numerical-recipes-32": CATEGORICAL[1],
    "lcg-numerical-recipes-32-numpy": CATEGORICAL[2],
    "chacha20-random-key": CATEGORICAL[3],
    "lcg-drand48-high32": CATEGORICAL[4],
    "chacha20-seeded": CATEGORICAL[5],
    "chacha20-negative-control": CATEGORICAL[6],
}

PIPELINE_COLOURS = {"P1": CATEGORICAL[1], "P2": CATEGORICAL[0], "P3": CATEGORICAL[6]}

#: (Streamlit badge colour, Material icon, label) per source status: never colour alone.
STATUS_BADGES = {
    SourceStatus.HEALTHY: ("green", ":material/check_circle:", "Healthy"),
    SourceStatus.DEGRADED: ("orange", ":material/warning:", "Degraded (warnings only)"),
    SourceStatus.FAILED: ("red", ":material/error:", "Failed (output withheld)"),
    SourceStatus.UNKNOWN: ("gray", ":material/help:", "Unknown (output withheld)"),
}

OUTCOME_BADGES = {
    Outcome.CRITICAL: ("red", ":material/error:", "Critical"),
    Outcome.WARNING: ("orange", ":material/warning:", "Warning (exploratory)"),
    Outcome.ERROR: ("gray", ":material/bug_report:", "Detector error"),
    Outcome.PASS: ("green", ":material/check:", "Pass"),
    Outcome.INSUFFICIENT_DATA: ("gray", ":material/hourglass_empty:", "Insufficient data"),
}

TEMPLATE_NAME = "entropylens"


def register_template() -> None:
    """Register the dark chart template (idempotent)."""
    if TEMPLATE_NAME in pio.templates:
        return
    axis = {
        "gridcolor": GRID,
        "gridwidth": 1,
        "linecolor": BASELINE,
        "showline": True,
        "zerolinecolor": BASELINE,
        "tickcolor": BASELINE,
        "tickfont": {"color": MUTED},
        "title": {"font": {"color": INK_SECONDARY}},
    }
    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font={"family": FONT, "color": INK_SECONDARY, "size": 13},
            title={"font": {"color": INK, "size": 15}, "x": 0, "xanchor": "left"},
            colorway=list(CATEGORICAL),
            xaxis=axis,
            yaxis=axis,
            # Legend below the plot, so it never collides with the title on narrow screens.
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.3,
                "x": 0,
                "font": {"color": INK_SECONDARY},
            },
            hoverlabel={"bgcolor": "#262624", "bordercolor": BASELINE, "font": {"color": INK}},
            hovermode="closest",
            margin={"l": 64, "r": 24, "t": 56, "b": 64},
            barcornerradius=4,
            bargap=0.2,
        )
    )
    template.data.scatter = [go.Scatter(line={"width": 2}, marker={"size": 8})]
    pio.templates[TEMPLATE_NAME] = template


def figure(title: str, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    """An empty figure with the dashboard template, axis titles and a fixed height.

    The height includes the legend band below the x-axis, so nothing is clipped.
    """
    register_template()
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
    )
    return fig
