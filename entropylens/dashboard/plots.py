"""Chart builders for the monitoring pages.

Conventions: one y-axis per chart; detectors keep fixed colours (``theme.DETECTOR_COLOURS``); the
fault region is shaded with the reserved *critical* status colour because it marks a state, and is
labelled in the legend; reference lines are solid and muted.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from entropylens.dashboard import theme
from entropylens.dashboard.simulation import MonitorRun, windowed_fraction, windowed_longest_run
from entropylens.monitoring.alerts import SourceStatus

STATUS_LEVEL = {
    SourceStatus.UNKNOWN: 0,
    SourceStatus.HEALTHY: 1,
    SourceStatus.DEGRADED: 2,
    SourceStatus.FAILED: 3,
}
STATUS_COLOUR = {
    SourceStatus.UNKNOWN: theme.MUTED,
    SourceStatus.HEALTHY: theme.STATUS_COLOURS["good"],
    SourceStatus.DEGRADED: theme.STATUS_COLOURS["warning"],
    SourceStatus.FAILED: theme.STATUS_COLOURS["critical"],
}


def fault_segments(faulty: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges where the fault label is set."""
    if faulty.size == 0 or not faulty.any():
        return []
    padded = np.concatenate(([False], faulty, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def window_for(length: int, points: int = 800) -> int:
    """A window length giving roughly ``points`` windows, rounded to a power of two (>= 64)."""
    return int(max(64, 1 << max(6, int(np.log2(max(1, length // points))))))


def _has_fault(run: MonitorRun) -> bool:
    return bool(run.trace.faulty.any()) or run.trace.outage is not None


def _shade_faults(fig: go.Figure, run: MonitorRun) -> None:
    segments = fault_segments(run.trace.faulty)
    for start, end in segments[:200]:  # bursty faults: cap the number of shapes drawn
        fig.add_vrect(
            x0=start, x1=end, fillcolor=theme.STATUS_COLOURS["critical"], opacity=0.14, line_width=0
        )
    if segments:  # legend entry for the shading
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="fault active (ground truth)",
                marker={
                    "symbol": "square",
                    "size": 12,
                    "color": theme.STATUS_COLOURS["critical"],
                    "opacity": 0.4,
                },
            )
        )
    if not _has_fault(run):
        return
    fig.add_vline(x=run.trace.onset, line_color=theme.MUTED, line_width=1)
    fig.add_annotation(
        x=run.trace.onset,
        y=1,
        yref="paper",
        text="onset",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"color": theme.MUTED, "size": 11},
    )


def signal_figure(run: MonitorRun, binary: bool, nominal_p_one: float) -> go.Figure:
    """Windowed fraction of ones with the fault region and each detector's first alarm."""
    samples = run.trace.samples
    window = window_for(samples.size)
    ends, fraction = windowed_fraction(samples, binary, window)
    fig = theme.figure(
        f"Fraction of ones per {window}-sample window", "Sample index", "Fraction of ones"
    )
    _shade_faults(fig, run)
    fig.add_hline(y=nominal_p_one, line_color=theme.BASELINE, line_width=1)
    fig.add_trace(
        go.Scatter(
            x=ends,
            y=fraction,
            mode="lines",
            name="observed",
            line={"color": theme.INK_SECONDARY},
            hovertemplate="sample %{x:,}<br>fraction %{y:.3f}<extra></extra>",
        )
    )
    for name, outcome in run.record.detectors.items():
        index = outcome.first_alarm_after
        if index is None:
            continue
        position = min(int(np.searchsorted(ends, index)), max(0, fraction.size - 1))
        y = float(fraction[position]) if fraction.size else nominal_p_one
        fig.add_trace(
            go.Scatter(
                x=[index],
                y=[y],
                mode="markers",
                name=f"first alarm: {name}",
                marker={
                    "size": 11,
                    "color": theme.DETECTOR_COLOURS.get(name, theme.INK),
                    "line": {"width": 2, "color": theme.SURFACE},
                },
                hovertemplate=f"{name}<br>sample %{{x:,}}<extra></extra>",
            )
        )
    return fig


def repetition_figure(run: MonitorRun, cutoff: int | None) -> go.Figure:
    """Longest run of identical samples per window, against the RCT cutoff."""
    samples = run.trace.samples
    window = window_for(samples.size)
    ends, longest = windowed_longest_run(samples, window)
    fig = theme.figure(
        f"Longest run of identical samples per {window}-sample window",
        "Sample index",
        "Run length (samples)",
        height=380,
    )
    _shade_faults(fig, run)
    fig.add_trace(
        go.Scatter(
            x=ends,
            y=longest,
            mode="lines",
            name="longest run",
            line={"color": theme.CATEGORICAL[0]},
            hovertemplate="sample %{x:,}<br>longest run %{y}<extra></extra>",
        )
    )
    if cutoff is not None:
        fig.add_hline(y=cutoff, line_color=theme.STATUS_COLOURS["serious"], line_width=1)
        fig.add_annotation(
            x=1,
            xref="paper",
            y=cutoff,
            text=f"RCT cutoff C = {cutoff}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": theme.INK_SECONDARY, "size": 11},
        )
    return fig


def status_timeline(run: MonitorRun) -> go.Figure:
    """Monitor status after each buffer: labelled, colour-coded markers on one categorical axis."""
    fig = theme.figure("Source status after each buffer", "Sample index", "", height=320)
    for status in STATUS_LEVEL:
        points = [end for end, s in zip(run.buffer_ends, run.statuses, strict=True) if s is status]
        if points:
            fig.add_trace(
                go.Scatter(
                    x=points,
                    y=[status.value] * len(points),
                    mode="markers",
                    name=status.value,
                    marker={"size": 10, "color": STATUS_COLOUR[status], "symbol": "square"},
                    hovertemplate="after sample %{x:,}: " + status.value + "<extra></extra>",
                )
            )
    fig.update_yaxes(
        categoryorder="array", categoryarray=[s.value for s in STATUS_LEVEL], type="category"
    )
    if _has_fault(run):
        fig.add_vline(x=run.trace.onset, line_color=theme.MUTED, line_width=1)
    return fig
