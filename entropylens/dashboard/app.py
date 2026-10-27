"""EntropyLens dashboard entry point.

Run from the repository root (the server binds to localhost, see ``.streamlit/config.toml``)::

    streamlit run entropylens/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from entropylens.dashboard import theme
from entropylens.dashboard.pages import (
    benchmarking,
    fault_lab,
    health_monitor,
    history,
    overview,
    randomness_lab,
    report,
    statistical_analysis,
)


def main() -> None:
    """Configure the page and route to the eight dashboard pages."""
    st.set_page_config(page_title="EntropyLens", page_icon=":material/insights:", layout="wide")
    theme.register_template()
    pages = {
        "Research": [
            st.Page(
                overview.render,
                title="Overview",
                icon=":material/dashboard:",
                url_path="overview",
                default=True,
            ),
            st.Page(
                report.render, title="Research Report", icon=":material/article:", url_path="report"
            ),
        ],
        "Laboratory": [
            st.Page(
                fault_lab.render,
                title="Fault Injection Lab",
                icon=":material/science:",
                url_path="fault-lab",
            ),
            st.Page(
                health_monitor.render,
                title="Entropy & Health Monitor",
                icon=":material/monitor_heart:",
                url_path="health-monitor",
            ),
            st.Page(
                randomness_lab.render,
                title="Randomness Lab",
                icon=":material/casino:",
                url_path="randomness-lab",
            ),
            st.Page(
                statistical_analysis.render,
                title="Statistical Analysis",
                icon=":material/query_stats:",
                url_path="statistics",
            ),
        ],
        "Measurements": [
            st.Page(
                benchmarking.render,
                title="Benchmarking",
                icon=":material/speed:",
                url_path="benchmarking",
            ),
            st.Page(
                history.render,
                title="Experiment History",
                icon=":material/history:",
                url_path="history",
            ),
        ],
    }
    navigation = st.navigation(pages)
    with st.sidebar:
        st.caption(
            "Research prototype. Not a certified or NIST-compliant security product. "
            "Faults are injected only into simulated sources."
        )
    navigation.run()


main()
