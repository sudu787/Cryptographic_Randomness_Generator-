"""Dashboard smoke tests with Streamlit's AppTest: every page renders, and the main actions work.

Each page module is run on its own (``render()``), plus the full app with navigation. The results
directory points at the shared fixture's files, so history-backed pages render real data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = (
    "overview",
    "randomness_lab",
    "health_monitor",
    "fault_lab",
    "statistical_analysis",
    "benchmarking",
    "history",
    "report",
)


@pytest.fixture
def results(results_dir: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "results"
    shutil.copytree(results_dir, copy, ignore=shutil.ignore_patterns("*.sqlite3"))
    return copy


def _page(module: str, results: Path) -> AppTest:
    script = f"from entropylens.dashboard.pages import {module}\n{module}.render()\n"
    app = AppTest.from_string(script, default_timeout=120)
    app.session_state["results_dir"] = str(results)
    return app.run()


@pytest.mark.parametrize("module", PAGES)
def test_every_page_renders_without_exceptions(module: str, results: Path) -> None:
    app = _page(module, results)
    assert not app.exception
    assert app.title[0].value  # every page has a title
    assert any("does not show" in m.value for m in app.markdown) or module == "report"


def test_full_app_with_navigation_renders(results: Path) -> None:
    app = AppTest.from_file(
        str(Path(__file__).parent.parent / "entropylens" / "dashboard" / "app.py"),
        default_timeout=120,
    )
    app.session_state["results_dir"] = str(results)
    app.run()
    assert not app.exception
    assert app.title[0].value == "EntropyLens"


def test_fault_lab_single_trace_explains_the_alarm(results: Path) -> None:
    app = _page("fault_lab", results)
    app.selectbox(key="fl_scenario").set_value("F1").run()
    app.button(key="fl_run").click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Detection latency (samples)"] != "not detected"
    assert any("Why it fired" in m.value for m in app.markdown)
    assert any(m.label == "Faulty raw samples released" for m in app.metric)


def test_fault_lab_repeated_trials(results: Path) -> None:
    app = _page("fault_lab", results)
    app.selectbox(key="fl_scenario").set_value("F1").run()
    app.number_input(key="fl_post").set_value(8192).run()
    app.number_input(key="fl_trials").set_value(5).run()
    app.button(key="fl_repeat").click().run()
    assert not app.exception
    table = app.dataframe[-1].value
    assert "full monitor" in set(table["detectors"])
    assert table.loc[table["detectors"] == "full monitor", "detected"].item() == "5/5"


def test_health_monitor_streams_and_records_status(results: Path) -> None:
    app = _page("health_monitor", results)
    app.number_input(key="hm_buffers").set_value(4).run()
    app.button(key="hm_run").click().run()
    assert not app.exception
    assert app.session_state["monitor_status"] is not None


def test_randomness_lab_never_reveals_os_csprng_output(results: Path) -> None:
    app = _page("randomness_lab", results)
    app.selectbox(key="rl_generator").set_value("os-csprng").run()
    app.button(key="rl_generate").click().run()
    assert not app.exception
    assert app.session_state["rl_last"]["bytes"] is None
    assert len(app.code) == 0
    assert len(app.get("download_button")) == 0


def test_randomness_lab_previews_deterministic_output_with_a_warning(results: Path) -> None:
    app = _page("randomness_lab", results)
    app.selectbox(key="rl_generator").set_value("lcg-nr32").run()
    app.button(key="rl_generate").click().run()
    assert not app.exception
    assert len(app.code) == 1
    assert any("Never use them as keys" in w.value for w in app.warning)


def test_randomness_lab_attacks_run(results: Path) -> None:
    app = _page("randomness_lab", results)
    app.button(key="rl_victim").click().run()
    assert not app.exception
    assert any("Recovered (a, c)" in m.value for m in app.markdown)
    assert any("found seed(s)" in m.value for m in app.markdown)


def test_statistical_analysis_runs_the_battery(results: Path) -> None:
    app = _page("statistical_analysis", results)
    app.selectbox(key="sa_source").set_value("chacha20-published").run()
    app.button(key="sa_run").click().run()
    assert not app.exception
    rows = app.dataframe[0].value
    assert len(rows) == 6
    assert app.success or app.error  # a plain-language verdict is always shown


def test_benchmarking_page_runs_a_small_benchmark(results: Path) -> None:
    app = _page("benchmarking", results)
    app.multiselect(key="bm_generators").set_value(["os-csprng"]).run()
    app.multiselect(key="bm_sizes").set_value([1024]).run()
    app.slider(key="bm_reps").set_value(3).run()
    app.button(key="bm_run").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 2  # generator table and overhead table


def test_history_page_filters_and_exports(results: Path) -> None:
    app = _page("history", results)
    assert not app.exception
    assert len(app.get("download_button")) >= 6  # CSV and JSON for runs, cells, benchmarks
    app.multiselect(key="hi_kinds").set_value(["benchmark"]).run()
    assert not app.exception
    assert app.subheader[1].value == "Runs (1)"
    app.button(key="hi_verify").click().run()
    assert not app.exception
