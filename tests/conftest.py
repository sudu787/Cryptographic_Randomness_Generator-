"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropylens import cli

TINY_DESIGN = {
    "warmup_samples": 4096,
    "onset_span": 0,
    "post_onset_samples": 8192,
    "buffer_samples": 4096,
}


def _config(name: str, analysis: str, cells: list[dict[str, object]], **extra: object) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "name": name,
            "purpose": "development",
            "master_seed": 99,
            "analysis": analysis,
            "cells": cells,
            **extra,
        }
    )


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A results directory holding one detection, one placement and one benchmark result file,
    all produced by the real CLI with tiny settings."""
    root = tmp_path_factory.mktemp("history")
    configs, results = root / "configs", root / "results"
    configs.mkdir()
    (configs / "tiny.json").write_text(
        _config(
            "tiny",
            "detection",
            [
                {
                    "name": "f1",
                    "model": "S0",
                    "fault": {"fault": "F1"},
                    "trials": 2,
                    "design": TINY_DESIGN,
                },
                {"name": "healthy", "model": "S0", "trials": 2, "design": TINY_DESIGN},
            ],
        ),
        encoding="utf-8",
    )
    placement_design = {**TINY_DESIGN, "post_onset_samples": 1 << 15}
    (configs / "place.json").write_text(
        _config(
            "place",
            "placement",
            [
                {
                    "name": "f1",
                    "model": "S0",
                    "fault": {"fault": "F1"},
                    "trials": 1,
                    "design": placement_design,
                }
            ],
            output_block_bits=4096,
        ),
        encoding="utf-8",
    )
    base = ["--configs", str(configs), "--results", str(results)]
    assert cli.main([*base, "run", "tiny", "--quiet"]) == 0
    assert cli.main([*base, "run", "place", "--quiet"]) == 0
    assert cli.main([*base, "benchmark", "--quick", "--name", "bench"]) == 0
    return results
