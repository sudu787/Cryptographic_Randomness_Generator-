"""End-to-end tests of the command-line interface with tiny configurations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropylens import cli
from entropylens.security import secure_source
from entropylens.storage.repository import read_result

TINY_DESIGN = {
    "warmup_samples": 4096,
    "onset_span": 0,
    "post_onset_samples": 8192,
    "buffer_samples": 4096,
}


def _write_config(directory: Path, name: str, **extra: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "name": name,
        "purpose": "development",
        "master_seed": 99,
        "cells": [
            {
                "name": "f1",
                "model": "S0",
                "fault": {"fault": "F1"},
                "trials": 2,
                "design": TINY_DESIGN,
            },
            {"name": "healthy", "model": "S0", "trials": 2, "design": TINY_DESIGN},
        ],
        **extra,
    }
    (directory / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")


def _main(tmp_path: Path, *args: str) -> int:
    return cli.main(
        ["--configs", str(tmp_path / "configs"), "--results", str(tmp_path / "out"), *args]
    )


def test_list_shows_valid_and_invalid_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(tmp_path / "configs", "tiny")
    (tmp_path / "configs" / "broken.json").write_text("{", encoding="utf-8")
    assert _main(tmp_path, "list") == 0
    output = capsys.readouterr().out
    assert "tiny" in output
    assert "2 cells" in output
    assert "broken" in output
    assert "INVALID" in output


def test_run_detection_writes_a_valid_result_file(tmp_path: Path) -> None:
    _write_config(tmp_path / "configs", "tiny")
    assert _main(tmp_path, "run", "tiny", "--quiet") == 0
    [path] = list((tmp_path / "out").glob("tiny-*.json"))
    document = read_result(path, tmp_path / "out")
    assert document.metadata.kind == "detection"
    assert document.config is not None
    assert document.config["master_seed"] == 99
    cells = document.payload["cells"]
    assert [cell["spec"]["name"] for cell in cells] == ["f1", "healthy"]
    assert cells[0]["summary"]["detected"] == 2
    assert cells[1]["summary"]["detected"] == 0
    assert cells[0]["trials"][0]["spawn_key"] == [0, 0]
    assert cells[1]["trials"][1]["spawn_key"] == [1, 1]
    assert len(cells[0]["trials"][0]["detectors"]["rct"]) == len(cells[0]["detector_fields"])


def test_running_one_cell_keeps_its_seed_position(tmp_path: Path) -> None:
    _write_config(tmp_path / "configs", "tiny")
    assert _main(tmp_path, "run", "tiny", "--quiet", "--cell", "healthy") == 0
    [path] = list((tmp_path / "out").glob("tiny-*.json"))
    cells = read_result(path, tmp_path / "out").payload["cells"]
    assert [cell["cell_index"] for cell in cells] == [1]


def test_run_placement(tmp_path: Path) -> None:
    design = {**TINY_DESIGN, "post_onset_samples": 1 << 15}
    configs = tmp_path / "configs"
    configs.mkdir()
    document = {
        "schema_version": 1,
        "name": "place",
        "purpose": "development",
        "master_seed": 5,
        "analysis": "placement",
        "output_block_bits": 4096,
        "cells": [
            {"name": "f1", "model": "S0", "fault": {"fault": "F1"}, "trials": 1, "design": design}
        ],
    }
    (configs / "place.json").write_text(json.dumps(document), encoding="utf-8")
    assert _main(tmp_path, "run", "place", "--quiet") == 0
    [path] = list((tmp_path / "out").glob("place-*.json"))
    cell = read_result(path, tmp_path / "out").payload["cells"][0]
    assert set(cell["summary"]["detected"]) == {"P1", "P2", "P3"}
    assert "p1" in cell["trials"][0]
    assert "p3" in cell["trials"][0]


def test_errors_return_exit_code_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_config(tmp_path / "configs", "tiny")
    assert _main(tmp_path, "run", "absent") == 2
    assert _main(tmp_path, "run", "../tiny") == 2
    assert _main(tmp_path, "run", "tiny", "--cell", "nope") == 2
    assert "error" in capsys.readouterr().err


def test_benchmark_result_contains_no_generator_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = bytes.fromhex("deadbeefcafef00d")
    monkeypatch.setattr(secure_source.secrets, "token_bytes", lambda n: (marker * (n // 8 + 1))[:n])
    assert _main(tmp_path, "benchmark", "--quick", "--name", "bench") == 0
    [path] = list((tmp_path / "out").glob("bench-*.json"))
    text = path.read_text(encoding="utf-8")
    assert "deadbeef" not in text.lower()
    document = read_result(path, tmp_path / "out")
    assert document.metadata.kind == "benchmark"
    assert {g["generator"] for g in document.payload["generators"]} >= {"os-csprng"}
    assert {p["variant"] for p in document.payload["pipeline"]} == {"end-to-end", "processing-only"}
    assert any("not a security metric" in note for note in document.payload["notes"])


def test_metadata_command_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["metadata"]) == 0
    assert "python" in json.loads(capsys.readouterr().out)
