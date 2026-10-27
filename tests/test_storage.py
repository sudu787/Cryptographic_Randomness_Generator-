"""Tests for result files: safe serialisation, confinement, no overwrite, validation."""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from entropylens.analysis.intervals import Interval
from entropylens.storage.repository import (
    StorageError,
    canonical_json,
    read_result,
    to_jsonable,
    write_result,
)
from entropylens.storage.schemas import ResultDocument, RunMetadata


def _document(name: str = "smoke") -> ResultDocument:
    return ResultDocument(
        metadata=RunMetadata(
            schema_version=1,
            kind="detection",
            name=name,
            started_utc="2026-10-27T09:15:00+00:00",
            finished_utc="2026-10-27T09:16:00+00:00",
            command="run smoke",
            machine={"python": "3.13", "logical_cpus": 8},
        ),
        config={"name": name},
        payload={"cells": [{"detected": 3}]},
    )


class _Colour(enum.Enum):
    RED = "red"


@dataclass(frozen=True)
class _Record:
    interval: Interval
    values: tuple[float, ...]


def test_to_jsonable_handles_result_structures() -> None:
    record = _Record(Interval(0.5, 0.2, 0.8, "wilson"), (1.0, float("nan"), float("inf")))
    converted = to_jsonable(
        {
            "record": record,
            "array": np.array([1, 2]),
            "scalar": np.float64(0.25),
            "enum": _Colour.RED,
            "set": frozenset({3}),
        }
    )
    assert converted == {
        "record": {
            "interval": {
                "estimate": 0.5,
                "low": 0.2,
                "high": 0.8,
                "method": "wilson",
                "confidence": 0.95,
            },
            "values": [1.0, None, None],
        },
        "array": [1, 2],
        "scalar": 0.25,
        "enum": "red",
        "set": [3],
    }


@pytest.mark.parametrize("raw", [b"\x00", bytearray(b"\x01"), memoryview(b"\x02")])
def test_to_jsonable_refuses_raw_bytes(raw: object) -> None:
    with pytest.raises(StorageError, match="raw bytes"):
        to_jsonable({"nested": [raw]})


def test_to_jsonable_refuses_unknown_types() -> None:
    with pytest.raises(StorageError, match="cannot serialise"):
        to_jsonable(object())


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    document = _document()
    path, digest = write_result(document, tmp_path)
    assert path.name == "smoke-20261027T091500Z.json"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_text(encoding="utf-8") == canonical_json(document)
    assert read_result(path, tmp_path) == document


def test_existing_files_are_never_overwritten(tmp_path: Path) -> None:
    write_result(_document(), tmp_path)
    with pytest.raises(StorageError, match="overwrite"):
        write_result(_document(), tmp_path)


def test_reading_is_confined_to_the_results_directory(tmp_path: Path) -> None:
    inside = tmp_path / "results"
    path, _ = write_result(_document(), inside)
    with pytest.raises(StorageError, match="escapes"):
        read_result(path, tmp_path / "elsewhere")
    with pytest.raises(StorageError, match="escapes"):
        read_result(inside / ".." / path.name, inside)


def test_invalid_result_files_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    data = json.loads(canonical_json(_document()))
    data["metadata"]["schema_version"] = 99
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(StorageError, match="invalid"):
        read_result(bad, tmp_path)
    text_file = tmp_path / "notes.txt"
    text_file.write_text("{}", encoding="utf-8")
    with pytest.raises(StorageError, match=r"\.json"):
        read_result(text_file, tmp_path)


def test_metadata_names_are_validated() -> None:
    with pytest.raises(ValueError, match="pattern"):
        _document(name="../escape")
