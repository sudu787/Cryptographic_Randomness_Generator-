"""Result-file storage: safe conversion to JSON, confined paths, and integrity digests.

Phase 5 adds the SQLite experiment-history index on top of these files.

Controls (``docs/threat_model.md``):

* T4: :func:`to_jsonable` refuses ``bytes``, so raw output can never be written by accident.
* T10: file names come from validated run names and timestamps, and every path is resolved and
  confined to the results directory. Existing files are never overwritten.
* T12: the SHA-256 of each file's canonical JSON is returned for the history index. It detects
  accidental modification only; it is not a signature.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from entropylens.storage.schemas import ResultDocument

MAX_RESULT_BYTES = 256 * 1024 * 1024


class StorageError(ValueError):
    """A result file could not be written or read safely."""


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, Pydantic models, enums, mappings, sequences and NumPy scalars to JSON.

    Non-finite floats become ``None``.

    Raises:
        StorageError: For ``bytes`` (raw output must never reach a result file) and unknown types.
    """
    if isinstance(value, bytes | bytearray | memoryview):
        raise StorageError("refusing to serialise raw bytes into a result file")
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    raise StorageError(f"cannot serialise {type(value).__name__}")


def canonical_json(document: ResultDocument) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, no NaN."""
    return json.dumps(
        document.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _confined(path: Path, directory: Path) -> Path:
    base = directory.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise StorageError(f"path escapes the results directory {base}")
    return resolved


def write_result(document: ResultDocument, results_dir: Path) -> tuple[Path, str]:
    """Write ``document`` as ``<name>-<timestamp>.json`` in ``results_dir``.

    Returns:
        The file path and the SHA-256 hex digest of its contents.

    Raises:
        StorageError: If the file exists already or the path would leave ``results_dir``.
    """
    try:
        started = datetime.fromisoformat(document.metadata.started_utc)
    except ValueError:
        raise StorageError("started_utc is not an ISO 8601 timestamp") from None
    stamp = started.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = _confined(results_dir / f"{document.metadata.name}-{stamp}.json", results_dir)
    text = canonical_json(document)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        raise StorageError(f"refusing to overwrite {path.name}") from None
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_result(path: Path, results_dir: Path) -> ResultDocument:
    """Read and validate a result file located inside ``results_dir``."""
    resolved = _confined(path, results_dir)
    if resolved.suffix != ".json":
        raise StorageError("result files must have a .json suffix")
    if resolved.stat().st_size > MAX_RESULT_BYTES:
        raise StorageError("result file is too large")
    try:
        return ResultDocument.model_validate_json(resolved.read_bytes())
    except ValidationError as exc:
        raise StorageError(f"invalid result file {resolved.name}: {exc}") from exc
