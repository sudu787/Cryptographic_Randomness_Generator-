"""Versioned schema of result files written by the CLI (and, from Phase 5, indexed by history).

A result file is one JSON document: run metadata, the configuration exactly as validated, and a
payload whose shape depends on the run kind. Result files hold measurements and metadata only:
never generator output, keys, seeds of production sources, or raw sample data
(``docs/threat_model.md`` threat T4).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

RESULT_SCHEMA_VERSION = 1
RunKind = Literal["detection", "placement", "benchmark"]
MetadataValue = str | int | float | bool | None


class RunMetadata(BaseModel):
    """Who, what and when of one run. Timestamps are ISO 8601 in UTC."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int
    kind: RunKind
    name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
    started_utc: str
    finished_utc: str
    command: Annotated[str, StringConstraints(max_length=1000)]
    machine: dict[str, MetadataValue]

    @field_validator("schema_version")
    @classmethod
    def _supported(cls, value: int) -> int:
        if value != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema_version {value}")
        return value


class ResultDocument(BaseModel):
    """A complete result file."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metadata: RunMetadata
    config: dict[str, Any] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
