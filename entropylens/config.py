"""Validated, JSON-only experiment configuration.

Configuration files are data, never code (``docs/threat_model.md``, threats T9 and T10):

* They are parsed with Pydantic's JSON parser. No ``eval``, ``pickle`` or YAML is involved.
* Files are size-limited and are looked up by a name matching a strict pattern, resolved inside a
  given configs directory.
* Validation is strict. Unknown fields, wrong types (``"5"`` or ``true`` where an integer is
  expected) and out-of-range values are errors; nothing is silently coerced.

Generator configurations and the header are shared by every experiment. Fault-detection experiments
add cells whose source-model, fault, trial-design and monitor sections are the strict models defined
next to the code that uses them (``experiments.evaluation``, ``experiments.fault_injection``,
``monitoring.source_health``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from entropylens.experiments.evaluation import CellSpec
from entropylens.security.crypto_reference import SEED_LIMIT, ChaCha20Reference
from entropylens.security.educational_lcg import PRESETS, EducationalLCG
from entropylens.security.secure_source import SecureSource

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 1024 * 1024
NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_NAME_RE = re.compile(NAME_PATTERN)

ExperimentName = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
Seed128 = Annotated[int, Field(ge=0, lt=SEED_LIMIT)]


class ConfigError(ValueError):
    """A configuration file, name or value is invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SecureSourceConfig(_StrictModel):
    """The production OS CSPRNG. It deliberately accepts no parameters, in particular no seed."""

    kind: Literal["os-csprng"] = "os-csprng"

    def build(self) -> SecureSource:
        """Create the generator."""
        return SecureSource()


class LCGConfig(_StrictModel):
    """An educational LCG preset with an explicit seed. Insecure by design."""

    kind: Literal["lcg"] = "lcg"
    preset: Literal["numerical-recipes-32", "drand48-high32"] = "numerical-recipes-32"
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _seed_below_modulus(self) -> LCGConfig:
        modulus = PRESETS[self.preset].modulus
        if self.seed >= modulus:
            raise ValueError(f"seed must be < {modulus} for preset {self.preset!r}")
        return self

    def build(self) -> EducationalLCG:
        """Create the generator."""
        return EducationalLCG(self.seed, PRESETS[self.preset])


class ChaCha20Config(_StrictModel):
    """The ChaCha20 reference generator.

    Raw keys are never written into configuration files. ``"seeded"`` derives the key from ``seed``
    (reproducible, zero secrecy against anyone who has the file). ``"random"`` draws a fresh key
    from the OS CSPRNG. ``"negative-control"`` uses the published key.
    """

    kind: Literal["chacha20"] = "chacha20"
    key_mode: Literal["random", "seeded", "negative-control"] = "random"
    seed: Seed128 | None = None

    @model_validator(mode="after")
    def _seed_iff_seeded(self) -> ChaCha20Config:
        if (self.key_mode == "seeded") != (self.seed is not None):
            raise ValueError("seed is required for key_mode 'seeded' and forbidden otherwise")
        return self

    def build(self) -> ChaCha20Reference:
        """Create the generator."""
        if self.key_mode == "seeded":
            if self.seed is None:  # unreachable after validation; narrows the type
                raise ConfigError("key_mode 'seeded' requires a seed")
            return ChaCha20Reference.from_seed(self.seed)
        if self.key_mode == "negative-control":
            return ChaCha20Reference.negative_control()
        return ChaCha20Reference()


GeneratorConfig = Annotated[
    SecureSourceConfig | LCGConfig | ChaCha20Config, Field(discriminator="kind")
]


class ExperimentConfig(_StrictModel):
    """Header shared by every experiment configuration.

    Attributes:
        schema_version: Must equal :data:`SCHEMA_VERSION`; bumped on incompatible changes.
        name: Lower-case identifier, also the configuration's file stem.
        purpose: ``"evaluation"`` runs produce reported results. Their master seed must be fixed
            before the run, and must differ from the seeds used during development.
        master_seed: Root of every per-cell and per-trial seed (via NumPy ``SeedSequence``).
        description: Free text for humans.
        generators: Generators the experiment compares (benchmarks, output statistics).
    """

    schema_version: int
    name: ExperimentName
    purpose: Literal["development", "evaluation", "demonstration"]
    master_seed: Seed128
    description: Annotated[str, StringConstraints(max_length=2000)] = ""
    generators: Annotated[tuple[GeneratorConfig, ...], Field(max_length=16)] = ()

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        # A plain int field in strict mode rejects JSON `true`, which Literal[1] would accept.
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value}; expected {SCHEMA_VERSION}")
        return value


class DetectionExperimentConfig(ExperimentConfig):
    """A fault-detection experiment: the shared header plus the cells to run (methodology E1-E7).

    ``analysis`` selects raw monitoring only (``"detection"``) or the E3 comparison of P1, P2 and
    P3 on identical traces (``"placement"``), whose output tests use ``output_block_bits`` and
    ``output_family_alpha``.

    Cell indices (their position in ``cells``) are part of every trial's seed, so reordering cells
    changes which random streams they receive. Append new cells instead of reordering.
    """

    cells: Annotated[tuple[CellSpec, ...], Field(min_length=1, max_length=256)]
    analysis: Literal["detection", "placement"] = "detection"
    output_block_bits: Annotated[int, Field(ge=1024, le=1 << 22, multiple_of=8)] = 1 << 16
    output_family_alpha: Annotated[float, Field(gt=0, lt=1)] = 0.01

    @model_validator(mode="after")
    def _unique_cell_names(self) -> DetectionExperimentConfig:
        names = [cell.name for cell in self.cells]
        if len(set(names)) != len(names):
            raise ValueError("cell names must be unique")
        return self


def validate_name(name: object) -> str:
    """Return ``name`` if it is a valid experiment or file name, else raise :class:`ConfigError`.

    Valid names are 1 to 64 characters of ``a-z``, ``0-9``, ``-`` and ``_``, starting with a letter
    or digit. Path separators, dots and drive letters can therefore never appear.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ConfigError(f"invalid name {name!r}: expected pattern {NAME_PATTERN}")
    return name


def resolve_config_path(name: str, configs_dir: Path) -> Path:
    """Resolve ``<configs_dir>/<name>.json`` and confirm it stays inside ``configs_dir``."""
    base = configs_dir.resolve()
    path = (base / f"{validate_name(name)}.json").resolve()
    if not path.is_relative_to(base):  # defence in depth, e.g. against symbolic links
        raise ConfigError(f"configuration path escapes {base}")
    return path


def parse_config[ModelT: BaseModel](data: str | bytes, model: type[ModelT]) -> ModelT:
    """Parse and strictly validate JSON text as ``model``.

    Raises:
        ConfigError: If the text is too large, not valid JSON, or fails validation.
    """
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        return model.model_validate_json(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid {model.__name__}: {exc}") from exc


def load_config[ModelT: BaseModel](name: str, configs_dir: Path, model: type[ModelT]) -> ModelT:
    """Load the configuration called ``name`` from ``configs_dir``.

    Raises:
        ConfigError: For an invalid name, a missing or oversized file, or invalid content.
    """
    path = resolve_config_path(name, configs_dir)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise ConfigError(f"no configuration named {name!r} in {configs_dir}") from None
    if size > MAX_CONFIG_BYTES:
        raise ConfigError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")
    return parse_config(path.read_bytes(), model)


def canonical_json(model: BaseModel) -> str:
    """Deterministic JSON (sorted keys, no insignificant whitespace) for recording in results."""
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
