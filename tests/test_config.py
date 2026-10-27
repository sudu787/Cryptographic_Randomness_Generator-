"""Tests for validated JSON experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest

from entropylens.config import (
    MAX_CONFIG_BYTES,
    ChaCha20Config,
    ConfigError,
    ExperimentConfig,
    LCGConfig,
    SecureSourceConfig,
    canonical_json,
    load_config,
    parse_config,
    resolve_config_path,
    validate_name,
)
from entropylens.security import ChaCha20Reference, EducationalLCG, SecureSource
from entropylens.security.educational_lcg import PRESETS


def _config(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "name": "e0-generator-smoke",
        "purpose": "development",
        "master_seed": 20261026,
        "description": "Phase 2 smoke configuration",
        "generators": [
            {"kind": "os-csprng"},
            {"kind": "lcg", "preset": "drand48-high32", "seed": 2**40},
            {"kind": "chacha20", "key_mode": "seeded", "seed": 7},
            {"kind": "chacha20", "key_mode": "negative-control"},
            {"kind": "chacha20"},
        ],
    }
    document.update(overrides)
    return document


def _parse(document: dict[str, Any]) -> ExperimentConfig:
    return parse_config(json.dumps(document), ExperimentConfig)


# --- Parsing and building ---


def test_valid_configuration_parses_and_builds_every_generator() -> None:
    config = _parse(_config())
    built = [generator.build() for generator in config.generators]

    assert isinstance(built[0], SecureSource)
    assert isinstance(built[1], EducationalLCG)
    assert built[1].params.name == "drand48-high32"
    assert [g.info.identifier for g in built[2:]] == [
        "chacha20-seeded",
        "chacha20-negative-control",
        "chacha20-random-key",
    ]
    assert all(isinstance(g, ChaCha20Reference) for g in built[2:])


def test_deterministic_generators_are_reproducible_from_configuration() -> None:
    first, second = _parse(_config()), _parse(_config())
    for index in (1, 2, 3):  # LCG, seeded ChaCha20, negative control
        stream_a = first.generators[index].build().generate(256)
        stream_b = second.generators[index].build().generate(256)
        assert stream_a == stream_b


def test_defaults_apply_to_optional_fields() -> None:
    config = _parse(_config(description="", generators=[{"kind": "lcg", "seed": 5}]))
    lcg = config.generators[0]
    assert isinstance(lcg, LCGConfig)
    assert lcg.preset == "numerical-recipes-32"
    assert _parse({k: v for k, v in _config().items() if k != "generators"}).generators == ()


def test_lcg_preset_choices_match_the_preset_registry() -> None:
    assert set(get_args(LCGConfig.model_fields["preset"].annotation)) == set(PRESETS)


def test_models_are_immutable() -> None:
    config = _parse(_config())
    with pytest.raises(ValueError, match="frozen"):
        config.master_seed = 1  # type: ignore[misc]


# --- Strict validation ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"master_seed": "20261026"},  # string where an int is required
        {"master_seed": True},  # bool is not an int here
        {"master_seed": 1.0},  # float is not an int here
        {"master_seed": -1},
        {"master_seed": 2**128},
        {"schema_version": 2},
        {"schema_version": True},
        {"purpose": "production"},
        {"name": "E1"},
        {"description": "x" * 2001},
        {"unexpected_field": 1},
        {"generators": [{"kind": "os-csprng", "seed": 1}]},  # the secure source takes no seed
        {"generators": [{"kind": "mersenne-twister"}]},
        {"generators": [{"kind": "lcg"}]},  # seed is required
        {"generators": [{"kind": "lcg", "seed": 2**32}]},  # seed >= modulus of the default preset
        {"generators": [{"kind": "lcg", "preset": "glibc", "seed": 1}]},
        {"generators": [{"kind": "chacha20", "key_mode": "seeded"}]},
        {"generators": [{"kind": "chacha20", "key_mode": "random", "seed": 3}]},
        {"generators": [{"kind": "chacha20", "key_mode": "negative-control", "seed": 3}]},
        {"generators": [{"kind": "chacha20", "key": "00" * 32}]},  # raw keys are never accepted
        {"generators": [{"kind": "os-csprng"}] * 17},
    ],
)
def test_invalid_configurations_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        _parse(_config(**overrides))


def test_missing_required_fields_are_rejected() -> None:
    for field in ("schema_version", "name", "purpose", "master_seed"):
        document = _config()
        del document[field]
        with pytest.raises(ConfigError, match=field):
            _parse(document)


def test_largest_permitted_seeds_are_accepted() -> None:
    config = _parse(
        _config(
            master_seed=2**128 - 1,
            generators=[{"kind": "chacha20", "key_mode": "seeded", "seed": 2**128 - 1}],
        )
    )
    assert config.master_seed == 2**128 - 1


@pytest.mark.parametrize("text", ["", "not json", "[1, 2, 3]", "{'single': 'quotes'}", "null"])
def test_malformed_json_is_rejected(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_config(text, ExperimentConfig)


def test_oversized_text_is_rejected_before_parsing() -> None:
    with pytest.raises(ConfigError, match="exceeds"):
        parse_config(" " * (MAX_CONFIG_BYTES + 1), ExperimentConfig)


def test_individual_generator_models_validate_directly() -> None:
    assert SecureSourceConfig().kind == "os-csprng"
    assert ChaCha20Config(key_mode="seeded", seed=1).seed == 1
    with pytest.raises(ValueError, match="seed is required"):
        ChaCha20Config(key_mode="seeded")


# --- Names, paths and loading ---


@pytest.mark.parametrize("name", ["e1", "e1-detection_v2", "0", "a" * 64])
def test_valid_names(name: str) -> None:
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "E1",
        "a" * 65,
        "-leading-dash",
        "../escape",
        "..",
        "sub/dir",
        "c:\\x",
        "name.json",
        " e1",
        7,
    ],
)
def test_invalid_names(name: object) -> None:
    with pytest.raises(ConfigError, match="invalid name"):
        validate_name(name)


def test_config_path_stays_inside_configs_directory(tmp_path: Path) -> None:
    path = resolve_config_path("e1-detection", tmp_path)
    assert path == (tmp_path / "e1-detection.json").resolve()
    with pytest.raises(ConfigError):
        resolve_config_path("../outside", tmp_path)


def test_load_config_round_trip(tmp_path: Path) -> None:
    (tmp_path / "e0-generator-smoke.json").write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_config("e0-generator-smoke", tmp_path, ExperimentConfig)
    assert loaded == _parse(_config())


def test_load_config_reports_missing_and_oversized_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no configuration named"):
        load_config("absent", tmp_path, ExperimentConfig)
    (tmp_path / "huge.json").write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="exceeds"):
        load_config("huge", tmp_path, ExperimentConfig)


def test_load_config_rejects_invalid_content(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid ExperimentConfig"):
        load_config("broken", tmp_path, ExperimentConfig)


# --- Canonical JSON ---


def test_canonical_json_is_deterministic_and_round_trips() -> None:
    config = _parse(_config())
    text = canonical_json(config)
    reordered = _parse(dict(reversed(list(_config().items()))))

    assert text == canonical_json(reordered)
    assert " " not in text.replace("Phase 2 smoke configuration", "")
    assert parse_config(text, ExperimentConfig) == config
    assert list(json.loads(text)) == sorted(json.loads(text))
