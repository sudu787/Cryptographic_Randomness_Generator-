"""Tests for Layer A: the production secure source.

These tests exercise the production path. They check lengths, types, error handling, and the
security properties of the interface. They never assert specific output values or a specific output
distribution, which would make them flaky. Deterministic statistical tests live in separate modules.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from entropylens.security import secure_source
from entropylens.security.base import MAX_REQUEST_BYTES, ByteGenerator, validate_request_size
from entropylens.security.secure_source import SecureSource, SecureSourceError

SECURITY_PACKAGE_DIR = Path(secure_source.__file__).parent


class _IndexLike:
    """Integer-like object, as NumPy integers are, without importing NumPy in Layer A tests."""

    def __index__(self) -> int:
        return 8


@pytest.fixture
def source() -> SecureSource:
    return SecureSource()


# --- Output length and type ---------------------------------------------------------------------


@pytest.mark.parametrize("n_bytes", [0, 1, 16, 32, 1000, 4096])
def test_generate_returns_bytes_of_requested_length(source: SecureSource, n_bytes: int) -> None:
    output = source.generate(n_bytes)
    assert type(output) is bytes
    assert len(output) == n_bytes


def test_generate_accepts_integer_like_sizes(source: SecureSource) -> None:
    assert len(source.generate(_IndexLike())) == 8  # type: ignore[arg-type]


def test_consecutive_outputs_differ(source: SecureSource) -> None:
    # Two independent 32-byte outputs are equal with probability 2**-256. Equality would indicate a
    # constant or cached source, not bad luck.
    assert source.generate(32) != source.generate(32)


# --- Input validation ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_bytes", [-1, -4096])
def test_negative_sizes_are_rejected(source: SecureSource, n_bytes: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        source.generate(n_bytes)


@pytest.mark.parametrize("n_bytes", [16.0, "16", None, b"\x10", True, False])
def test_non_integer_sizes_are_rejected(source: SecureSource, n_bytes: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        source.generate(n_bytes)  # type: ignore[arg-type]


def test_oversized_request_is_rejected_before_calling_the_os(
    source: SecureSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_be_called(n_bytes: int) -> bytes:
        raise AssertionError("the OS CSPRNG must not be called for an invalid request")

    monkeypatch.setattr(secure_source.secrets, "token_bytes", must_not_be_called)
    with pytest.raises(ValueError, match="exceeds"):
        source.generate(MAX_REQUEST_BYTES + 1)


def test_request_limit_is_inclusive() -> None:
    assert validate_request_size(MAX_REQUEST_BYTES) == MAX_REQUEST_BYTES
    assert validate_request_size(10, limit=10) == 10
    with pytest.raises(ValueError, match="exceeds"):
        validate_request_size(11, limit=10)


def test_validated_size_is_a_plain_int() -> None:
    assert type(validate_request_size(_IndexLike())) is int


# --- Delegation and failure handling -------------------------------------------------------------


def test_generation_is_delegated_to_secrets(
    source: SecureSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_token_bytes(n_bytes: int) -> bytes:
        calls.append(n_bytes)
        return bytes(n_bytes)

    monkeypatch.setattr(secure_source.secrets, "token_bytes", fake_token_bytes)
    assert source.generate(5) == bytes(5)
    assert calls == [5]


@pytest.mark.parametrize(
    "failure", [OSError("entropy unavailable"), NotImplementedError("no randomness source")]
)
def test_os_failures_raise_secure_source_error_without_fallback(
    source: SecureSource, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def failing_token_bytes(n_bytes: int) -> bytes:
        raise failure

    monkeypatch.setattr(secure_source.secrets, "token_bytes", failing_token_bytes)
    with pytest.raises(SecureSourceError) as excinfo:
        source.generate(16)
    assert excinfo.value.__cause__ is failure


# --- Auditable metadata and interface -----------------------------------------------------------


def test_info_describes_a_production_non_deterministic_source(source: SecureSource) -> None:
    info = source.info
    assert info.identifier == "os-csprng"
    assert info.security_status == "production"
    assert info.deterministic is False
    assert "os.urandom" in info.backend


def test_info_is_immutable(source: SecureSource) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.info.deterministic = True  # type: ignore[misc]


def test_satisfies_byte_generator_protocol(source: SecureSource) -> None:
    assert isinstance(source, ByteGenerator)


def test_exposes_no_seeding_or_state_api(source: SecureSource) -> None:
    forbidden = {"seed", "reseed", "getstate", "setstate", "state", "set_seed"}
    assert forbidden.isdisjoint(dir(source))


def test_output_is_not_retained_in_repr_or_metadata(
    source: SecureSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = bytes.fromhex("deadbeefcafef00d")
    monkeypatch.setattr(secure_source.secrets, "token_bytes", lambda n_bytes: marker[:n_bytes])
    source.generate(len(marker))
    rendered = (repr(source) + repr(source.info)).lower()
    assert marker.hex() not in rendered
    assert repr(marker) not in rendered


# --- Static security guard -----------------------------------------------------------------------


def _imported_modules(tree: ast.AST) -> set[str]:
    """Fully qualified names of every module imported in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_security_package_never_imports_a_non_cryptographic_random_module() -> None:
    scanned = sorted(SECURITY_PACKAGE_DIR.glob("*.py"))
    assert SECURITY_PACKAGE_DIR / "secure_source.py" in scanned  # guard against a vacuous pass

    for path in scanned:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offending = {
            name
            for name in _imported_modules(tree)
            if name == "random" or name.startswith("random.") or name.endswith(".random")
        }
        assert not offending, f"{path.name} imports non-cryptographic randomness: {offending}"
