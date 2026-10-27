"""Shared generator interface and request validation.

Every byte generator in EntropyLens (production, educational, or deterministic reference) exposes
the same two members, ``info`` and ``generate``. Benchmarks, statistics, and the dashboard can then
treat them uniformly, while :attr:`GeneratorInfo.security_status` keeps their very different
security properties explicit wherever their output is shown or recorded.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal, Protocol, SupportsIndex, runtime_checkable

SecurityStatus = Literal["production", "educational-insecure", "reference-deterministic"]

MAX_REQUEST_BYTES = 64 * 1024 * 1024
"""Upper bound on a single ``generate`` request (64 MiB).

This guards against accidental resource exhaustion from dashboard inputs or experiment
configurations. It is not a cryptographic limit; larger volumes are produced by repeated calls.
"""


@dataclass(frozen=True, slots=True)
class GeneratorInfo:
    """Non-secret, auditable description of a generator.

    Instances are safe to log and to store in experiment metadata. They never contain seeds, keys,
    internal state, or output.

    Attributes:
        identifier: Stable machine-readable ID recorded in experiment results.
        display_name: Human-readable name for the dashboard and reports.
        security_status: ``"production"`` only for the OS-backed secure source.
        deterministic: True if the output is a function of a seed or key.
        backend: Description of the underlying mechanism.
    """

    identifier: str
    display_name: str
    security_status: SecurityStatus
    deterministic: bool
    backend: str


@runtime_checkable
class ByteGenerator(Protocol):
    """Interface implemented by every byte generator."""

    @property
    def info(self) -> GeneratorInfo:
        """Non-secret description of this generator."""
        ...

    def generate(self, n_bytes: int) -> bytes:
        """Return exactly ``n_bytes`` bytes of output."""
        ...


def validate_request_size(n_bytes: object, *, limit: int = MAX_REQUEST_BYTES) -> int:
    """Validate a requested output length and return it as a plain ``int``.

    Integer-like objects (anything implementing ``__index__``, such as NumPy integers) are accepted.
    Booleans are rejected even though ``bool`` subclasses ``int``, because ``generate(True)`` is
    almost certainly a bug.

    Args:
        n_bytes: The requested number of bytes.
        limit: Maximum permitted request size in bytes.

    Returns:
        The validated size.

    Raises:
        TypeError: If ``n_bytes`` is not integer-like, or is a boolean.
        ValueError: If ``n_bytes`` is negative or exceeds ``limit``.
    """
    if isinstance(n_bytes, bool) or not isinstance(n_bytes, SupportsIndex):
        raise TypeError(f"n_bytes must be an integer, not {type(n_bytes).__name__}")
    size = operator.index(n_bytes)
    if size < 0:
        raise ValueError(f"n_bytes must be non-negative, got {size}")
    if size > limit:
        raise ValueError(f"n_bytes={size} exceeds the per-request limit of {limit} bytes")
    return size
