"""Layer A: the production randomness interface.

EntropyLens never generates security-relevant randomness itself. :class:`SecureSource` delegates
every request to :func:`secrets.token_bytes`, which reads from :func:`os.urandom`, which asks the
operating system's CSPRNG:

* Windows (CPython 3.11 and later): ``BCryptGenRandom``.
* Linux: the ``getrandom()`` system call, which blocks until the kernel pool is initialised.
* Other platforms: the CSPRNG interface CPython selects (see the ``os.urandom`` documentation).

Design rules enforced here (``docs/threat_model.md``, threats T1 to T4):

* No seed, no internal state, and no fallback to the non-cryptographic ``random`` module.
* Output goes to the caller only. It is never logged, cached, or persisted.
* OS failures surface as :class:`SecureSourceError`. They are never masked by a weaker source.
"""

from __future__ import annotations

import platform
import secrets

from entropylens.security.base import GeneratorInfo, validate_request_size

_BACKENDS = {
    "Windows": "BCryptGenRandom via CPython os.urandom",
    "Linux": "getrandom() system call via CPython os.urandom",
}


class SecureSourceError(RuntimeError):
    """The operating system's CSPRNG could not satisfy a request."""


class SecureSource:
    """Stateless access to the operating-system CSPRNG.

    Instances hold no randomness-related state, so they are safe to share between threads. Two
    instances are interchangeable.
    """

    __slots__ = ("_info",)

    def __init__(self) -> None:
        system = platform.system()
        self._info = GeneratorInfo(
            identifier="os-csprng",
            display_name="Operating-system CSPRNG (secrets)",
            security_status="production",
            deterministic=False,
            backend=_BACKENDS.get(system, f"platform CSPRNG via CPython os.urandom ({system})"),
        )

    @property
    def info(self) -> GeneratorInfo:
        """Non-secret description of the source, safe to log and record."""
        return self._info

    def generate(self, n_bytes: int) -> bytes:
        """Return ``n_bytes`` bytes from the operating-system CSPRNG.

        Args:
            n_bytes: Number of bytes to return (0 to ``MAX_REQUEST_BYTES`` inclusive).

        Returns:
            Fresh random bytes. Treat them as secret if they are used as key material.

        Raises:
            TypeError: If ``n_bytes`` is not an integer.
            ValueError: If ``n_bytes`` is negative or too large. The OS is not called.
            SecureSourceError: If the operating system cannot supply randomness.
        """
        size = validate_request_size(n_bytes)
        try:
            return secrets.token_bytes(size)
        except (OSError, NotImplementedError) as exc:
            # os.urandom raises NotImplementedError when no randomness source exists.
            raise SecureSourceError(
                "the operating-system CSPRNG failed; no fallback source is used"
            ) from exc

    def __repr__(self) -> str:
        return f"SecureSource(backend={self._info.backend!r})"
