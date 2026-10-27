"""Layer C: a deterministic cryptographic reference generator (ChaCha20 keystream).

:class:`ChaCha20Reference` returns the ChaCha20 keystream (RFC 8439), computed by the
``cryptography`` library (OpenSSL backend), by encrypting zero bytes. It serves as a
*deterministic* comparison point next to the OS CSPRNG and the insecure LCG.

What it is not:

* **Not an SP 800-90A DRBG.** It has no instantiate/reseed/update functions, no prediction or
  backtracking resistance, and no health tests. Anyone who learns the key can compute the whole
  keystream for that (key, nonce), past and future.
* **Not a new construction.** The cipher is used exactly as specified: 256-bit key, 96-bit nonce,
  32-bit block counter.

Parameter handling:

* RFC 8439 limits one (key, nonce) pair to 2^32 blocks of 64 bytes (256 GiB). This class tracks the
  remaining keystream and never asks the library to pass the counter limit.
* Random-key mode (the default) draws a fresh key and nonce from the OS CSPRNG and rekeys
  transparently when the limit is reached. Every other mode is a fixed, reproducible stream and
  raises :class:`KeystreamExhaustedError` instead.
* The key is handed to the library and not retained by this object. ``repr`` never shows it, and
  instances refuse to be pickled.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Literal, NoReturn

import cryptography
from cryptography.hazmat.primitives.ciphers import Cipher, CipherContext, algorithms

from entropylens.security.base import GeneratorInfo, validate_request_size
from entropylens.security.secure_source import SecureSource

KEY_BYTES = 32
NONCE_BYTES = 12
BLOCK_BYTES = 64
COUNTER_LIMIT = 1 << 32
"""Number of distinct 32-bit block-counter values per (key, nonce)."""

SEED_LIMIT = 1 << 128
"""Experiment seeds for :meth:`ChaCha20Reference.from_seed` must lie in ``[0, 2^128)``."""

NEGATIVE_CONTROL_KEY = bytes(range(KEY_BYTES))
"""Published key 00 01 … 1f (the RFC 8439 example key) for negative controls. ZERO secrecy."""

NEGATIVE_CONTROL_NONCE = bytes(NONCE_BYTES)

_SEEDED_KEY_LABEL = b"entropylens/chacha20-experiment-key/v1"

KeyMode = Literal["random", "seeded", "explicit", "negative-control"]

_MODE_INFO: dict[KeyMode, tuple[str, str]] = {
    "random": ("chacha20-random-key", "ChaCha20 keystream (random key from OS CSPRNG)"),
    "seeded": ("chacha20-seeded", "ChaCha20 keystream (key derived from experiment seed)"),
    "explicit": ("chacha20-explicit-key", "ChaCha20 keystream (caller-supplied key)"),
    "negative-control": (
        "chacha20-negative-control",
        "ChaCha20 keystream (PUBLISHED key: zero secrecy, negative control)",
    ),
}


class KeystreamExhaustedError(RuntimeError):
    """A fixed (key, nonce) stream has no keystream left before the 32-bit counter would wrap."""


def _check_bytes(value: object, length: int, what: str) -> bytes:
    if not isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(f"{what} must be bytes-like, not {type(value).__name__}")
    data = bytes(value)
    if len(data) != length:
        raise ValueError(f"{what} must be exactly {length} bytes, got {len(data)}")
    return data


def _check_counter(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"initial_counter must be an int, not {type(value).__name__}")
    if not 0 <= value < COUNTER_LIMIT:
        raise ValueError(f"initial_counter must satisfy 0 <= counter < 2^32, got {value}")
    return value


class ChaCha20Reference:
    """Deterministic ChaCha20 keystream generator (``security_status = reference-deterministic``).

    Use the constructors that state intent:

    * ``ChaCha20Reference()``: random key and nonce from the OS CSPRNG (default).
    * :meth:`from_key`: caller-supplied key and nonce, for example RFC 8439 test vectors.
    * :meth:`from_seed`: reproducible experiments. The key is derived from a recorded seed, so
      anyone who knows the seed knows the stream.
    * :meth:`negative_control`: the published :data:`NEGATIVE_CONTROL_KEY`.

    Output is a continuous stream across ``generate`` calls. Instances are not thread-safe.
    """

    __slots__ = ("_encryptor", "_info", "_mode", "_rekeys", "_remaining")
    _encryptor: CipherContext
    _info: GeneratorInfo
    _mode: KeyMode
    _rekeys: int
    _remaining: int

    def __init__(self, *, initial_counter: int = 0) -> None:
        """Create a random-key generator. The key and nonce come from the OS CSPRNG."""
        self._mode = "random"
        self._rekeys = 0
        self._start_stream(*self._fresh_key_and_nonce(), _check_counter(initial_counter))
        self._info = self._make_info()

    @classmethod
    def _with_stream(
        cls, mode: KeyMode, key: bytes, nonce: bytes, initial_counter: int
    ) -> ChaCha20Reference:
        instance = cls.__new__(cls)
        instance._mode = mode
        instance._rekeys = 0
        instance._start_stream(
            _check_bytes(key, KEY_BYTES, "key"),
            _check_bytes(nonce, NONCE_BYTES, "nonce"),
            _check_counter(initial_counter),
        )
        instance._info = instance._make_info()
        return instance

    @classmethod
    def from_key(cls, key: bytes, nonce: bytes, *, initial_counter: int = 0) -> ChaCha20Reference:
        """Keystream for an explicit 32-byte key and 12-byte nonce, starting at ``initial_counter``.

        The caller is responsible for never reusing a (key, nonce) pair where secrecy matters.
        """
        return cls._with_stream("explicit", key, nonce, initial_counter)

    @classmethod
    def from_seed(cls, seed: int) -> ChaCha20Reference:
        """Reproducible keystream with ``key = SHA-256(label || seed)`` and an all-zero nonce.

        For isolated, labelled experiments only: the stream is exactly as secret as the seed, and
        experiment seeds are recorded in result files.
        """
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, not {type(seed).__name__}")
        if not 0 <= seed < SEED_LIMIT:
            raise ValueError("seed must satisfy 0 <= seed < 2^128")
        key = hashlib.sha256(_SEEDED_KEY_LABEL + seed.to_bytes(16, "big")).digest()
        return cls._with_stream("seeded", key, bytes(NONCE_BYTES), 0)

    @classmethod
    def negative_control(cls) -> ChaCha20Reference:
        """Keystream under the published key: statistically excellent, cryptographically void."""
        return cls._with_stream("negative-control", NEGATIVE_CONTROL_KEY, NEGATIVE_CONTROL_NONCE, 0)

    @staticmethod
    def _fresh_key_and_nonce() -> tuple[bytes, bytes]:
        material = SecureSource().generate(KEY_BYTES + NONCE_BYTES)
        return material[:KEY_BYTES], material[KEY_BYTES:]

    def _start_stream(self, key: bytes, nonce: bytes, initial_counter: int) -> None:
        # cryptography's 16-byte "nonce" is the 32-bit little-endian block counter followed by the
        # 96-bit RFC 8439 nonce (state words 12 and 13-15).
        full_nonce = struct.pack("<I", initial_counter) + nonce
        self._encryptor = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None).encryptor()
        self._remaining = (COUNTER_LIMIT - initial_counter) * BLOCK_BYTES

    def _make_info(self) -> GeneratorInfo:
        identifier, display_name = _MODE_INFO[self._mode]
        return GeneratorInfo(
            identifier=identifier,
            display_name=display_name,
            security_status="reference-deterministic",
            deterministic=True,
            backend=f"ChaCha20 (RFC 8439) keystream via cryptography {cryptography.__version__}",
        )

    @property
    def info(self) -> GeneratorInfo:
        """Non-secret description of this generator."""
        return self._info

    @property
    def mode(self) -> KeyMode:
        """How the key was chosen."""
        return self._mode

    @property
    def bytes_remaining(self) -> int:
        """Keystream bytes left for the current (key, nonce) before the counter limit."""
        return self._remaining

    @property
    def rekey_count(self) -> int:
        """Number of transparent rekeys performed (random-key mode only)."""
        return self._rekeys

    def generate(self, n_bytes: int) -> bytes:
        """Return the next ``n_bytes`` bytes of keystream.

        Raises:
            KeystreamExhaustedError: In a fixed-key mode, if fewer than ``n_bytes`` bytes remain.
                No keystream is consumed in that case.
        """
        size = validate_request_size(n_bytes)
        if size <= self._remaining:
            self._remaining -= size
            return self._encryptor.update(bytes(size))
        if self._mode != "random":
            raise KeystreamExhaustedError(
                f"{self._remaining} keystream bytes remain for this (key, nonce); requested {size}"
            )
        parts = []
        needed = size
        while needed > 0:
            if self._remaining == 0:
                self._start_stream(*self._fresh_key_and_nonce(), 0)
                self._rekeys += 1
            take = min(needed, self._remaining)
            parts.append(self._encryptor.update(bytes(take)))
            self._remaining -= take
            needed -= take
        return b"".join(parts)

    def __repr__(self) -> str:
        return f"ChaCha20Reference(mode={self._mode!r}, rekeys={self._rekeys})"

    def __reduce__(self) -> NoReturn:
        raise TypeError("ChaCha20Reference holds key material in its cipher context; not picklable")
