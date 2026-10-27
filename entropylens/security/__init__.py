"""Randomness sources: the production OS-backed source and clearly labelled deterministic baselines.

Only :class:`SecureSource` (``security_status == "production"``) may supply security-relevant
randomness. :class:`EducationalLCG` is insecure by design. :class:`ChaCha20Reference` is a
deterministic reference generator, not an SP 800-90A DRBG.
"""

from entropylens.security.base import (
    MAX_REQUEST_BYTES,
    ByteGenerator,
    GeneratorInfo,
    SecurityStatus,
    validate_request_size,
)
from entropylens.security.crypto_reference import ChaCha20Reference, KeystreamExhaustedError
from entropylens.security.educational_lcg import (
    DRAND48_HIGH32,
    NUMERICAL_RECIPES_32,
    PRESETS,
    EducationalLCG,
    LCGParameters,
)
from entropylens.security.secure_source import SecureSource, SecureSourceError

__all__ = [
    "DRAND48_HIGH32",
    "MAX_REQUEST_BYTES",
    "NUMERICAL_RECIPES_32",
    "PRESETS",
    "ByteGenerator",
    "ChaCha20Reference",
    "EducationalLCG",
    "GeneratorInfo",
    "KeystreamExhaustedError",
    "LCGParameters",
    "SecureSource",
    "SecureSourceError",
    "SecurityStatus",
    "validate_request_size",
]
