"""Generators offered in the dashboard, with what may be shown of their output.

Output previews and downloads are offered only where the output is explicitly *not secret*:
deterministic educational or reference streams whose seed or key is known. Output of the OS CSPRNG,
and of ChaCha20 under a fresh random key, is analysed statistically but never displayed or offered
for download.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from entropylens.security.base import ByteGenerator
from entropylens.security.crypto_reference import ChaCha20Reference
from entropylens.security.educational_lcg import DRAND48_HIGH32, EducationalLCG, VectorizedLCG
from entropylens.security.secure_source import SecureSource


@dataclass(frozen=True, slots=True)
class GeneratorChoice:
    """One selectable generator.

    Attributes:
        key: Stable widget value.
        label: Text shown in the selector.
        factory: Builds the generator from a seed (ignored where ``seeded`` is False).
        seeded: Whether the seed input applies.
        reveal_output: Whether a preview and a download may be shown.
    """

    key: str
    label: str
    factory: Callable[[int], ByteGenerator]
    seeded: bool
    reveal_output: bool


GENERATORS: tuple[GeneratorChoice, ...] = (
    GeneratorChoice(
        "os-csprng", "Operating-system CSPRNG (secrets)", lambda _: SecureSource(), False, False
    ),
    GeneratorChoice(
        "lcg-nr32",
        "Educational LCG, 32-bit full state (INSECURE)",
        lambda seed: EducationalLCG(seed % (1 << 32)),
        True,
        True,
    ),
    GeneratorChoice(
        "lcg-drand48",
        "Educational LCG, drand48 top 32 bits (INSECURE)",
        lambda seed: EducationalLCG(seed % (1 << 48), DRAND48_HIGH32),
        True,
        True,
    ),
    GeneratorChoice(
        "lcg-numpy",
        "Educational LCG, NumPy-vectorised (INSECURE)",
        lambda seed: VectorizedLCG(seed % (1 << 32)),
        True,
        True,
    ),
    GeneratorChoice(
        "chacha20-random",
        "ChaCha20 keystream, fresh random key",
        lambda _: ChaCha20Reference(),
        False,
        False,
    ),
    GeneratorChoice(
        "chacha20-seeded",
        "ChaCha20 keystream, key from an experiment seed (zero secrecy)",
        ChaCha20Reference.from_seed,
        True,
        True,
    ),
    GeneratorChoice(
        "chacha20-published",
        "ChaCha20 keystream, published key (negative control)",
        lambda _: ChaCha20Reference.negative_control(),
        False,
        True,
    ),
)

BY_KEY = {choice.key: choice for choice in GENERATORS}
