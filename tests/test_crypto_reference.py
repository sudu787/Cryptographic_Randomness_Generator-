"""Tests for Layer C: the ChaCha20 keystream reference generator.

Known-answer tests use RFC 8439 vectors, transcribed from the RFC text and cross-checked against the
`cryptography` library before being committed here. Random-key behaviour is tested only through
lengths, inequality, and delegation to the secure source, never through specific values.
"""

from __future__ import annotations

import copy
import hashlib
import pickle

import pytest

from entropylens.security import crypto_reference
from entropylens.security.base import ByteGenerator
from entropylens.security.crypto_reference import (
    BLOCK_BYTES,
    COUNTER_LIMIT,
    NEGATIVE_CONTROL_KEY,
    ChaCha20Reference,
    KeystreamExhaustedError,
)

RFC_KEY = bytes(range(32))  # 00 01 02 ... 1f

# (key, nonce, initial block counter, serialized 64-byte keystream block)
RFC8439_BLOCKS = {
    "section-2.3.2": (
        RFC_KEY,
        bytes.fromhex("000000090000004a00000000"),
        1,
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e",
    ),
    "appendix-A.1-vector-1": (
        bytes(32),
        bytes(12),
        0,
        "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7"
        "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586",
    ),
    "appendix-A.1-vector-2": (
        bytes(32),
        bytes(12),
        1,
        "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed"
        "29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f",
    ),
    "appendix-A.1-vector-3": (
        bytes(31) + b"\x01",
        bytes(12),
        1,
        "3aeb5224ecf849929b9d828db1ced4dd832025e8018b8160b82284f3c949aa5a"
        "8eca00bbb4a73bdad192b5c42f73f2fd4e273644c8b36125a64addeb006c13a0",
    ),
}

SUNSCREEN_PLAINTEXT = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "
    b"sunscreen would be it."
)
SUNSCREEN_CIPHERTEXT = (
    "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
    "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
    "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
    "5af90bbf74a35be6b40b8eedf2785e42874d"
)


# --- Known-answer tests ---


@pytest.mark.parametrize("vector", sorted(RFC8439_BLOCKS))
def test_rfc8439_block_vectors(vector: str) -> None:
    key, nonce, counter, expected = RFC8439_BLOCKS[vector]
    generator = ChaCha20Reference.from_key(key, nonce, initial_counter=counter)
    assert generator.generate(BLOCK_BYTES).hex() == expected


def test_rfc8439_section_2_4_2_encryption_vector() -> None:
    nonce = bytes.fromhex("000000000000004a00000000")
    keystream = ChaCha20Reference.from_key(RFC_KEY, nonce, initial_counter=1).generate(
        len(SUNSCREEN_PLAINTEXT)
    )
    ciphertext = bytes(p ^ k for p, k in zip(SUNSCREEN_PLAINTEXT, keystream, strict=True))
    assert ciphertext.hex() == SUNSCREEN_CIPHERTEXT


def test_consecutive_blocks_follow_the_counter() -> None:
    # A.1 vectors 1 and 2 are blocks 0 and 1 of the same (key, nonce).
    two_blocks = ChaCha20Reference.from_key(bytes(32), bytes(12)).generate(2 * BLOCK_BYTES)
    assert two_blocks.hex() == (
        RFC8439_BLOCKS["appendix-A.1-vector-1"][3] + RFC8439_BLOCKS["appendix-A.1-vector-2"][3]
    )


@pytest.mark.parametrize("splits", [(10, 118), (1, 63, 64), (64, 0, 64), (127, 1)])
def test_keystream_is_continuous_across_calls(splits: tuple[int, ...]) -> None:
    generator = ChaCha20Reference.from_key(RFC_KEY, bytes(12))
    pieces = b"".join(generator.generate(n) for n in splits)
    assert pieces == ChaCha20Reference.from_key(RFC_KEY, bytes(12)).generate(sum(splits))


# --- Key modes ---


def test_random_key_instances_differ() -> None:
    # Distinct random 256-bit keys give equal 32-byte prefixes with negligible probability.
    assert ChaCha20Reference().generate(32) != ChaCha20Reference().generate(32)


def test_random_key_and_nonce_come_from_the_secure_source(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[int] = []

    class ZeroSource:
        def generate(self, n_bytes: int) -> bytes:
            requests.append(n_bytes)
            return bytes(n_bytes)

    monkeypatch.setattr(crypto_reference, "SecureSource", ZeroSource)
    generator = ChaCha20Reference()
    assert requests == [44]  # 32-byte key + 12-byte nonce
    assert generator.generate(64).hex() == RFC8439_BLOCKS["appendix-A.1-vector-1"][3]


def test_seeded_mode_is_reproducible_and_matches_documented_derivation() -> None:
    derived_key = hashlib.sha256(
        b"entropylens/chacha20-experiment-key/v1" + (42).to_bytes(16, "big")
    ).digest()
    expected = ChaCha20Reference.from_key(derived_key, bytes(12)).generate(256)

    assert ChaCha20Reference.from_seed(42).generate(256) == expected
    assert ChaCha20Reference.from_seed(42).generate(256) == expected
    assert ChaCha20Reference.from_seed(43).generate(256) != expected


def test_negative_control_uses_the_published_key() -> None:
    control = ChaCha20Reference.negative_control()
    assert NEGATIVE_CONTROL_KEY == RFC_KEY
    assert control.generate(128) == ChaCha20Reference.from_key(RFC_KEY, bytes(12)).generate(128)
    assert control.info.identifier == "chacha20-negative-control"
    assert "zero secrecy" in control.info.display_name.lower()


@pytest.mark.parametrize(
    ("factory", "mode", "identifier"),
    [
        (lambda: ChaCha20Reference(), "random", "chacha20-random-key"),
        (lambda: ChaCha20Reference.from_seed(1), "seeded", "chacha20-seeded"),
        (
            lambda: ChaCha20Reference.from_key(RFC_KEY, bytes(12)),
            "explicit",
            "chacha20-explicit-key",
        ),
        (ChaCha20Reference.negative_control, "negative-control", "chacha20-negative-control"),
    ],
)
def test_info_identifies_mode_and_deterministic_reference_status(
    factory: object, mode: str, identifier: str
) -> None:
    generator = factory()  # type: ignore[operator]
    assert isinstance(generator, ByteGenerator)
    assert generator.mode == mode
    assert generator.info.identifier == identifier
    assert generator.info.security_status == "reference-deterministic"
    assert generator.info.deterministic is True
    assert "RFC 8439" in generator.info.backend


# --- Validation ---


@pytest.mark.parametrize(
    ("key", "nonce", "error", "match"),
    [
        (bytes(31), bytes(12), ValueError, "key must be exactly 32"),
        (bytes(33), bytes(12), ValueError, "key must be exactly 32"),
        (RFC_KEY, bytes(8), ValueError, "nonce must be exactly 12"),
        ("00" * 32, bytes(12), TypeError, "key must be bytes-like"),
        (RFC_KEY, None, TypeError, "nonce must be bytes-like"),
    ],
)
def test_key_and_nonce_are_validated(
    key: object, nonce: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        ChaCha20Reference.from_key(key, nonce)  # type: ignore[arg-type]


def test_bytearray_and_memoryview_keys_are_accepted() -> None:
    expected = ChaCha20Reference.from_key(RFC_KEY, bytes(12)).generate(32)
    assert ChaCha20Reference.from_key(bytearray(RFC_KEY), bytes(12)).generate(32) == expected
    assert ChaCha20Reference.from_key(memoryview(RFC_KEY), bytes(12)).generate(32) == expected


@pytest.mark.parametrize(
    ("counter", "error"), [(-1, ValueError), (COUNTER_LIMIT, ValueError), (True, TypeError)]
)
def test_initial_counter_is_validated(counter: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="initial_counter"):
        ChaCha20Reference.from_key(RFC_KEY, bytes(12), initial_counter=counter)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("seed", "error"), [(-1, ValueError), (1 << 128, ValueError), ("1", TypeError)]
)
def test_seed_is_validated(seed: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="seed"):
        ChaCha20Reference.from_seed(seed)  # type: ignore[arg-type]


def test_generate_validates_request_size() -> None:
    generator = ChaCha20Reference.from_seed(0)
    with pytest.raises(ValueError, match="non-negative"):
        generator.generate(-1)
    with pytest.raises(TypeError, match="integer"):
        generator.generate(1.5)  # type: ignore[arg-type]


# --- Counter limit and rekeying ---


def test_fixed_key_stream_stops_at_the_counter_limit_without_consuming() -> None:
    near_end = COUNTER_LIMIT - 2  # exactly two blocks (128 bytes) remain
    generator = ChaCha20Reference.from_key(RFC_KEY, bytes(12), initial_counter=near_end)
    reference = ChaCha20Reference.from_key(RFC_KEY, bytes(12), initial_counter=near_end)
    full = reference.generate(128)

    assert generator.bytes_remaining == 128
    assert generator.generate(100) == full[:100]
    with pytest.raises(KeystreamExhaustedError, match="28 keystream bytes remain"):
        generator.generate(29)
    assert generator.generate(28) == full[100:]  # the failed request consumed nothing
    assert generator.bytes_remaining == 0
    assert generator.generate(0) == b""
    with pytest.raises(KeystreamExhaustedError):
        generator.generate(1)


def test_random_key_mode_rekeys_transparently_at_the_counter_limit() -> None:
    generator = ChaCha20Reference(initial_counter=COUNTER_LIMIT - 1)  # one block remains
    assert generator.bytes_remaining == BLOCK_BYTES

    output = generator.generate(100)

    assert len(output) == 100
    assert generator.rekey_count == 1
    assert generator.bytes_remaining == COUNTER_LIMIT * BLOCK_BYTES - 36


# --- Key confidentiality ---


def test_repr_does_not_reveal_the_key() -> None:
    text = repr(ChaCha20Reference.from_key(RFC_KEY, bytes(12))) + repr(
        ChaCha20Reference.negative_control().info
    )
    assert RFC_KEY.hex() not in text
    assert repr(RFC_KEY) not in text


def test_key_is_not_stored_as_an_attribute() -> None:
    generator = ChaCha20Reference.from_key(RFC_KEY, bytes(12))
    values = [getattr(generator, slot) for slot in ChaCha20Reference.__slots__]
    assert all(value != RFC_KEY for value in values)
    assert not hasattr(generator, "__dict__")


@pytest.mark.parametrize("clone", [pickle.dumps, copy.copy, copy.deepcopy])
def test_instances_refuse_serialisation_and_copying(clone: object) -> None:
    with pytest.raises(TypeError, match="not picklable"):
        clone(ChaCha20Reference.from_seed(7))  # type: ignore[operator]
