"""Conditioning and the simulated output pipelines P0, P2 and P3 (docs/methodology.md section 5).

The conditioner hashes non-overlapping blocks of raw samples with SHA-256, a vetted conditioning
function in SP 800-90B section 3.1.5.1.1. Each 512-sample input block carries at least 441 bits of
claimed min-entropy for every model used here (512 x 0.8625), comfortably above the 256-bit output.

* P0 (no monitoring): every raw sample is conditioned and released.
* P2 (proposed): raw samples pass through :class:`HealthMonitor` first. A buffer is conditioned and
  released only if the gate is open after the monitor has processed it.
* P3: conditioned output feeds a stateful ChaCha20-based DRBG stand-in (:func:`drbg_chain`),
  showing how little of the source's behaviour remains visible after a deterministic generator.

The output-test baseline P1 (P0 output examined by the SP 800-22 subset) lives in
``experiments.placement``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from entropylens.monitoring.alerts import Alert, SourceStatus
from entropylens.monitoring.samples import SampleKind, Samples, validate_samples
from entropylens.monitoring.source_health import HealthMonitor
from entropylens.security.crypto_reference import NONCE_BYTES, ChaCha20Reference

CONDITIONING_INPUT_SAMPLES = 512
DIGEST_BYTES = 32


def pack_samples(samples: Samples, kind: SampleKind) -> bytes:
    """Serialise samples to bytes: 8 binary samples per byte (MSB first), or one byte each."""
    return np.packbits(samples).tobytes() if kind == "binary" else samples.tobytes()


def sha256_condition(
    samples: Samples, kind: SampleKind, input_samples: int = CONDITIONING_INPUT_SAMPLES
) -> bytes:
    """SHA-256 of each complete block of ``input_samples`` samples, concatenated.

    A trailing partial block is not conditioned.
    """
    validate_samples(samples, kind)
    if input_samples < 1 or (kind == "binary" and input_samples % 8):
        raise ValueError("input_samples must be positive (and a multiple of 8 for binary sources)")
    blocks = samples.size // input_samples
    return b"".join(
        hashlib.sha256(
            pack_samples(samples[i * input_samples : (i + 1) * input_samples], kind)
        ).digest()
        for i in range(blocks)
    )


def drbg_chain(conditioned: bytes, bytes_per_input: int = DIGEST_BYTES) -> bytes:
    """P3: a stateful DRBG stand-in fed by the conditioned blocks.

    For each 32-byte conditioned block ``d_i``: ``key_i = SHA-256(key_{i-1} || d_i)`` with
    ``key_0`` all zeros, then emit ``bytes_per_input`` bytes of ChaCha20 keystream under
    ``key_i``. Like a real DRBG, the state carries forward, so even a constant input yields output
    that looks statistically perfect. That is precisely what this pipeline demonstrates.
    """
    if len(conditioned) % DIGEST_BYTES:
        raise ValueError("conditioned output must be a whole number of 32-byte blocks")
    if bytes_per_input < 1:
        raise ValueError("bytes_per_input must be positive")
    key = bytes(DIGEST_BYTES)
    output = []
    for i in range(0, len(conditioned), DIGEST_BYTES):
        key = hashlib.sha256(key + conditioned[i : i + DIGEST_BYTES]).digest()
        output.append(ChaCha20Reference.from_key(key, bytes(NONCE_BYTES)).generate(bytes_per_input))
    return b"".join(output)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """What a simulated pipeline released.

    Attributes:
        output: Conditioned output bytes released to the consumer.
        released_samples: Raw samples whose buffers were released.
        withheld_samples: Raw samples whose buffers were blocked by the gate.
        released_faulty_samples: Released raw samples labelled faulty (the *exposure*); ``None``
            when no labels were supplied.
        final_status: Monitor status after the last buffer (``None`` for P0).
        failure: The first latched failure, if any.
    """

    output: bytes
    released_samples: int
    withheld_samples: int
    released_faulty_samples: int | None
    final_status: SourceStatus | None
    failure: Alert | None


def run_pipeline(
    samples: Samples,
    kind: SampleKind,
    monitor: HealthMonitor | None,
    buffer_samples: int,
    faulty: np.ndarray | None = None,
    input_samples: int = CONDITIONING_INPUT_SAMPLES,
) -> PipelineResult:
    """Run P0 (``monitor=None``) or P2 over ``samples`` in buffers of ``buffer_samples``."""
    validate_samples(samples, kind)
    if buffer_samples < 1:
        raise ValueError("buffer_samples must be positive")
    if faulty is not None and faulty.shape != samples.shape:
        raise ValueError("faulty labels must match the samples")
    pending = np.zeros(0, dtype=np.uint8)
    output: list[bytes] = []
    released = withheld = exposed = 0
    status = None
    for start in range(0, samples.size, buffer_samples):
        buffer = samples[start : start + buffer_samples]
        if monitor is not None:
            report = monitor.process(buffer)
            status = report.status
            if not report.gate_open:
                withheld += buffer.size
                continue
        released += buffer.size
        if faulty is not None:
            exposed += int(np.count_nonzero(faulty[start : start + buffer.size]))
        pending = np.concatenate((pending, buffer))
        usable = (pending.size // input_samples) * input_samples
        output.append(sha256_condition(pending[:usable], kind, input_samples))
        pending = pending[usable:]
    return PipelineResult(
        output=b"".join(output),
        released_samples=released,
        withheld_samples=withheld,
        released_faulty_samples=exposed if faulty is not None else None,
        final_status=status,
        failure=monitor.failure if monitor is not None else None,
    )
