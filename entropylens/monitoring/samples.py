"""Raw noise-sample representation shared by sources, detectors and the monitor.

Samples are one-dimensional ``uint8`` arrays. Binary sources use the values 0 and 1, one sample per
array element. Byte sources use 0..255.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

SampleKind = Literal["binary", "byte"]
Samples = npt.NDArray[np.uint8]

BITS_PER_SAMPLE: dict[SampleKind, int] = {"binary": 1, "byte": 8}


def validate_samples(samples: object, kind: SampleKind) -> Samples:
    """Return ``samples`` if it is a valid sample array for ``kind``, else raise.

    Raises:
        TypeError: If ``samples`` is not a one-dimensional ``uint8`` NumPy array.
        ValueError: If a binary array contains values other than 0 and 1.
    """
    if not isinstance(samples, np.ndarray) or samples.dtype != np.uint8 or samples.ndim != 1:
        raise TypeError("samples must be a one-dimensional numpy array of dtype uint8")
    if kind == "binary" and samples.size and int(samples.max()) > 1:
        raise ValueError("binary samples must be 0 or 1")
    return samples
