"""What the Netpbm formats share: one-byte samples scaled against ``maxval``.

PPM (``P6``) and PAM (``P7``) differ only in their headers. The raster that
follows is the same run of ``width * height * 3`` samples, one byte each when
``maxval`` is at most 255, red-green-blue and top row first. Decoding and
encoding that raster live here so the two formats cannot drift apart.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import BinaryIO

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image.base import Pixel

__all__ = ["NETPBM_MAX_SAMPLE", "encode_samples", "read_samples", "rescale_sample"]

#: The largest ``maxval`` these readers handle, i.e. one byte per sample.
NETPBM_MAX_SAMPLE = 255


def rescale_sample(value: int, maxval: int, what: str) -> int:
    """Scale one sample from a 0..maxval range to 0..255.

    Args:
        value: The raw sample.
        maxval: The file's declared maximum, in 1..255.
        what: The format name, used only in the error message.

    Returns:
        The sample expressed against a maximum of 255, rounded to nearest.

    Raises:
        UnsupportedFileFormatError: If the sample exceeds ``maxval``, which the
            Netpbm specifications forbid.
    """
    if value > maxval:
        raise UnsupportedFileFormatError(f"invalid {what} sample: {value} exceeds maxval {maxval}")
    if maxval == NETPBM_MAX_SAMPLE:
        return value
    return (value * NETPBM_MAX_SAMPLE + maxval // 2) // maxval


def read_samples(file: BinaryIO, pixels: int, maxval: int, what: str) -> list[Pixel]:
    """Read ``pixels`` RGB triples of one-byte samples, rescaled to 0..255.

    The whole raster is decoded in a handful of array operations: one read,
    one lookup-table pass when ``maxval`` is below 255, and one ``zip`` over
    the three channel columns, which builds the tuples in C.

    Args:
        file: Stream positioned at the first sample.
        pixels: How many pixels the header promised. At least one.
        maxval: The declared maximum sample value, in 1..255.
        what: The format name, used only in error messages.

    Returns:
        The pixels as integer triples, top row first.

    Raises:
        UnsupportedFileFormatError: If the data is shorter than promised, or a
            sample exceeds ``maxval``.
    """
    expected = pixels * 3
    data = file.read(expected)
    if len(data) < expected:
        raise UnsupportedFileFormatError(
            f"truncated {what} data: expected {expected} bytes, got {len(data)}"
        )

    samples = np.frombuffer(data, dtype=np.uint8)
    if maxval != NETPBM_MAX_SAMPLE:
        largest = int(samples.max())
        if largest > maxval:
            raise UnsupportedFileFormatError(
                f"invalid {what} sample: {largest} exceeds maxval {maxval}"
            )
        table = (np.arange(maxval + 1) * NETPBM_MAX_SAMPLE + maxval // 2) // maxval
        samples = table[samples].astype(np.uint8)

    r, g, b = samples.reshape(-1, 3).T.tolist()
    return list(zip(r, g, b, strict=True))


def encode_samples(pixels: Sequence[Pixel]) -> bytes:
    """Serialise RGB triples as one byte per sample, in order.

    ``bytes`` over a chained iterator consumes it in C, about twice as fast
    as a generator expression over the same tuples.

    Args:
        pixels: The pixels, top row first.

    Returns:
        ``3 * len(pixels)`` bytes.
    """
    return bytes(itertools.chain.from_iterable(pixels))
