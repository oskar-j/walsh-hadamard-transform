"""What the Netpbm formats share: one-byte samples scaled against ``maxval``.

PPM (``P6``) and PAM (``P7``) differ only in their headers. The raster that
follows is the same run of ``width * height * 3`` samples, one byte each when
``maxval`` is at most 255, red-green-blue and top row first. Decoding and
encoding that raster live here so the two formats cannot drift apart.
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image.base import CHANNELS, PixelArray

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


def read_samples(file: BinaryIO, pixels: int, maxval: int, what: str) -> PixelArray:
    """Read ``pixels`` RGB triples of one-byte samples, rescaled to 0..255.

    The whole raster is decoded in two array operations: one read, and one
    lookup-table pass when ``maxval`` is below 255. Nothing per pixel.

    Args:
        file: Stream positioned at the first sample.
        pixels: How many pixels the header promised. At least one.
        maxval: The declared maximum sample value, in 1..255.
        what: The format name, used only in error messages.

    Returns:
        The pixels as an ``(n, 3)`` ``uint8`` array, top row first.

    Raises:
        UnsupportedFileFormatError: If the data is shorter than promised, or a
            sample exceeds ``maxval``.
    """
    expected = pixels * CHANNELS
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

    return samples.reshape(-1, CHANNELS)


def encode_samples(pixels: PixelArray) -> bytes:
    """Serialise a pixel array as one byte per sample, in order.

    Args:
        pixels: The pixels as a ``uint8`` array, top row first.

    Returns:
        ``3 * n`` bytes.
    """
    return np.ascontiguousarray(pixels).tobytes()
