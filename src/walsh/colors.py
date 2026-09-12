"""Colour model conversions between RGB and YCbCr.

The whole-array functions :func:`rgb_to_ycbcr` and :func:`ycbcr_to_rgb` are the
implementation; they convert every pixel of an image in a handful of numpy
operations. The :class:`ColorModel` classes are the per-pixel interface the
package has always had, and each method is a one-pixel call into the same
arithmetic, so the two can never disagree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

__all__ = [
    "ColorModel",
    "Rgb",
    "RgbColorModel",
    "Triple",
    "YCbCrColorModel",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
]

#: A pixel as three numbers, in whatever model the class in question uses.
Triple = tuple[float, float, float]

#: A pixel as three 0-255 integers.
Rgb = tuple[int, int, int]

#: Chroma is stored offset so that neutral grey sits at this value.
CHROMA_OFFSET = 128

#: The range every RGB channel is clamped into.
RGB_MIN = 0
RGB_MAX = 255


def rgb_to_ycbcr(pixels: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert RGB pixels to YCbCr using the ITU-R BT.601 weights.

    Args:
        pixels: An array whose last axis holds ``(r, g, b)``: shape ``(3,)``
            for one pixel, ``(n, 3)`` for an image. Integer input is fine.

    Returns:
        A float64 array of the same shape holding ``(y, cb, cr)``. Values are
        unrounded, and chroma is centred on 128.
    """
    rgb = np.asarray(pixels, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ycbcr = np.empty_like(rgb)
    ycbcr[..., 0] = 0.299 * r + 0.587 * g + 0.114 * b
    ycbcr[..., 1] = CHROMA_OFFSET - 0.168736 * r - 0.331264 * g + 0.5 * b
    ycbcr[..., 2] = CHROMA_OFFSET + 0.5 * r - 0.418688 * g - 0.081312 * b
    return ycbcr


def ycbcr_to_rgb(pixels: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Convert YCbCr pixels back to RGB, truncated and clamped into 0-255.

    Clamping matters because the YCbCr cube is larger than the RGB one, so a
    lossy round trip can land outside it.

    Args:
        pixels: An array whose last axis holds ``(y, cb, cr)``, chroma
            centred on 128: shape ``(3,)`` for one pixel, ``(n, 3)`` for an
            image.

    Returns:
        A ``uint8`` array of the same shape holding ``(r, g, b)``.
    """
    ycbcr = np.asarray(pixels, dtype=np.float64)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    rgb = np.empty_like(ycbcr)
    rgb[..., 0] = y + 1.402 * (cr - CHROMA_OFFSET)
    rgb[..., 1] = y - 0.34414 * (cb - CHROMA_OFFSET) - 0.71414 * (cr - CHROMA_OFFSET)
    rgb[..., 2] = y + 1.772 * (cb - CHROMA_OFFSET)
    return _truncate_and_clamp(rgb)


def _truncate_and_clamp(values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Truncate towards zero, as ``int()`` does, then clamp into 0-255.

    Args:
        values: Channel values, possibly fractional or out of range.

    Returns:
        The same shape as ``uint8``.
    """
    clamped: npt.NDArray[np.uint8] = np.clip(np.trunc(values), RGB_MIN, RGB_MAX).astype(np.uint8)
    return clamped


class ColorModel(ABC):
    """Converts a single pixel between its native model and RGB/YCbCr.

    Per-pixel calls are convenient for a handful of pixels. For an image, use
    :func:`rgb_to_ycbcr` or :func:`ycbcr_to_rgb` on the whole array instead;
    these methods each go through numpy for one pixel at a time.
    """

    @abstractmethod
    def get_rgb(self, color: Triple) -> Rgb:
        """Convert one pixel from this model to RGB.

        Args:
            color: The pixel, in whatever model this class represents.

        Returns:
            The pixel as ``(r, g, b)``, each channel a 0-255 integer.
        """

    @abstractmethod
    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Convert one pixel from this model to YCbCr.

        Args:
            color: The pixel, in whatever model this class represents.

        Returns:
            The pixel as ``(y, cb, cr)``. Values are unrounded floats, and
            chroma is centred on 128.
        """


class RgbColorModel(ColorModel):
    """Pixels stored as RGB."""

    def get_rgb(self, color: Triple) -> Rgb:
        """Return the pixel unchanged, as integers.

        Args:
            color: An RGB pixel.

        Returns:
            The same pixel, truncated and clamped into 0-255.
        """
        r, g, b = _truncate_and_clamp(np.asarray(color, dtype=np.float64)).tolist()
        return r, g, b

    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Convert an RGB pixel to YCbCr using the ITU-R BT.601 weights.

        Args:
            color: The pixel as ``(r, g, b)``.

        Returns:
            The pixel as ``(y, cb, cr)``, chroma centred on 128.
        """
        y, cb, cr = rgb_to_ycbcr(color).tolist()
        return y, cb, cr


class YCbCrColorModel(ColorModel):
    """Pixels stored as YCbCr."""

    def get_rgb(self, color: Triple) -> Rgb:
        """Convert a YCbCr pixel back to RGB.

        Args:
            color: The pixel as ``(y, cb, cr)``, chroma centred on 128.

        Returns:
            The pixel as ``(r, g, b)``, truncated and clamped into 0-255.
        """
        r, g, b = ycbcr_to_rgb(color).tolist()
        return r, g, b

    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Return the pixel unchanged.

        Args:
            color: A YCbCr pixel.

        Returns:
            The same pixel.
        """
        return color
