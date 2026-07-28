"""Colour model conversions between RGB and YCbCr."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["ColorModel", "Rgb", "RgbColorModel", "Triple", "YCbCrColorModel"]

#: A pixel as three numbers, in whatever model the class in question uses.
Triple = tuple[float, float, float]

#: A pixel as three 0-255 integers.
Rgb = tuple[int, int, int]


class ColorModel(ABC):
    """Converts a single pixel between its native model and RGB/YCbCr."""

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
        return _as_rgb(*color)

    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Convert an RGB pixel to YCbCr using the ITU-R BT.601 weights.

        Args:
            color: The pixel as ``(r, g, b)``.

        Returns:
            The pixel as ``(y, cb, cr)``, chroma centred on 128.
        """
        r, g, b = color
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return y, cb, cr


class YCbCrColorModel(ColorModel):
    """Pixels stored as YCbCr."""

    def get_rgb(self, color: Triple) -> Rgb:
        """Convert a YCbCr pixel back to RGB.

        Args:
            color: The pixel as ``(y, cb, cr)``, chroma centred on 128.

        Returns:
            The pixel as ``(r, g, b)``, truncated and clamped into 0-255.
            Clamping matters because the YCbCr cube is larger than the RGB one,
            so a lossy round trip can land outside it.
        """
        y, cb, cr = color
        return _as_rgb(
            y + 1.402 * (cr - 128),
            y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128),
            y + 1.772 * (cb - 128),
        )

    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Return the pixel unchanged.

        Args:
            color: A YCbCr pixel.

        Returns:
            The same pixel.
        """
        return color


def _as_rgb(r: float, g: float, b: float) -> Rgb:
    """Truncate towards zero and clamp each channel into the 0-255 range.

    Args:
        r: Red channel, possibly fractional or out of range.
        g: Green channel.
        b: Blue channel.

    Returns:
        The pixel as three 0-255 integers.
    """
    return _clamp(int(r)), _clamp(int(g)), _clamp(int(b))


def _clamp(value: int, low: int = 0, high: int = 255) -> int:
    """Constrain ``value`` to a closed range.

    Args:
        value: The value to constrain.
        low: Lower bound, inclusive.
        high: Upper bound, inclusive.

    Returns:
        ``value`` moved to the nearest bound if it fell outside them.
    """
    return max(low, min(high, value))
