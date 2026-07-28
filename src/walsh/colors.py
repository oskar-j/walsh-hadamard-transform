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
        """Return ``color`` as an ``(r, g, b)`` triple of 0-255 ints."""

    @abstractmethod
    def get_y_cb_cr(self, color: Triple) -> Triple:
        """Return ``color`` as a ``(y, cb, cr)`` triple."""


class RgbColorModel(ColorModel):
    """Pixels stored as RGB."""

    def get_rgb(self, color: Triple) -> Rgb:
        return _as_rgb(*color)

    def get_y_cb_cr(self, color: Triple) -> Triple:
        r, g, b = color
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return y, cb, cr


class YCbCrColorModel(ColorModel):
    """Pixels stored as YCbCr."""

    def get_rgb(self, color: Triple) -> Rgb:
        y, cb, cr = color
        return _as_rgb(
            y + 1.402 * (cr - 128),
            y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128),
            y + 1.772 * (cb - 128),
        )

    def get_y_cb_cr(self, color: Triple) -> Triple:
        return color


def _as_rgb(r: float, g: float, b: float) -> Rgb:
    """Truncate towards zero and clamp each channel into the 0-255 range."""
    return _clamp(int(r)), _clamp(int(g)), _clamp(int(b))


def _clamp(value: int, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, value))
