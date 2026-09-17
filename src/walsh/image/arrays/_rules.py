"""What makes a NumPy array an image, shared by every reader that meets one.

``.npy`` files and pickles both deliver arrays, and an array carries no colour
metadata, so the profile is declared here once rather than detected: ``uint8``
only; ``(height, width, 3)`` is RGB, ``(height, width)`` and ``(height, width,
1)`` are greyscale replicated to three channels, and ``(height, width, 4)`` is
RGBA accepted only when alpha is 255 throughout. See :mod:`walsh.image.arrays.npy`
for why each of the others is refused.
"""

from __future__ import annotations

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError

__all__ = ["IMAGE_DTYPE", "RGB_CHANNELS", "to_rgb", "validate_image_array"]

#: The one sample type accepted.
IMAGE_DTYPE = np.dtype(np.uint8)

#: Channels of the array every reader hands on.
RGB_CHANNELS = 3

_GREY = 1
_RGBA = 4
_OPAQUE = 255


def validate_image_array(shape: tuple[int, ...], dtype: np.dtype[np.generic], label: str) -> None:
    """Reject, by name, anything but an 8-bit image-shaped array.

    Needs only the shape and dtype, so a reader that learns those from a
    header can refuse a huge or hostile array before reading its body.

    Args:
        shape: The array's shape.
        dtype: The array's dtype.
        label: What to call the source in a message, such as ``".npy"``.

    Raises:
        UnsupportedFileFormatError: If the dtype is not ``uint8``, or the
            shape is not ``(height, width)`` or ``(height, width, channels)``
            with positive dimensions and 1, 3 or 4 channels.
    """
    if dtype != IMAGE_DTYPE:
        raise UnsupportedFileFormatError(
            f"unsupported {label} dtype {dtype}: expected uint8 samples in 0-255"
        )
    if len(shape) not in (2, 3):
        raise UnsupportedFileFormatError(
            f"unsupported {label} shape {shape}: expected (height, width) or "
            f"(height, width, channels)"
        )
    height, width = shape[0], shape[1]
    if height < 1 or width < 1:
        raise UnsupportedFileFormatError(
            f"invalid {label} shape {shape}: dimensions must be positive"
        )
    channels = shape[2] if len(shape) == 3 else _GREY
    if channels not in (_GREY, RGB_CHANNELS, _RGBA):
        raise UnsupportedFileFormatError(
            f"unsupported {label} shape {shape}: {channels} channels; expected 1 "
            f"(greyscale), 3 (RGB) or 4 (RGBA)"
        )


def to_rgb(array: np.ndarray, label: str) -> np.ndarray:
    """Bring a validated array to ``(height, width, 3)``.

    Args:
        array: A ``uint8`` array of a shape :func:`validate_image_array`
            accepted.
        label: What to call the source in a message, such as ``".npy"``.

    Returns:
        The RGB array, C-contiguous.

    Raises:
        UnsupportedFileFormatError: If a four-channel array is not fully
            opaque. That is either transparency, which cannot be dropped
            without inventing a background, or CMYK, which the shape alone
            cannot tell apart from RGBA and which is not a permutation of RGB.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.shape[2] == _GREY:
        array = np.repeat(array, RGB_CHANNELS, axis=2)
    elif array.shape[2] == _RGBA:
        if int(array[:, :, 3].min()) < _OPAQUE:
            raise UnsupportedFileFormatError(
                f"unsupported {label} pixels: the fourth channel is not fully opaque; "
                "RGBA with transparency and CMYK are not supported, only RGBA "
                "whose alpha is 255 throughout"
            )
        array = array[:, :, :RGB_CHANNELS]
    return np.ascontiguousarray(array)
