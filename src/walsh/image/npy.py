"""NumPy ``.npy`` arrays as images: the raw pixel matrix, with no picture format.

``.npy`` is NumPy's own serialisation: a magic string, a one-line header giving
shape, dtype and memory order, then the array's bytes. It is the natural way to
hand this codec an image that already lives in an array -- from Pillow, OpenCV,
a camera, or a numerical pipeline -- without converting to a picture format
first, and it needs no dependency this package does not already have.

An array carries no colour metadata, so the profile has to be declared rather
than detected. This reader accepts ``uint8`` arrays of shape ``(height, width,
3)`` as RGB; ``(height, width)`` and ``(height, width, 1)`` as greyscale,
replicated across the three channels; and ``(height, width, 4)`` as RGBA only
when every alpha value is 255, in which case the fourth channel is dropped.
Everything else is rejected by name. Other dtypes, because a float array could
be scaled 0-1 or 0-255 and guessing is how silent corruption starts. Other
channel counts, and a fourth channel that is not fully opaque, because CMYK is
a colour-space conversion that is undefined without a profile rather than a
permutation, and transparency can only be flattened by inventing a background.
Channel order is RGB by definition: a BGR array, as OpenCV produces, is one
expression away, ``array[..., ::-1]``.

The header is validated before a byte of the body is read, and pickled object
arrays are refused from the header alone and never loaded. The writer always
produces ``(height, width, 3)`` ``uint8`` in C order, so a file written here
round-trips through ``numpy.load`` unchanged.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import BinaryIO, Literal

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write, read_up_to
from walsh.image.base import RasterImage

__all__ = ["NPY_CHANNELS", "NPY_DTYPE", "NPYImage"]

log = logging.getLogger(__name__)

#: The one sample type accepted and the one written.
NPY_DTYPE = np.dtype(np.uint8)

#: Channels in the one shape the writer produces.
NPY_CHANNELS = 3

#: Channel counts the reader accepts, and how each is brought to RGB.
_GREY = 1
_RGBA = 4
_OPAQUE = 255

#: Header versions this reader parses. NumPy writes 1.0 for any plain array
#: and 2.0 only for a header over 64 KiB; 3.0 exists for non-latin-1 field
#: names in structured dtypes, which cannot describe an image.
_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


class NPYImage(RasterImage):
    """An image stored as a NumPy array in ``.npy`` form.

    Rows come first, as in every array-based image library, so the shape is
    ``(height, width, channels)`` and the raster contract's top-row-first
    order needs no conversion.
    """

    @staticmethod
    def _read_header(file: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[np.generic]]:
        """Parse the magic string and the array header, and nothing after them.

        Args:
            file: Stream positioned at the start of the file.

        Returns:
            The declared shape, whether the body is Fortran-ordered, and the
            dtype.

        Raises:
            UnsupportedFileFormatError: If the file does not start with the
                ``.npy`` magic, its header is malformed or truncated, or its
                format version is one NumPy uses only for structured dtypes.
        """
        try:
            version = np.lib.format.read_magic(file)
        except ValueError as error:
            raise UnsupportedFileFormatError(f"not a .npy file: {error}") from None
        reader = _HEADER_READERS.get(version)
        if reader is None:
            raise UnsupportedFileFormatError(
                f"unsupported .npy format version {version[0]}.{version[1]}: expected 1.0 or 2.0"
            )
        try:
            shape, fortran_order, dtype = reader(file)
        except ValueError as error:
            raise UnsupportedFileFormatError(f"invalid .npy header: {error}") from None
        return tuple(int(n) for n in shape), bool(fortran_order), dtype

    @staticmethod
    def _validate(shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> None:
        """Reject, by name, anything but an 8-bit image-shaped array.

        Runs on the header alone, before the body is read, so a file whose
        header declares a huge or hostile array costs nothing to refuse.

        Args:
            shape: The declared shape.
            dtype: The declared dtype.

        Raises:
            UnsupportedFileFormatError: If the dtype is not ``uint8`` --
                including ``object``, which is how a pickled array presents
                and which is never loaded -- or the shape is not
                ``(height, width)`` or ``(height, width, channels)`` with
                positive dimensions and 1, 3 or 4 channels.
        """
        if dtype != NPY_DTYPE:
            raise UnsupportedFileFormatError(
                f"unsupported .npy dtype {dtype}: expected uint8 samples in 0-255"
            )
        if len(shape) not in (2, 3):
            raise UnsupportedFileFormatError(
                f"unsupported .npy shape {shape}: expected (height, width) or "
                f"(height, width, channels)"
            )
        height, width = shape[0], shape[1]
        if height < 1 or width < 1:
            raise UnsupportedFileFormatError(
                f"invalid .npy shape {shape}: dimensions must be positive"
            )
        channels = shape[2] if len(shape) == 3 else _GREY
        if channels not in (_GREY, NPY_CHANNELS, _RGBA):
            raise UnsupportedFileFormatError(
                f"unsupported .npy shape {shape}: {channels} channels; expected 1 "
                f"(greyscale), 3 (RGB) or 4 (RGBA)"
            )

    @staticmethod
    def _to_rgb(array: np.ndarray) -> np.ndarray:
        """Bring a validated array to ``(height, width, 3)``.

        Args:
            array: A ``uint8`` array of a shape :meth:`_validate` accepted.

        Returns:
            The RGB array, C-contiguous.

        Raises:
            UnsupportedFileFormatError: If a four-channel array is not fully
                opaque. That is either transparency, which cannot be dropped
                without inventing a background, or CMYK, which the shape alone
                cannot tell apart from RGBA and which is not a permutation of
                RGB.
        """
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        if array.shape[2] == _GREY:
            array = np.repeat(array, NPY_CHANNELS, axis=2)
        elif array.shape[2] == _RGBA:
            if int(array[:, :, 3].min()) < _OPAQUE:
                raise UnsupportedFileFormatError(
                    "unsupported .npy pixels: the fourth channel is not fully opaque; "
                    "RGBA with transparency and CMYK are not supported, only RGBA "
                    "whose alpha is 255 throughout"
                )
            array = array[:, :, :NPY_CHANNELS]
        return np.ascontiguousarray(array)

    def load(self, filename: FileSource) -> None:
        """Read a ``.npy`` from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``.
                Reads forward only, so a pipe works.

        Raises:
            UnsupportedFileFormatError: If the file is not a ``.npy``, its
                header describes anything but an 8-bit image-shaped array, its
                body is shorter than the header declares, or a four-channel
                array is not fully opaque.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            shape, fortran_order, dtype = self._read_header(file)
            self._validate(shape, dtype)
            expected = math.prod(shape) * NPY_DTYPE.itemsize
            # Sized from a header field, so never asked for in one call.
            body = read_up_to(file, expected)
            if len(body) < expected:
                raise UnsupportedFileFormatError(
                    f"truncated .npy data: expected {expected} bytes, got {len(body)}"
                )

        order: Literal["C", "F"] = "F" if fortran_order else "C"
        array = self._to_rgb(np.frombuffer(body, dtype=NPY_DTYPE).reshape(shape, order=order))
        self._height, self._width = array.shape[0], array.shape[1]
        r, g, b = array.reshape(-1, NPY_CHANNELS).T.tolist()
        self._raw_data = list(zip(r, g, b, strict=True))
        log.debug("loaded .npy %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename`` as a ``(height, width, 3)`` ``uint8`` array.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If the pixel count does not match the dimensions.
            OSError: If the file cannot be written.
        """
        self._check_complete()
        samples = itertools.chain.from_iterable(self._raw_data)
        array = np.fromiter(samples, dtype=NPY_DTYPE, count=len(self._raw_data) * NPY_CHANNELS)
        with open_binary_write(filename) as file:
            np.save(
                file, array.reshape(self._height, self._width, NPY_CHANNELS), allow_pickle=False
            )
