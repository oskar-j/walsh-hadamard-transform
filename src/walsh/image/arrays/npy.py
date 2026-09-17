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

The header is validated before a byte of the body is read. An ``object``
array, which NumPy stores as a pickle inside the ``.npy`` and only
``numpy.load(allow_pickle=True)`` will open, is read since 0.4.15 through the
allowlisted unpickler of :mod:`walsh.image.arrays.pkl`, so nothing in it is executed
and ``numpy.load`` is still never called with pickling enabled; what it holds
is then judged as a pickle of pixels would be. The writer always produces
``(height, width, 3)`` ``uint8`` in C order, so a file written here round-trips
through ``numpy.load`` unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import BinaryIO, Literal

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write, read_up_to
from walsh.image.arrays._rules import IMAGE_DTYPE, RGB_CHANNELS, to_rgb, validate_image_array
from walsh.image.arrays.pkl import pixels_from_object, safe_loads
from walsh.image.base import RasterImage

__all__ = ["NPY_CHANNELS", "NPY_DTYPE", "NPYImage"]

log = logging.getLogger(__name__)

#: The one sample type accepted and the one written.
NPY_DTYPE = IMAGE_DTYPE

#: Channels in the one shape the writer produces.
NPY_CHANNELS = RGB_CHANNELS

#: What this format is called in a message.
_LABEL = ".npy"
_OBJECT_LABEL = ".npy object array"

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
        header declares a huge or hostile array costs nothing to refuse. The
        rules are :func:`~walsh.image.arrays._rules.validate_image_array`'s, shared
        with the pickle reader.

        Args:
            shape: The declared shape.
            dtype: The declared dtype.

        Raises:
            UnsupportedFileFormatError: If the dtype is not ``uint8`` or the
                shape is not ``(height, width)`` or ``(height, width,
                channels)`` with positive dimensions and 1, 3 or 4 channels.
        """
        validate_image_array(shape, dtype, _LABEL)

    @staticmethod
    def _to_rgb(array: np.ndarray) -> np.ndarray:
        """Bring a validated array to ``(height, width, 3)``.

        Args:
            array: A ``uint8`` array of a shape :meth:`_validate` accepted.

        Returns:
            The RGB array, C-contiguous.

        Raises:
            UnsupportedFileFormatError: If a four-channel array is not fully
                opaque; see :func:`~walsh.image.arrays._rules.to_rgb`.
        """
        return to_rgb(array, _LABEL)

    def _load_object_array(self, file: BinaryIO, shape: tuple[int, ...]) -> None:
        """Read the pickled body of an ``object`` array, executing nothing.

        Args:
            file: Stream positioned just after the header.
            shape: The shape the header declared.

        Raises:
            UnsupportedFileFormatError: If the body is not a pickle the
                allowlisted unpickler reads, does not hold an array of the
                declared shape, or that array does not hold an image.
        """
        # A pickle's length is not declared anywhere, so this is the rest of
        # the file, whose size is its own.
        loaded = safe_loads(file.read(), _OBJECT_LABEL)
        if not isinstance(loaded, np.ndarray) or loaded.shape != shape:
            raise UnsupportedFileFormatError(
                f"invalid {_OBJECT_LABEL}: the body does not hold an array of the "
                f"declared shape {shape}"
            )
        self.set_array(pixels_from_object(loaded, self._declared_size, _OBJECT_LABEL))

    def load(self, filename: FileSource) -> None:
        """Read a ``.npy`` from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``.
                Reads forward only, so a pipe works.

        Raises:
            UnsupportedFileFormatError: If the file is not a ``.npy``, its
                header describes anything but an 8-bit image-shaped array or
                an ``object`` array holding pixels, its body is shorter than
                the header declares, or a four-channel array is not fully
                opaque.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            shape, fortran_order, dtype = self._read_header(file)
            if dtype.hasobject:
                self._load_object_array(file, shape)
                log.debug("loaded .npy object array from %s", filename)
                return
            self._validate(shape, dtype)
            expected = math.prod(shape) * NPY_DTYPE.itemsize
            # Sized from a header field, so never asked for in one call.
            body = read_up_to(file, expected)
            if len(body) < expected:
                raise UnsupportedFileFormatError(
                    f"truncated .npy data: expected {expected} bytes, got {len(body)}"
                )

        order: Literal["C", "F"] = "F" if fortran_order else "C"
        self.set_array(
            self._to_rgb(np.frombuffer(body, dtype=NPY_DTYPE).reshape(shape, order=order))
        )
        log.debug("loaded .npy %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename`` as a ``(height, width, 3)`` ``uint8`` array.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If the pixel count does not match the dimensions.
            OSError: If the file cannot be written.
        """
        array = self.get_array()
        with open_binary_write(filename) as file:
            np.save(file, array, allow_pickle=False)
