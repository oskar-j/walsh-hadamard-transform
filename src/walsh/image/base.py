"""The in-memory contract every raster format in this package shares."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from walsh.image._io import FileSource

__all__ = ["CHANNELS", "Pixel", "PixelArray", "RasterImage"]

#: One pixel as ``(red, green, blue)``, each channel a 0-255 integer.
Pixel = tuple[int, int, int]

#: An image as an array: ``uint8``, shape ``(height, width, 3)``, RGB.
PixelArray = npt.NDArray[np.uint8]

#: Samples per pixel in the in-memory contract.
CHANNELS = 3


class RasterImage(ABC):
    """A rectangular image held as a flat list of pixels.

    Subclasses handle one file format each. They all present the same in-memory
    view, so a picture read from one format can be written to another:

    * pixels are ``(red, green, blue)`` triples, **never** the file's own
      channel order;
    * rows run **top to bottom**, and within a row, left to right;
    * there are exactly ``width * height`` of them.

    A format whose on-disk layout differs, as BMP's does on both counts, is
    responsible for converting in :meth:`load` and :meth:`save`.

    The pixels are held as one ``uint8`` array, and :meth:`get_array` /
    :meth:`set_array` are the primary accessors (0.4.10). Up to 0.4.9 the
    store was a Python list of tuples, and marshalling pixels through it was
    about half of the codec's wall time and some 70 bytes per pixel of
    memory. :meth:`get_raw_data` / :meth:`set_raw_data` remain as converters
    for callers that still want the list.
    """

    def __init__(self) -> None:
        """Create an empty image with zero dimensions and no pixels."""
        self._width = 0
        self._height = 0
        self._pixels: PixelArray = np.empty((0, CHANNELS), dtype=np.uint8)
        self._declared_size: tuple[int, int] | None = None

    @abstractmethod
    def load(self, filename: FileSource) -> None:
        """Read an image from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``.
                Whether a non-seekable stdin works depends on the format; each
                subclass says.

        Raises:
            UnsupportedFileFormatError: If the data is not in this subclass's
                format, or uses a variant of it that is not supported.
            OSError: If the file cannot be read.
        """

    @abstractmethod
    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename``.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            OSError: If the file cannot be written.
        """

    def declare_size(self, width: int, height: int) -> None:
        """Tell the image, before :meth:`load`, how large the picture is.

        Almost every format carries its own size and its reader ignores this.
        It exists for input that does not: a pickled flat list of pixels is
        ``width * height`` tuples with nothing to say which is which. It does
        not set the dimensions; ``load`` does, and :class:`~walsh.task.Task`
        checks the result against the declaration for every format, so a
        declared size is honoured or verified, never silently dropped.

        Args:
            width: Declared width in pixels.
            height: Declared height in pixels.

        Raises:
            ValueError: If either is not a positive integer.
        """
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"declared {name} must be a positive integer, got {value!r}")
        self._declared_size = (width, height)

    def get_dimensions(self) -> tuple[int, int]:
        """Return the image size.

        Returns:
            The ``(width, height)`` pair, in pixels.
        """
        return self._width, self._height

    def set_dimensions(self, width: int, height: int) -> None:
        """Set the image size.

        This does not resize any pixel data already held; callers are expected
        to follow it with :meth:`set_raw_data`.

        Args:
            width: New width in pixels.
            height: New height in pixels.
        """
        self._width = width
        self._height = height

    def _check_complete(self) -> None:
        """Verify the pixel count matches the dimensions, before writing.

        The class contract is ``width * height`` pixels, but the two-call
        build -- :meth:`set_dimensions` then :meth:`set_raw_data` -- is
        transiently inconsistent by design, so this cannot live in either
        setter. Every :meth:`save` calls it first instead, and before writing
        any header: too few pixels used to produce a file that no reader in
        this package can load, and too many used to drop the surplus with no
        error at all. :meth:`get_array` calls it too, since it cannot shape
        the array otherwise.

        Raises:
            ValueError: If the number of pixels held is not exactly
                ``width * height``.
        """
        expected = self._width * self._height
        if len(self._pixels) != expected:
            raise ValueError(
                f"image holds {len(self._pixels)} pixel(s) but its dimensions "
                f"{self._width}x{self._height} require {expected}"
            )

    def get_array(self) -> PixelArray:
        """Return the pixels as a ``(height, width, 3)`` ``uint8`` array.

        This is the primary accessor and the one the codec uses: no per-pixel
        Python object is created on the way in or out.

        Returns:
            A view of the image's own storage, so mutating it mutates the
            image. It is C-contiguous, so ``reshape(-1, 3)`` and ``tobytes()``
            are free.

        Raises:
            ValueError: If the pixel count does not match the dimensions,
                which can only happen midway through a
                :meth:`set_dimensions` / :meth:`set_raw_data` build.
        """
        self._check_complete()
        return self._pixels.reshape(self._height, self._width, CHANNELS)

    def set_array(self, array: npt.ArrayLike) -> None:
        """Replace the pixels and the dimensions from one array.

        Args:
            array: ``uint8`` of shape ``(height, width, 3)``, RGB, top row
                first. The image keeps a reference rather than a copy when
                the array is already contiguous ``uint8``, so copy first if
                you will go on mutating it. A read-only array (for example one
                made by ``np.frombuffer``) is copied, so :meth:`get_array`
                always hands back something writable.

        Raises:
            ValueError: If the array is not three-dimensional with three
                channels, or its dtype is not ``uint8``. The dtype is checked
                rather than cast because a silent cast is how a float or a
                value above 255 would become a wrong pixel with no error.
        """
        pixels = np.asarray(array)
        if pixels.dtype != np.uint8:
            raise ValueError(f"pixels must be uint8, got {pixels.dtype}")
        if pixels.ndim != 3 or pixels.shape[2] != CHANNELS:
            raise ValueError(
                f"pixels must be shaped (height, width, {CHANNELS}), got {pixels.shape}"
            )
        pixels = np.ascontiguousarray(pixels)
        if not pixels.flags.writeable:
            pixels = pixels.copy()
        # Through the method, not the attributes: a subclass may derive header
        # fields from the dimensions there, as BMP does for its size fields.
        self.set_dimensions(pixels.shape[1], pixels.shape[0])
        self._pixels = pixels.reshape(-1, CHANNELS)

    def get_raw_data(self) -> list[Pixel]:
        """Return the pixels as a list of RGB triples, top row first.

        A converter kept for callers that predate :meth:`get_array`. It builds
        a fresh list on every call, so mutating the result does **not**
        mutate the image -- that was true of the list store up to 0.4.9 and
        is not any more. Use :meth:`get_array` for the live pixels.

        Returns:
            ``width * height`` tuples of three ints.
        """
        r, g, b = self._pixels.T.tolist()
        return list(zip(r, g, b, strict=True))

    def set_raw_data(self, new_data: Sequence[Pixel]) -> None:
        """Replace the pixels from a list of RGB triples.

        A converter kept for callers that predate :meth:`set_array`. It does
        not touch the dimensions, so the documented two-call build --
        :meth:`set_dimensions` then this -- still works; :meth:`save` checks
        that the two agree.

        Args:
            new_data: ``width * height`` RGB triples, top row first, each
                channel an int in 0-255. Copied, so the caller may reuse the
                sequence.
        """
        samples = itertools.chain.from_iterable(new_data)
        self._pixels = np.fromiter(samples, dtype=np.uint8, count=len(new_data) * CHANNELS).reshape(
            -1, CHANNELS
        )
