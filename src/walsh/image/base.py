"""The in-memory contract every raster format in this package shares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from walsh.image._io import FileSource

__all__ = ["Pixel", "RasterImage"]

#: One pixel as ``(red, green, blue)``, each channel a 0-255 integer.
Pixel = tuple[int, int, int]


class RasterImage(ABC):
    """A rectangular image held as a flat list of pixels.

    Subclasses handle one file format each. They all present the same in-memory
    view, so a picture read from one format can be written to another:

    * pixels are ``(red, green, blue)`` triples, **never** the file's own
      channel order;
    * rows run **top to bottom**, and within a row, left to right;
    * ``get_raw_data()`` returns ``width * height`` pixels.

    A format whose on-disk layout differs, as BMP's does on both counts, is
    responsible for converting in :meth:`load` and :meth:`save`.
    """

    def __init__(self) -> None:
        """Create an empty image with zero dimensions and no pixels."""
        self._width = 0
        self._height = 0
        self._raw_data: list[Pixel] = []

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

        The class contract says ``get_raw_data()`` holds ``width * height``
        pixels, but the two-call build -- :meth:`set_dimensions` then
        :meth:`set_raw_data` -- is transiently inconsistent by design, so this
        cannot live in either setter. Every :meth:`save` calls it first
        instead, and before writing any header: too few pixels used to produce
        a file that no reader in this package can load, and too many used to
        drop the surplus with no error at all.

        Raises:
            ValueError: If the number of pixels held is not exactly
                ``width * height``.
        """
        expected = self._width * self._height
        if len(self._raw_data) != expected:
            raise ValueError(
                f"image holds {len(self._raw_data)} pixel(s) but its dimensions "
                f"{self._width}x{self._height} require {expected}"
            )

    def get_raw_data(self) -> list[Pixel]:
        """Return the pixels, as RGB triples, top row first.

        Returns:
            The live internal list of ``width * height`` pixels. Mutating it
            mutates the image.
        """
        return self._raw_data

    def set_raw_data(self, new_data: Sequence[Pixel]) -> None:
        """Replace the pixels.

        Args:
            new_data: ``width * height`` RGB triples, top row first. Copied
                into the image, so the caller may reuse the sequence.
        """
        self._raw_data = list(new_data)
