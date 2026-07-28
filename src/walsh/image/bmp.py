"""24-bit uncompressed Windows bitmaps.

BMP disagrees with this package's in-memory contract on two counts, and
:class:`BMPImage` reconciles both:

* pixels are stored blue-green-red, not red-green-blue;
* rows are stored bottom-up when the header height is positive. A negative
  height means top-down, which is why the header is parsed with a signed field.
"""

from __future__ import annotations

import logging
import struct
from typing import BinaryIO

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, align, open_binary
from walsh.image.base import Pixel, RasterImage

__all__ = [
    "BMP_HEADER_FORMAT",
    "BMP_HEADER_SIZE",
    "BMP_PIXEL_OFFSET",
    "BMP_SIGNATURE",
    "BMPImage",
]

log = logging.getLogger(__name__)

#: Width and height are signed: a negative height marks a top-down bitmap.
BMP_HEADER_FORMAT = "<2sIHHIIiiHHIIIIII"
BMP_SIGNATURE = b"BM"
BMP_HEADER_SIZE = 40
BMP_PIXEL_OFFSET = 54

#: Rows are padded out to a multiple of this many bytes.
BMP_ROW_ALIGNMENT = 4


class BMPImage(RasterImage):
    """A 24-bit uncompressed Windows bitmap.

    Only the 40-byte ``BITMAPINFOHEADER`` variant with 24 bits per pixel, one
    plane and no compression is supported. Anything else is rejected rather
    than guessed at.
    """

    def __init__(self) -> None:
        """Create an empty bitmap with default header fields."""
        super().__init__()
        self._signature = BMP_SIGNATURE
        self._size = 0
        self._offset = BMP_PIXEL_OFFSET
        self._header_size = BMP_HEADER_SIZE
        self._planes = 1
        self._bpp = 24
        self._compression = 0
        self._size_of_data = 0
        self._horizontal_res = 2835
        self._vertical_res = 2835
        self._top_down = False

    def _read_header(self, file: BinaryIO) -> None:
        """Parse the file and info headers, and validate what they describe.

        Args:
            file: Stream positioned at the start of the file.

        Raises:
            UnsupportedFileFormatError: If the header is truncated, or does not
                describe a 24-bit single-plane uncompressed bitmap.
        """
        raw = file.read(struct.calcsize(BMP_HEADER_FORMAT))
        if len(raw) < struct.calcsize(BMP_HEADER_FORMAT):
            raise UnsupportedFileFormatError(
                f"truncated BMP header: expected {struct.calcsize(BMP_HEADER_FORMAT)} "
                f"bytes, got {len(raw)}"
            )

        (
            self._signature,
            self._size,
            _,
            _,
            self._offset,
            self._header_size,
            self._width,
            height,
            self._planes,
            self._bpp,
            self._compression,
            self._size_of_data,
            self._horizontal_res,
            self._vertical_res,
            _,
            _,
        ) = struct.unpack(BMP_HEADER_FORMAT, raw)

        actual = (self._signature, self._planes, self._bpp, self._compression)
        if actual != (BMP_SIGNATURE, 1, 24, 0):
            raise UnsupportedFileFormatError(
                "expected a 24-bit single-plane uncompressed BMP, got "
                f"signature={self._signature!r} planes={self._planes} "
                f"bpp={self._bpp} compression={self._compression}"
            )

        # A negative height means the rows are already stored top-down.
        self._top_down = height < 0
        self._height = abs(height)
        if self._width < 0:
            raise UnsupportedFileFormatError(f"negative BMP width: {self._width}")

    def _read_data(self, file: BinaryIO) -> None:
        """Read the pixel array into the top-down RGB contract.

        Args:
            file: Stream from which to read; seeks to the pixel offset first.

        Raises:
            UnsupportedFileFormatError: If a row is truncated.
        """
        file.seek(self._offset)
        stride = align(self._width * 3, BMP_ROW_ALIGNMENT)

        rows: list[list[Pixel]] = []
        for index in range(self._height):
            line = file.read(stride)
            if len(line) < self._width * 3:
                raise UnsupportedFileFormatError(
                    f"truncated BMP pixel data: row {index} of {self._height} "
                    f"has {len(line)} bytes, expected at least {self._width * 3}"
                )
            # BMP stores blue, green, red; the contract is red, green, blue.
            rows.append([(line[i + 2], line[i + 1], line[i]) for i in range(0, self._width * 3, 3)])

        if not self._top_down:
            rows.reverse()

        self._raw_data = [pixel for row in rows for pixel in row]

    def load(self, filename: FileSource) -> None:
        """Read a BMP from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Raises:
            UnsupportedFileFormatError: If the data is not a supported BMP.
            OSError: If the file cannot be read.
        """
        with open_binary(filename, "rb") as file:
            self._read_header(file)
            self._read_data(file)
        log.debug("loaded BMP %dx%d from %s", self._width, self._height, filename)

    def _write_header(self, file: BinaryIO) -> None:
        """Write the 54-byte file and info headers.

        Rows are always written bottom-up, so the stored height is positive
        regardless of how the source file was laid out.

        Args:
            file: Stream to write to.
        """
        ignored = 0
        file.write(
            struct.pack(
                BMP_HEADER_FORMAT,
                BMP_SIGNATURE,
                self._size,
                ignored,
                ignored,
                BMP_PIXEL_OFFSET,
                BMP_HEADER_SIZE,
                self._width,
                self._height,
                1,
                24,
                0,
                self._size_of_data,
                self._horizontal_res,
                self._vertical_res,
                ignored,
                ignored,
            )
        )

    def _write_data(self, file: BinaryIO) -> None:
        """Write the pixel array, converting back to BMP's own layout.

        Args:
            file: Stream to write to.
        """
        padding = align(self._width * 3, BMP_ROW_ALIGNMENT) - self._width * 3
        pad = b"\x00" * padding
        log.debug("row padding: %d byte(s)", padding)

        for start in range(len(self._raw_data) - self._width, -1, -self._width):
            row = self._raw_data[start : start + self._width]
            file.write(b"".join(struct.pack("<BBB", b, g, r) for r, g, b in row))
            file.write(pad)

    def save(self, filename: FileSource) -> None:
        """Write this bitmap to ``filename``.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            OSError: If the file cannot be written.
        """
        with open_binary(filename, "wb") as file:
            self._write_header(file)
            self._write_data(file)

    def set_dimensions(self, width: int, height: int) -> None:
        """Set the image size and refresh the derived header byte counts.

        Args:
            width: New width in pixels.
            height: New height in pixels.
        """
        super().set_dimensions(width, height)
        self._size_of_data = align(width * 3, BMP_ROW_ALIGNMENT) * height
        self._size = BMP_PIXEL_OFFSET + self._size_of_data
