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

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, align, open_binary_read, open_binary_write, read_up_to
from walsh.image.base import RasterImage

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

        header_end = 14 + self._header_size
        if self._offset < header_end:
            raise UnsupportedFileFormatError(
                f"BMP pixel offset {self._offset} is before the header end at byte "
                f"{header_end} (info header size {self._header_size})"
            )

        # A negative height means the rows are already stored top-down.
        self._top_down = height < 0
        self._height = abs(height)
        if self._width <= 0 or self._height <= 0:
            # Zero is the interesting case, not the negative one: a zero width
            # makes the row stride zero, so the truncation guard in _read_data
            # compares 0 < 0 and can never fire, and the reader loops over the
            # declared height against a file with no pixel data at all. PPM and
            # PAM already reject a dimension of zero; this brings BMP in line.
            raise UnsupportedFileFormatError(
                f"BMP dimensions must be positive, got {self._width}x{self._height}"
            )

    def _read_data(self, file: BinaryIO) -> None:
        """Read the pixel array into the top-down RGB contract.

        Args:
            file: Stream from which to read; seeks to the pixel offset first.

        Raises:
            UnsupportedFileFormatError: If a row is truncated.
        """
        file.seek(self._offset)
        row_bytes = self._width * 3
        stride = align(row_bytes, BMP_ROW_ALIGNMENT)

        # Sized from header fields, so never asked for in one call. The last
        # row's padding may be missing, as before; only its pixels must be.
        data = read_up_to(file, stride * self._height)
        needed = stride * (self._height - 1) + row_bytes
        if len(data) < needed:
            index = min(len(data) // stride, self._height - 1)
            present = min(len(data) - index * stride, stride)
            raise UnsupportedFileFormatError(
                f"truncated BMP pixel data: row {index} of {self._height} "
                f"has {present} bytes, expected at least {row_bytes}"
            )
        data = data.ljust(stride * self._height, b"\x00")

        # One reshape drops the row padding; BMP stores blue, green, red and,
        # unless the height was negative, bottom row first.
        rows = np.frombuffer(data, dtype=np.uint8).reshape(self._height, stride)
        pixels = rows[:, :row_bytes].reshape(self._height, self._width, 3)[:, :, ::-1]
        if not self._top_down:
            pixels = pixels[::-1]
        self.set_array(pixels)

    def load(self, filename: FileSource) -> None:
        """Read a BMP from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``,
                which must be seekable: this reader seeks while parsing, so a
                pipe raises ``io.UnsupportedOperation``.

        Raises:
            UnsupportedFileFormatError: If the data is not a supported BMP.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
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
        log.debug("row padding: %d byte(s)", padding)

        # Red-green-blue top-down in, blue-green-red bottom-up out, each row
        # padded to a multiple of four bytes: three views and one pad.
        bgr = self.get_array()[::-1, :, ::-1].reshape(self._height, self._width * 3)
        file.write(np.pad(bgr, ((0, 0), (0, padding))).tobytes())

    def save(self, filename: FileSource) -> None:
        """Write this bitmap to ``filename``.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If the pixel count does not match the dimensions.
            OSError: If the file cannot be written.
        """
        self._check_complete()
        with open_binary_write(filename) as file:
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
