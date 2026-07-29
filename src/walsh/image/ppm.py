"""Portable Pixmap (PPM) images, from the Netpbm suite.

PPM is the closest Unix equivalent to the uncompressed BMP this codec started
with: a tiny ASCII header followed by raw samples, no compression, no palette.

Two variants exist and both are read:

* **P6**, binary, one byte per sample. This is what :meth:`PPMImage.save`
  writes.
* **P3**, the same data as whitespace-separated decimal numbers. Read only,
  because it is several times larger for no benefit.

The header may carry ``#`` comments between any two tokens, which the parser
skips. Files whose ``maxval`` is below 255 are rescaled to the full 0-255 range
on load; ``maxval`` above 255 means 16-bit samples, which are not supported.
"""

from __future__ import annotations

import logging
from typing import BinaryIO

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write
from walsh.image.base import RasterImage

__all__ = ["PPM_ASCII_MAGIC", "PPM_BINARY_MAGIC", "PPM_MAX_SAMPLE", "PPMImage"]

log = logging.getLogger(__name__)

PPM_BINARY_MAGIC = b"P6"
PPM_ASCII_MAGIC = b"P3"

#: The largest ``maxval`` this reader handles, i.e. one byte per sample.
PPM_MAX_SAMPLE = 255

_WHITESPACE = b" \t\r\n\v\f"
_COMMENT = b"#"


class PPMImage(RasterImage):
    """A Portable Pixmap image.

    Pixels are stored red-green-blue and top-down on disk, which is already
    this package's in-memory contract, so no conversion is needed either way.
    """

    def __init__(self) -> None:
        """Create an empty pixmap."""
        super().__init__()
        self._magic = PPM_BINARY_MAGIC

    @staticmethod
    def _read_token(file: BinaryIO) -> bytes:
        """Read one whitespace-delimited header token, skipping comments.

        Args:
            file: Stream positioned anywhere in the header.

        Returns:
            The token, without its trailing whitespace byte.

        Raises:
            UnsupportedFileFormatError: If the stream ends before a complete
                token is read.
        """
        token = bytearray()
        while True:
            char = file.read(1)
            if not char:
                if token:
                    return bytes(token)
                raise UnsupportedFileFormatError("truncated PPM header: expected another value")
            if char == _COMMENT:
                while char and char not in b"\r\n":
                    char = file.read(1)
                continue
            if char in _WHITESPACE:
                if token:
                    return bytes(token)
                continue
            token += char

    def _read_int(self, file: BinaryIO, what: str) -> int:
        """Read one header token and interpret it as a positive integer.

        Args:
            file: Stream positioned in the header.
            what: Field name, used only in the error message.

        Returns:
            The parsed value.

        Raises:
            UnsupportedFileFormatError: If the token is not a positive integer.
        """
        token = self._read_token(file)
        try:
            value = int(token)
        except ValueError:
            raise UnsupportedFileFormatError(
                f"invalid PPM {what}: {token!r} is not a number"
            ) from None
        if value <= 0:
            raise UnsupportedFileFormatError(f"invalid PPM {what}: {value}")
        return value

    def _read_header(self, file: BinaryIO) -> int:
        """Parse the magic number, dimensions and maximum sample value.

        Args:
            file: Stream positioned at the start of the file.

        Returns:
            The declared ``maxval``.

        Raises:
            UnsupportedFileFormatError: If the magic is not ``P3``/``P6``, a
                field is malformed, or ``maxval`` exceeds 255.
        """
        self._magic = self._read_token(file)
        if self._magic not in (PPM_BINARY_MAGIC, PPM_ASCII_MAGIC):
            raise UnsupportedFileFormatError(
                f"expected a PPM starting P3 or P6, got {self._magic!r}"
            )

        self._width = self._read_int(file, "width")
        self._height = self._read_int(file, "height")
        maxval = self._read_int(file, "maxval")

        if maxval > PPM_MAX_SAMPLE:
            raise UnsupportedFileFormatError(
                f"16-bit PPM samples are not supported: maxval is {maxval}, "
                f"expected at most {PPM_MAX_SAMPLE}"
            )
        return maxval

    @staticmethod
    def _rescale(value: int, maxval: int) -> int:
        """Scale one sample from a 0..maxval range to 0..255.

        Args:
            value: The raw sample.
            maxval: The file's declared maximum, in 1..255.

        Returns:
            The sample expressed against a maximum of 255, rounded to nearest.
        """
        if maxval == PPM_MAX_SAMPLE:
            return value
        return (value * PPM_MAX_SAMPLE + maxval // 2) // maxval

    def _read_binary_data(self, file: BinaryIO, maxval: int) -> None:
        """Read P6 pixel data, one byte per sample.

        Args:
            file: Stream positioned at the first sample.
            maxval: The declared maximum sample value.

        Raises:
            UnsupportedFileFormatError: If the data is shorter than the header
                promises.
        """
        expected = self._width * self._height * 3
        data = file.read(expected)
        if len(data) < expected:
            raise UnsupportedFileFormatError(
                f"truncated PPM data: expected {expected} bytes, got {len(data)}"
            )
        self._raw_data = [
            (
                self._rescale(data[i], maxval),
                self._rescale(data[i + 1], maxval),
                self._rescale(data[i + 2], maxval),
            )
            for i in range(0, expected, 3)
        ]

    def _read_ascii_data(self, file: BinaryIO, maxval: int) -> None:
        """Read P3 pixel data, as whitespace-separated decimal numbers.

        Args:
            file: Stream positioned at the first sample.
            maxval: The declared maximum sample value.

        Raises:
            UnsupportedFileFormatError: If a sample is not a number, or there
                are fewer samples than the header promises.
        """
        expected = self._width * self._height * 3
        samples: list[int] = []
        while len(samples) < expected:
            token = file.read(1)
            if not token:
                break
            if token in _WHITESPACE:
                continue
            file.seek(-1, 1)
            raw = self._read_token(file)
            try:
                samples.append(self._rescale(int(raw), maxval))
            except ValueError:
                raise UnsupportedFileFormatError(
                    f"invalid PPM sample: {raw!r} is not a number"
                ) from None

        if len(samples) < expected:
            raise UnsupportedFileFormatError(
                f"truncated PPM data: expected {expected} samples, got {len(samples)}"
            )
        self._raw_data = [
            (samples[i], samples[i + 1], samples[i + 2]) for i in range(0, expected, 3)
        ]

    def load(self, filename: FileSource) -> None:
        """Read a PPM from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Raises:
            UnsupportedFileFormatError: If the data is not a supported PPM.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            maxval = self._read_header(file)
            if self._magic == PPM_BINARY_MAGIC:
                self._read_binary_data(file, maxval)
            else:
                self._read_ascii_data(file, maxval)
        log.debug("loaded PPM %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename`` as a binary P6 pixmap.

        P6 is always used, even if the image was read from a P3 file, since the
        two carry identical information and P6 is far more compact.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            OSError: If the file cannot be written.
        """
        header = b"%s\n%d %d\n%d\n" % (
            PPM_BINARY_MAGIC,
            self._width,
            self._height,
            PPM_MAX_SAMPLE,
        )
        body = bytes(channel for pixel in self._raw_data for channel in pixel)
        with open_binary_write(filename) as file:
            file.write(header)
            file.write(body)
