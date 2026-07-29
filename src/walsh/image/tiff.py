"""Uncompressed baseline TIFF images.

TIFF is a container rather than a single layout, so this reader deliberately
handles one profile and rejects the rest rather than guessing: uncompressed
(``Compression = 1``), RGB (``PhotometricInterpretation = 2``), 8 bits per
sample, three samples per pixel, chunky (``PlanarConfiguration = 1``) and the
default top-left orientation. That is the profile the rest of this package can
represent; anything else -- LZW, palettes, CMYK, 16-bit, tiles -- is reported
rather than silently misread.

Both byte orders are read, since the header declares its own endianness.
Writing always produces a little-endian single-strip file.
"""

from __future__ import annotations

import logging
import struct
from typing import BinaryIO

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write
from walsh.image.base import RasterImage

__all__ = ["TIFF_BIG_ENDIAN", "TIFF_LITTLE_ENDIAN", "TIFF_MAGIC", "TIFFImage"]

log = logging.getLogger(__name__)

TIFF_LITTLE_ENDIAN = b"II"
TIFF_BIG_ENDIAN = b"MM"
TIFF_MAGIC = 42

# The subset of baseline tags this profile needs.
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIP_OFFSETS = 273
TAG_ORIENTATION = 274
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIG = 284

COMPRESSION_NONE = 1
PHOTOMETRIC_RGB = 2
PLANAR_CHUNKY = 1
ORIENTATION_TOP_LEFT = 1
SAMPLES_PER_PIXEL = 3
BITS_PER_SAMPLE = 8

#: Field type code to its size in bytes. Only the types this profile uses.
TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
TYPE_BYTE = 1
TYPE_SHORT = 3
TYPE_LONG = 4

#: An IFD entry is tag, type, count, then four bytes that are either the value
#: itself or an offset to it.
ENTRY_SIZE = 12

_HUMAN_COMPRESSION = {
    1: "none",
    2: "CCITT modified Huffman",
    5: "LZW",
    6: "JPEG (old style)",
    7: "JPEG",
    8: "Deflate (Adobe)",
    32773: "PackBits",
    32946: "Deflate",
}
_HUMAN_PHOTOMETRIC = {
    0: "white is zero (bilevel or greyscale)",
    1: "black is zero (greyscale)",
    2: "RGB",
    3: "palette colour",
    4: "transparency mask",
    5: "CMYK",
    6: "YCbCr",
}


class TIFFImage(RasterImage):
    """An uncompressed baseline RGB TIFF.

    Pixels are stored red-green-blue and, at the default orientation, top-down,
    which already matches this package's in-memory contract. Unlike BMP, rows
    carry no padding.
    """

    def __init__(self) -> None:
        """Create an empty image that will be written little-endian."""
        super().__init__()
        self._byte_order = TIFF_LITTLE_ENDIAN
        self._prefix = "<"

    def _read_header(self, file: BinaryIO) -> int:
        """Read the 8-byte header and learn the file's byte order.

        Args:
            file: Stream positioned at the start of the file.

        Returns:
            The offset of the first image file directory.

        Raises:
            UnsupportedFileFormatError: If the header is truncated, the byte
                order marker is not ``II``/``MM``, or the magic is not 42.
        """
        raw = file.read(8)
        if len(raw) < 8:
            raise UnsupportedFileFormatError(
                f"truncated TIFF header: expected 8 bytes, got {len(raw)}"
            )

        self._byte_order = raw[:2]
        if self._byte_order == TIFF_LITTLE_ENDIAN:
            self._prefix = "<"
        elif self._byte_order == TIFF_BIG_ENDIAN:
            self._prefix = ">"
        else:
            raise UnsupportedFileFormatError(
                f"expected a TIFF byte order marker of II or MM, got {self._byte_order!r}"
            )

        magic, ifd_offset = struct.unpack(f"{self._prefix}HI", raw[2:8])
        if magic != TIFF_MAGIC:
            raise UnsupportedFileFormatError(f"expected TIFF magic {TIFF_MAGIC}, got {magic}")
        return int(ifd_offset)

    def _read_values(
        self, file: BinaryIO, field_type: int, count: int, raw: bytes
    ) -> tuple[int, ...]:
        """Decode one IFD entry's value, following the offset when it has one.

        Values of four bytes or fewer live in the entry itself, left-justified;
        anything larger is stored elsewhere and the entry holds its offset.

        Args:
            file: Stream to read from when the value is out of line.
            field_type: The TIFF type code.
            count: How many values the entry holds.
            raw: The entry's four-byte value/offset field.

        Returns:
            The decoded values.

        Raises:
            UnsupportedFileFormatError: If the type is unknown or the values
                lie beyond the end of the file.
        """
        if field_type not in TYPE_SIZES:
            raise UnsupportedFileFormatError(f"unknown TIFF field type {field_type}")

        size = TYPE_SIZES[field_type] * count
        if size > 4:
            (offset,) = struct.unpack(f"{self._prefix}I", raw)
            here = file.tell()
            file.seek(offset)
            payload = file.read(size)
            file.seek(here)
            if len(payload) < size:
                raise UnsupportedFileFormatError(
                    f"TIFF field at offset {offset} runs past the end of the file"
                )
        else:
            payload = raw[:size]

        codes = {TYPE_BYTE: "B", TYPE_SHORT: "H", TYPE_LONG: "I"}
        if field_type not in codes:
            # Types this profile never needs; keep them opaque rather than lying.
            return ()
        return struct.unpack(f"{self._prefix}{count}{codes[field_type]}", payload)

    def _read_ifd(self, file: BinaryIO, offset: int) -> dict[int, tuple[int, ...]]:
        """Read one image file directory into a tag-to-values mapping.

        Args:
            file: Stream to read from.
            offset: Byte offset of the directory.

        Returns:
            Every entry in the directory, keyed by tag.

        Raises:
            UnsupportedFileFormatError: If the directory is truncated.
        """
        file.seek(offset)
        raw = file.read(2)
        if len(raw) < 2:
            raise UnsupportedFileFormatError(f"truncated TIFF directory at offset {offset}")
        (count,) = struct.unpack(f"{self._prefix}H", raw)

        entries: dict[int, tuple[int, ...]] = {}
        for index in range(count):
            entry = file.read(ENTRY_SIZE)
            if len(entry) < ENTRY_SIZE:
                raise UnsupportedFileFormatError(
                    f"truncated TIFF directory entry {index} of {count}"
                )
            tag, field_type, value_count = struct.unpack(f"{self._prefix}HHI", entry[:8])
            entries[tag] = self._read_values(file, field_type, value_count, entry[8:12])
        return entries

    @staticmethod
    def _single(entries: dict[int, tuple[int, ...]], tag: int, default: int | None = None) -> int:
        """Fetch a tag expected to hold exactly one value.

        Args:
            entries: The directory to look in.
            tag: The tag to fetch.
            default: Value to use when the tag is absent, or ``None`` to make
                the tag required.

        Returns:
            The tag's value.

        Raises:
            UnsupportedFileFormatError: If a required tag is missing or empty.
        """
        values = entries.get(tag)
        if not values:
            if default is not None:
                return default
            raise UnsupportedFileFormatError(f"TIFF is missing required tag {tag}")
        return int(values[0])

    def _validate(self, entries: dict[int, tuple[int, ...]]) -> None:
        """Reject any TIFF outside the one profile this reader supports.

        Args:
            entries: The image's directory.

        Raises:
            UnsupportedFileFormatError: If the file is compressed, not RGB, not
                8 bits per sample, planar, or not top-left oriented.
        """
        compression = self._single(entries, TAG_COMPRESSION, COMPRESSION_NONE)
        if compression != COMPRESSION_NONE:
            name = _HUMAN_COMPRESSION.get(compression, "unknown")
            raise UnsupportedFileFormatError(
                f"only uncompressed TIFF is supported, got compression {compression} ({name})"
            )

        photometric = self._single(entries, TAG_PHOTOMETRIC)
        if photometric != PHOTOMETRIC_RGB:
            name = _HUMAN_PHOTOMETRIC.get(photometric, "unknown")
            raise UnsupportedFileFormatError(
                f"only RGB TIFF is supported, got photometric interpretation {photometric} ({name})"
            )

        samples = self._single(entries, TAG_SAMPLES_PER_PIXEL, SAMPLES_PER_PIXEL)
        if samples != SAMPLES_PER_PIXEL:
            raise UnsupportedFileFormatError(
                f"expected {SAMPLES_PER_PIXEL} samples per pixel, got {samples}"
            )

        bits = entries.get(TAG_BITS_PER_SAMPLE, (BITS_PER_SAMPLE,) * SAMPLES_PER_PIXEL)
        if tuple(bits) != (BITS_PER_SAMPLE,) * SAMPLES_PER_PIXEL:
            raise UnsupportedFileFormatError(
                f"only 8 bits per sample is supported, got {tuple(bits)}"
            )

        planar = self._single(entries, TAG_PLANAR_CONFIG, PLANAR_CHUNKY)
        if planar != PLANAR_CHUNKY:
            raise UnsupportedFileFormatError(
                f"only chunky (interleaved) TIFF is supported, got planar configuration {planar}"
            )

        orientation = self._single(entries, TAG_ORIENTATION, ORIENTATION_TOP_LEFT)
        if orientation != ORIENTATION_TOP_LEFT:
            raise UnsupportedFileFormatError(
                f"only top-left orientation is supported, got {orientation}"
            )

    def _read_strips(self, file: BinaryIO, entries: dict[int, tuple[int, ...]]) -> bytes:
        """Concatenate the image's strips into one block of pixel bytes.

        Args:
            file: Stream to read from.
            entries: The image's directory.

        Returns:
            ``width * height * 3`` bytes of interleaved RGB samples.

        Raises:
            UnsupportedFileFormatError: If the strip tags disagree with each
                other, or a strip runs past the end of the file.
        """
        offsets = entries.get(TAG_STRIP_OFFSETS, ())
        counts = entries.get(TAG_STRIP_BYTE_COUNTS, ())
        if not offsets:
            raise UnsupportedFileFormatError("TIFF has no strip offsets")
        if len(offsets) != len(counts):
            raise UnsupportedFileFormatError(
                f"TIFF has {len(offsets)} strip offsets but {len(counts)} byte counts"
            )

        data = bytearray()
        for index, (offset, length) in enumerate(zip(offsets, counts, strict=True)):
            file.seek(offset)
            chunk = file.read(length)
            if len(chunk) < length:
                raise UnsupportedFileFormatError(
                    f"truncated TIFF strip {index}: expected {length} bytes, got {len(chunk)}"
                )
            data += chunk

        expected = self._width * self._height * SAMPLES_PER_PIXEL
        if len(data) < expected:
            raise UnsupportedFileFormatError(
                f"truncated TIFF pixel data: expected {expected} bytes, got {len(data)}"
            )
        return bytes(data[:expected])

    def load(self, filename: FileSource) -> None:
        """Read a TIFF from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Raises:
            UnsupportedFileFormatError: If the file is not a TIFF, or is one
                this profile does not cover.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            ifd_offset = self._read_header(file)
            entries = self._read_ifd(file, ifd_offset)

            self._width = self._single(entries, TAG_IMAGE_WIDTH)
            self._height = self._single(entries, TAG_IMAGE_LENGTH)
            self._validate(entries)
            data = self._read_strips(file, entries)

        self._raw_data = [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data), 3)]
        log.debug("loaded TIFF %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image as a little-endian single-strip uncompressed TIFF.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            OSError: If the file cannot be written.
        """
        body = bytes(channel for pixel in self._raw_data for channel in pixel)

        # Directory entries must be ordered by tag. The only value too large to
        # sit inside its entry is BitsPerSample, which follows the directory.
        directory_size = 2 + len(_ENTRY_TAGS) * ENTRY_SIZE + 4
        bits_offset = 8 + directory_size
        data_offset = bits_offset + SAMPLES_PER_PIXEL * 2

        entries = {
            TAG_IMAGE_WIDTH: (TYPE_LONG, 1, self._width),
            TAG_IMAGE_LENGTH: (TYPE_LONG, 1, self._height),
            TAG_BITS_PER_SAMPLE: (TYPE_SHORT, SAMPLES_PER_PIXEL, bits_offset),
            TAG_COMPRESSION: (TYPE_SHORT, 1, COMPRESSION_NONE),
            TAG_PHOTOMETRIC: (TYPE_SHORT, 1, PHOTOMETRIC_RGB),
            TAG_STRIP_OFFSETS: (TYPE_LONG, 1, data_offset),
            TAG_SAMPLES_PER_PIXEL: (TYPE_SHORT, 1, SAMPLES_PER_PIXEL),
            TAG_ROWS_PER_STRIP: (TYPE_LONG, 1, self._height),
            TAG_STRIP_BYTE_COUNTS: (TYPE_LONG, 1, len(body)),
            TAG_PLANAR_CONFIG: (TYPE_SHORT, 1, PLANAR_CHUNKY),
        }

        out = bytearray()
        out += struct.pack("<2sHI", TIFF_LITTLE_ENDIAN, TIFF_MAGIC, 8)
        out += struct.pack("<H", len(_ENTRY_TAGS))
        for tag in _ENTRY_TAGS:
            field_type, count, value = entries[tag]
            out += struct.pack("<HHI", tag, field_type, count)
            # The value is left-justified in its four bytes, so a SHORT sits in
            # the low half and the rest stays zero.
            out += struct.pack("<H2x" if field_type == TYPE_SHORT and count == 1 else "<I", value)
        out += struct.pack("<I", 0)  # no further directories
        out += struct.pack("<3H", *(BITS_PER_SAMPLE,) * SAMPLES_PER_PIXEL)
        out += body

        with open_binary_write(filename) as file:
            file.write(bytes(out))


#: Tags written by :meth:`TIFFImage.save`, in the ascending order the spec wants.
_ENTRY_TAGS = (
    TAG_IMAGE_WIDTH,
    TAG_IMAGE_LENGTH,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
    TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS,
    TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP,
    TAG_STRIP_BYTE_COUNTS,
    TAG_PLANAR_CONFIG,
)
