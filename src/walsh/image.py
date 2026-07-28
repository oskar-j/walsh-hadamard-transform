"""Container formats: 24-bit BMP in, custom ``.cim`` spectral format out."""

from __future__ import annotations

import contextlib
import logging
import struct
import sys
from collections.abc import Generator, Sequence
from os import PathLike
from typing import BinaryIO, NamedTuple

import numpy as np
import numpy.typing as npt

from walsh.exceptions import UnsupportedFileFormatError

# Re-exported so `from walsh.image import UnsupportedFileFormatError` keeps
# working; it is defined in walsh.exceptions.
__all__ = [
    "BMPImage",
    "BlockDescription",
    "CustomizableImage",
    "UnsupportedFileFormatError",
    "align",
]

log = logging.getLogger(__name__)

FileSource = str | PathLike[str] | None

Block = npt.NDArray[np.float64]
Pixel = tuple[int, int, int]

BMP_HEADER_FORMAT = "<2sIHHIIIIHHIIIIII"
BMP_SIGNATURE = b"BM"
BMP_HEADER_SIZE = 40
BMP_PIXEL_OFFSET = 54

#: Spectral coefficients are stored as little-endian signed 16-bit integers.
COEFF_DTYPE = np.dtype("<i2")
COEFF_MIN = np.iinfo(COEFF_DTYPE).min
COEFF_MAX = np.iinfo(COEFF_DTYPE).max


def align(x: int, a: int) -> int:
    """Round ``x`` up to the next multiple of ``a``."""
    return (((x - 1) // a) + 1) * a


@contextlib.contextmanager
def _open_binary(source: FileSource, mode: str) -> Generator[BinaryIO, None, None]:
    """Open ``source``, or fall back to std streams when it is ``None``.

    The std streams are deliberately not closed on exit.
    """
    if source is None:
        yield sys.stdin.buffer if "r" in mode else sys.stdout.buffer
        return
    with open(source, mode) as handle:
        yield handle  # type: ignore[misc]


class BMPImage:
    """A 24-bit uncompressed Windows bitmap.

    .. note::
       Pixel triples are kept in the order they appear in the file, which for
       BMP is blue-green-red. Reading and writing use the same order, so a
       round trip is lossless, but callers that interpret a triple as
       ``(r, g, b)`` are working with the channels swapped.
    """

    def __init__(self) -> None:
        self._signature = BMP_SIGNATURE
        self._size = 0
        self._offset = BMP_PIXEL_OFFSET
        self._header_size = BMP_HEADER_SIZE
        self._width = 0
        self._height = 0
        self._planes = 1
        self._bpp = 24
        self._compression = 0
        self._size_of_data = 0
        self._horizontal_res = 2835
        self._vertical_res = 2835
        self._raw_data: list[Pixel] = []

    def _read_header(self, file: BinaryIO) -> None:
        header = file.read(struct.calcsize(BMP_HEADER_FORMAT))
        (
            self._signature,
            self._size,
            _,
            _,
            self._offset,
            self._header_size,
            self._width,
            self._height,
            self._planes,
            self._bpp,
            self._compression,
            self._size_of_data,
            self._horizontal_res,
            self._vertical_res,
            _,
            _,
        ) = struct.unpack(BMP_HEADER_FORMAT, header)

        actual = (self._signature, self._planes, self._bpp, self._compression)
        if actual != (BMP_SIGNATURE, 1, 24, 0):
            raise UnsupportedFileFormatError(
                "expected a 24-bit single-plane uncompressed BMP, got "
                f"signature={self._signature!r} planes={self._planes} "
                f"bpp={self._bpp} compression={self._compression}"
            )

    def _read_data(self, file: BinaryIO) -> None:
        file.seek(self._offset)
        stride = align(self._width * 3, 4)
        self._raw_data = []
        for _ in range(self._height):
            line = file.read(stride)
            for i in range(0, self._width * 3, 3):
                self._raw_data.append(struct.unpack_from("<BBB", line, i))

    def load(self, filename: FileSource) -> None:
        """Read a BMP from ``filename``, or from stdin when it is ``None``."""
        with _open_binary(filename, "rb") as file:
            self._read_header(file)
            self._read_data(file)
        log.debug("loaded BMP %dx%d from %s", self._width, self._height, filename)

    def _write_header(self, file: BinaryIO) -> None:
        ignored = 0
        file.write(
            struct.pack(
                BMP_HEADER_FORMAT,
                self._signature,
                self._size,
                ignored,
                ignored,
                self._offset,
                self._header_size,
                self._width,
                self._height,
                self._planes,
                self._bpp,
                self._compression,
                self._size_of_data,
                self._horizontal_res,
                self._vertical_res,
                ignored,
                ignored,
            )
        )

    def _write_data(self, file: BinaryIO) -> None:
        padding = align(self._width * 3, 4) - self._width * 3
        log.debug("row padding: %d byte(s)", padding)
        pad = b"\x00" * padding
        for start in range(0, len(self._raw_data), self._width):
            for pixel in self._raw_data[start : start + self._width]:
                file.write(struct.pack("<BBB", *pixel))
            file.write(pad)

    def save(self, filename: FileSource) -> None:
        """Write this bitmap to ``filename``, or to stdout when it is ``None``."""
        with _open_binary(filename, "wb") as file:
            self._write_header(file)
            self._write_data(file)

    def get_dimensions(self) -> tuple[int, int]:
        return self._width, self._height

    def set_dimensions(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._size_of_data = width * height * 3
        self._size = BMP_PIXEL_OFFSET + align(width * 3, 4) * height

    def get_raw_data(self) -> list[Pixel]:
        return self._raw_data

    def set_raw_data(self, new_data: Sequence[Pixel]) -> None:
        self._raw_data = list(new_data)


class BlockDescription(NamedTuple):
    """How one channel's blocks are laid out in a ``.cim`` file."""

    original_block_size: int
    packed_block_size: int
    number_of_blocks: int


class CustomizableImage:
    """The custom ``.cim`` container holding truncated spectral blocks.

    Layout: ``<II`` width/height, then three ``<HHH`` :class:`BlockDescription`
    records (Y, Cb, Cr), then each channel's blocks as row-major ``int16``.
    """

    HEADER_FORMAT = "<II"
    DESCRIPTION_FORMAT = "<HHH"

    def __init__(self) -> None:
        self._width = 0
        self._height = 0
        self._descriptions: dict[str, BlockDescription | None] = {
            "y": None,
            "cb": None,
            "cr": None,
        }
        self._data: dict[str, list[Block]] = {"y": [], "cb": [], "cr": []}

    def _read_header(self, file: BinaryIO) -> None:
        size = struct.calcsize(self.HEADER_FORMAT)
        self._width, self._height = struct.unpack(self.HEADER_FORMAT, file.read(size))
        size = struct.calcsize(self.DESCRIPTION_FORMAT)
        for channel in self._descriptions:
            fields = struct.unpack(self.DESCRIPTION_FORMAT, file.read(size))
            self._descriptions[channel] = BlockDescription(*fields)

    @staticmethod
    def _read_blocks(file: BinaryIO, description: BlockDescription) -> list[Block]:
        original, packed, count = description
        pattern = "<" + "h" * (packed * packed)
        size = struct.calcsize(pattern)

        blocks = []
        for _ in range(count):
            data = struct.unpack(pattern, file.read(size))
            block = np.zeros((original, original), dtype=np.float64)
            block[:packed, :packed] = np.asarray(data, dtype=np.float64).reshape(packed, packed)
            blocks.append(block)
        return blocks

    @classmethod
    def load(cls, filename: FileSource) -> CustomizableImage:
        """Read a ``.cim`` from ``filename``, or from stdin when it is ``None``."""
        image = cls()
        with _open_binary(filename, "rb") as file:
            image._read_header(file)
            for channel, description in image._descriptions.items():
                if description is not None and description.number_of_blocks > 0:
                    image._data[channel] = cls._read_blocks(file, description)
        return image

    def get_y_data(self) -> list[Block]:
        return self._data["y"]

    def get_cb_data(self) -> list[Block]:
        return self._data["cb"]

    def get_cr_data(self) -> list[Block]:
        return self._data["cr"]

    def get_dimensions(self) -> tuple[int, int]:
        return self._width, self._height

    def set_dimensions(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def set_descriptions(
        self,
        y_description: BlockDescription,
        cb_description: BlockDescription,
        cr_description: BlockDescription,
    ) -> None:
        self._descriptions["y"] = BlockDescription(*y_description)
        self._descriptions["cb"] = BlockDescription(*cb_description)
        self._descriptions["cr"] = BlockDescription(*cr_description)

    def set_data(
        self,
        y_data: Sequence[Block],
        cb_data: Sequence[Block],
        cr_data: Sequence[Block],
    ) -> None:
        """Store blocks, keeping only the low-frequency corner of each.

        This is where the codec actually loses information: each block is
        cropped to ``packed_block_size`` x ``packed_block_size``.
        """
        for channel, blocks in (("y", y_data), ("cb", cb_data), ("cr", cr_data)):
            description = self._descriptions[channel]
            if description is None:
                raise ValueError("set_descriptions() must be called before set_data()")
            packed = description.packed_block_size
            self._data[channel] = [block[:packed, :packed] for block in blocks]

    def _write_header(self, file: BinaryIO) -> None:
        file.write(struct.pack(self.HEADER_FORMAT, self._width, self._height))
        for channel, description in self._descriptions.items():
            if description is None:
                raise ValueError(f"no block description set for channel {channel!r}")
            file.write(struct.pack(self.DESCRIPTION_FORMAT, *description))

    @staticmethod
    def _write_blocks(file: BinaryIO, blocks: Sequence[Block]) -> None:
        for block in blocks:
            data = np.clip(np.rint(block), COEFF_MIN, COEFF_MAX).astype(COEFF_DTYPE)
            file.write(data.reshape(-1).tobytes())

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename``, or to stdout when it is ``None``."""
        with _open_binary(filename, "wb") as file:
            self._write_header(file)
            for blocks in self._data.values():
                self._write_blocks(file, blocks)
