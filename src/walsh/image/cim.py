"""The custom ``.cim`` container holding truncated spectral blocks.

Layout: ``<II`` width and height, then three ``<HHH`` :class:`BlockDescription`
records in the order Y, Cb, Cr, then each channel's blocks row-major as
little-endian ``int16``. It is a project-specific format; nothing else reads it.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import BinaryIO, NamedTuple

import numpy as np
import numpy.typing as npt

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary

__all__ = ["COEFF_DTYPE", "BlockDescription", "CustomizableImage"]

Block = npt.NDArray[np.float64]

#: Spectral coefficients are stored as little-endian signed 16-bit integers.
COEFF_DTYPE = np.dtype("<i2")
COEFF_MIN = int(np.iinfo(COEFF_DTYPE).min)
COEFF_MAX = int(np.iinfo(COEFF_DTYPE).max)

#: Channel keys, in the order they appear on disk.
CHANNELS = ("y", "cb", "cr")


class BlockDescription(NamedTuple):
    """How one channel's blocks are laid out in a ``.cim`` file."""

    original_block_size: int
    packed_block_size: int
    number_of_blocks: int


class CustomizableImage:
    """A spectral image: dimensions, per-channel layouts, and coefficients.

    Unlike the raster formats this is not a :class:`~walsh.image.base.RasterImage`;
    it holds frequency-domain blocks rather than pixels, so it has no common
    interface to share with them.
    """

    HEADER_FORMAT = "<II"
    DESCRIPTION_FORMAT = "<HHH"

    def __init__(self) -> None:
        """Create an empty container with no descriptions and no blocks."""
        self._width = 0
        self._height = 0
        self._descriptions: dict[str, BlockDescription | None] = dict.fromkeys(CHANNELS)
        self._data: dict[str, list[Block]] = {channel: [] for channel in CHANNELS}

    @staticmethod
    def _read_exactly(file: BinaryIO, size: int, what: str) -> bytes:
        """Read exactly ``size`` bytes, or report the file as malformed.

        ``struct.unpack`` would otherwise raise ``struct.error``, which is not
        a :class:`~walsh.exceptions.WalshError` and so escapes callers that
        catch this package's own exceptions.

        Args:
            file: Stream to read from.
            size: Number of bytes required.
            what: What is being read, used only in the error message.

        Returns:
            Exactly ``size`` bytes.

        Raises:
            UnsupportedFileFormatError: If the stream ends first.
        """
        data = file.read(size)
        if len(data) < size:
            raise UnsupportedFileFormatError(
                f"truncated .cim {what}: expected {size} bytes, got {len(data)}"
            )
        return data

    def _read_header(self, file: BinaryIO) -> None:
        """Read the dimensions and the three block descriptions.

        Args:
            file: Stream positioned at the start of the file.

        Raises:
            UnsupportedFileFormatError: If the stream is too short to hold the
                header.
        """
        size = struct.calcsize(self.HEADER_FORMAT)
        self._width, self._height = struct.unpack(
            self.HEADER_FORMAT, self._read_exactly(file, size, "header")
        )
        size = struct.calcsize(self.DESCRIPTION_FORMAT)
        for channel in self._descriptions:
            raw = self._read_exactly(file, size, f"{channel} block description")
            self._descriptions[channel] = BlockDescription(
                *struct.unpack(self.DESCRIPTION_FORMAT, raw)
            )

    @staticmethod
    def _read_blocks(file: BinaryIO, description: BlockDescription) -> list[Block]:
        """Read one channel's blocks, zero-padding each back to full size.

        Args:
            file: Stream positioned at the channel's first block.
            description: The layout record for this channel.

        Returns:
            One square array per block, of edge ``original_block_size``, with
            the stored coefficients in the top-left corner and zeros elsewhere.

        Raises:
            UnsupportedFileFormatError: If the stream ends mid-block, or the
                description declares a packed size larger than the block.
        """
        original, packed, count = description
        if packed > original:
            raise UnsupportedFileFormatError(
                f"invalid .cim block description: packed size {packed} exceeds "
                f"block size {original}"
            )
        pattern = "<" + "h" * (packed * packed)
        size = struct.calcsize(pattern)

        blocks = []
        for index in range(count):
            raw = CustomizableImage._read_exactly(file, size, f"block {index}")
            data = struct.unpack(pattern, raw)
            block = np.zeros((original, original), dtype=np.float64)
            block[:packed, :packed] = np.asarray(data, dtype=np.float64).reshape(packed, packed)
            blocks.append(block)
        return blocks

    @classmethod
    def load(cls, filename: FileSource) -> CustomizableImage:
        """Read a ``.cim`` file.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Returns:
            The populated container.

        Raises:
            UnsupportedFileFormatError: If the file is truncated or not a
                ``.cim`` at all.
            OSError: If the file cannot be read.
        """
        image = cls()
        with open_binary(filename, "rb") as file:
            image._read_header(file)
            for channel, description in image._descriptions.items():
                if description is not None and description.number_of_blocks > 0:
                    image._data[channel] = cls._read_blocks(file, description)
        return image

    def get_y_data(self) -> list[Block]:
        """Return the luma blocks.

        Returns:
            The Y channel's blocks, empty if the file carried none.
        """
        return self._data["y"]

    def get_cb_data(self) -> list[Block]:
        """Return the blue-difference chroma blocks.

        Returns:
            The Cb channel's blocks, empty if the file carried none.
        """
        return self._data["cb"]

    def get_cr_data(self) -> list[Block]:
        """Return the red-difference chroma blocks.

        Returns:
            The Cr channel's blocks, empty if the file carried none.
        """
        return self._data["cr"]

    def get_dimensions(self) -> tuple[int, int]:
        """Return the dimensions of the picture these blocks encode.

        Returns:
            The ``(width, height)`` pair, in pixels.
        """
        return self._width, self._height

    def set_dimensions(self, width: int, height: int) -> None:
        """Record the dimensions of the picture these blocks encode.

        Args:
            width: Width in pixels.
            height: Height in pixels.
        """
        self._width = width
        self._height = height

    def set_descriptions(
        self,
        y_description: BlockDescription,
        cb_description: BlockDescription,
        cr_description: BlockDescription,
    ) -> None:
        """Record how each channel's blocks are laid out.

        Must be called before :meth:`set_data`, which reads the packed block
        size from these records.

        Args:
            y_description: Layout of the luma channel.
            cb_description: Layout of the blue-difference chroma channel.
            cr_description: Layout of the red-difference chroma channel.
        """
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
        cropped to ``packed_block_size`` squared coefficients.

        Args:
            y_data: Transformed luma blocks.
            cb_data: Transformed blue-difference chroma blocks.
            cr_data: Transformed red-difference chroma blocks.

        Raises:
            ValueError: If :meth:`set_descriptions` has not been called.
        """
        for channel, blocks in (("y", y_data), ("cb", cb_data), ("cr", cr_data)):
            description = self._descriptions[channel]
            if description is None:
                raise ValueError("set_descriptions() must be called before set_data()")
            packed = description.packed_block_size
            self._data[channel] = [block[:packed, :packed] for block in blocks]

    def _write_header(self, file: BinaryIO) -> None:
        """Write the dimensions and the three block descriptions.

        Args:
            file: Stream to write to.

        Raises:
            ValueError: If any channel has no description set.
        """
        file.write(struct.pack(self.HEADER_FORMAT, self._width, self._height))
        for channel, description in self._descriptions.items():
            if description is None:
                raise ValueError(f"no block description set for channel {channel!r}")
            file.write(struct.pack(self.DESCRIPTION_FORMAT, *description))

    @staticmethod
    def _write_blocks(file: BinaryIO, blocks: Sequence[Block]) -> None:
        """Write one channel's blocks as little-endian ``int16``.

        Coefficients are rounded to nearest and clipped into the ``int16``
        range. Clipping cannot trigger for 8-bit input, where the largest
        possible coefficient is well inside the range.

        Args:
            file: Stream to write to.
            blocks: The already-cropped blocks for one channel.
        """
        for block in blocks:
            data = np.clip(np.rint(block), COEFF_MIN, COEFF_MAX).astype(COEFF_DTYPE)
            file.write(data.reshape(-1).tobytes())

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename``.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If any channel has no description set.
            OSError: If the file cannot be written.
        """
        with open_binary(filename, "wb") as file:
            self._write_header(file)
            for blocks in self._data.values():
                self._write_blocks(file, blocks)
