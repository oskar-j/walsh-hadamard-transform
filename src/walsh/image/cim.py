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
from walsh.image._io import FileSource, open_binary_read, open_binary_write, read_up_to

__all__ = [
    "COEFF_DTYPE",
    "MAX_BLOCKS_PER_CHANNEL",
    "BlockDescription",
    "CustomizableImage",
    "blocks_for",
]

Block = npt.NDArray[np.float64]

#: Spectral coefficients are stored as little-endian signed 16-bit integers.
COEFF_DTYPE = np.dtype("<i2")
COEFF_MIN = int(np.iinfo(COEFF_DTYPE).min)
COEFF_MAX = int(np.iinfo(COEFF_DTYPE).max)

#: Channel keys, in the order they appear on disk.
CHANNELS = ("y", "cb", "cr")


def blocks_for(width: int, height: int, block_size: int) -> int:
    """Work out how many blocks a plane of this size is cut into.

    The encoder pads each axis up to a whole number of blocks, so this is
    exact rather than approximate: a valid ``.cim`` carries precisely this
    many blocks per channel, or none at all.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        block_size: Edge length of the blocks. Must be positive.

    Returns:
        Blocks per row times blocks per column.
    """
    return ((width - 1) // block_size + 1) * ((height - 1) // block_size + 1)


#: The most blocks one channel can describe. ``DESCRIPTION_FORMAT`` stores
#: ``number_of_blocks`` in a ``H``, an unsigned 16-bit field, so this is the
#: container's own ceiling and not a policy choice. At the default 8-pixel luma
#: block it caps an image at roughly 4.19 megapixels; a larger block size raises
#: it by the square of the ratio. Widening the field would change the on-disk
#: format and break every existing ``.cim``, so callers are expected to check
#: against this and report, rather than to silently truncate.
MAX_BLOCKS_PER_CHANNEL = 0xFFFF


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
        """Read the dimensions and the three block descriptions, then check them.

        Args:
            file: Stream positioned at the start of the file.

        Raises:
            UnsupportedFileFormatError: If the stream is too short to hold the
                header, or the header is not self-consistent; see
                :meth:`_validate_header`.
        """
        size = struct.calcsize(self.HEADER_FORMAT)
        self._width, self._height = struct.unpack(
            self.HEADER_FORMAT, self._read_exactly(file, size, "header")
        )
        size = struct.calcsize(self.DESCRIPTION_FORMAT)
        descriptions: dict[str, BlockDescription] = {}
        for channel in self._descriptions:
            raw = self._read_exactly(file, size, f"{channel} block description")
            descriptions[channel] = BlockDescription(*struct.unpack(self.DESCRIPTION_FORMAT, raw))
        self._validate_header(descriptions)
        self._descriptions.update(descriptions)

    def _validate_header(self, descriptions: dict[str, BlockDescription]) -> None:
        """Reject a header whose fields cannot describe one image.

        ``.cim`` has no signature, so any 26 bytes parse as a header, and every
        consumer downstream allocates, divides and reshapes on these fields.
        Fed the repository's own BMP, the old reader decoded it as a
        1396067650x7 image and tried to allocate 78 GB before failing; a
        crafted 26-byte file asked for 2 PiB and died with a bare
        ``MemoryError``. The geometry is fully determined, so it is checked
        here instead, before anything is allocated from it:

        1. both dimensions are positive;
        2. per channel, the block size is a positive power of two, which the
           transform requires anyway;
        3. the packed size is at least 1 and no larger than the block;
        4. the block count is either 0, the "empty channel" state that
           :meth:`~walsh.task.Task.extract` fills with a neutral value, or
           exactly the count a plane of these dimensions is cut into.

        It runs after all three descriptions are read, so a file cut short in
        the header is still reported as truncated rather than inconsistent.

        Args:
            descriptions: The three block descriptions, keyed by channel.

        Raises:
            UnsupportedFileFormatError: Naming the field that failed. The
                packed-size message is unchanged from earlier releases.
        """
        if self._width <= 0 or self._height <= 0:
            raise UnsupportedFileFormatError(
                f"invalid .cim header: dimensions must be positive, "
                f"got {self._width}x{self._height}"
            )
        for channel, (original, packed, count) in descriptions.items():
            if original < 1 or original & (original - 1):
                raise UnsupportedFileFormatError(
                    f"invalid .cim {channel} block description: block size {original} "
                    f"is not a positive power of two"
                )
            if packed > original:
                raise UnsupportedFileFormatError(
                    f"invalid .cim block description: packed size {packed} exceeds "
                    f"block size {original}"
                )
            if packed < 1:
                raise UnsupportedFileFormatError(
                    f"invalid .cim {channel} block description: packed size must be "
                    f"at least 1, got {packed}"
                )
            expected = blocks_for(self._width, self._height, original)
            if count not in (0, expected):
                raise UnsupportedFileFormatError(
                    f"invalid .cim {channel} block description: {count} blocks declared, "
                    f"but a {self._width}x{self._height} image in {original}-pixel blocks "
                    f"has {expected}"
                )

    @staticmethod
    def _read_blocks(file: BinaryIO, description: BlockDescription) -> list[Block]:
        """Read one channel's blocks, zero-padding each back to full size.

        Args:
            file: Stream positioned at the channel's first block.
            description: The layout record for this channel, already checked
                by :meth:`_validate_header`.

        Returns:
            One square array per block, of edge ``original_block_size``, with
            the stored coefficients in the top-left corner and zeros elsewhere.
            The blocks are views into one array, read and decoded in a single
            step rather than one ``struct.unpack`` per block.

        Raises:
            UnsupportedFileFormatError: If the stream ends mid-block.
        """
        original, packed, count = description
        block_size = packed * packed * COEFF_DTYPE.itemsize

        # Sized from a header field, so never asked for in one call: see read_up_to.
        raw = read_up_to(file, block_size * count)
        if len(raw) < block_size * count:
            index = len(raw) // block_size
            raise UnsupportedFileFormatError(
                f"truncated .cim block {index}: expected {block_size} bytes, "
                f"got {len(raw) - index * block_size}"
            )
        coefficients = np.frombuffer(raw, dtype=COEFF_DTYPE).astype(np.float64)

        blocks = np.zeros((count, original, original), dtype=np.float64)
        blocks[:, :packed, :packed] = coefficients.reshape(count, packed, packed)
        return list(blocks)

    @classmethod
    def load(cls, filename: FileSource) -> CustomizableImage:
        """Read a ``.cim`` file.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Returns:
            The populated container.

        Raises:
            UnsupportedFileFormatError: If the file is truncated, or its
                header does not describe a consistent image, which is how a
                file that is not a ``.cim`` at all is caught.
            OSError: If the file cannot be read.
        """
        image = cls()
        with open_binary_read(filename) as file:
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

        Raises:
            ValueError: If a channel declares more blocks than the container's
                16-bit count field can hold. Without this the overflow would
                surface from ``struct.pack`` during :meth:`save`, as a message
                naming neither the channel nor the limit. :class:`~walsh.task.Task`
                checks the same bound from the image dimensions before doing any
                work; this is the backstop for callers building a container
                directly.
        """
        for channel, description in (
            ("y", y_description),
            ("cb", cb_description),
            ("cr", cr_description),
        ):
            if description.number_of_blocks > MAX_BLOCKS_PER_CHANNEL:
                raise ValueError(
                    f"{channel} channel declares {description.number_of_blocks} blocks, "
                    f"but a .cim block count is a 16-bit field holding at most "
                    f"{MAX_BLOCKS_PER_CHANNEL}"
                )
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
        """Write one channel's blocks as little-endian ``int16``, in one write.

        Coefficients are rounded to nearest and clipped into the ``int16``
        range. Clipping cannot trigger for 8-bit input, where the largest
        possible coefficient is well inside the range. The blocks are stacked
        and encoded together, so a channel is one array operation and one
        ``write`` rather than one of each per block.

        Args:
            file: Stream to write to.
            blocks: The already-cropped blocks for one channel.
        """
        if not blocks:
            return
        data = np.clip(np.rint(np.stack(blocks)), COEFF_MIN, COEFF_MAX).astype(COEFF_DTYPE)
        file.write(data.tobytes())

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename``.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If any channel has no description set.
            OSError: If the file cannot be written.
        """
        with open_binary_write(filename) as file:
            self._write_header(file)
            for blocks in self._data.values():
                self._write_blocks(file, blocks)
