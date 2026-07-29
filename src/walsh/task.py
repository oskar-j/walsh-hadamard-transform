"""Orchestration: the compress and extract pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import Enum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from walsh.colors import RgbColorModel, YCbCrColorModel
from walsh.image import BlockDescription, CustomizableImage, FileSource, reader_for
from walsh.transforms import WalshHadamardTransform

__all__ = ["Action", "Task"]

log = logging.getLogger(__name__)

Block = npt.NDArray[np.float64]

#: Defaults carried over from the original implementation.
DEFAULT_Y_BLOCK_SIZE = 8
DEFAULT_CHROMA_BLOCK_SIZE = 16
DEFAULT_PACKED_BLOCK_SIZE = 4

#: Neutral fill values used when a channel carries no blocks.
NEUTRAL_LUMA = 0
NEUTRAL_CHROMA = 128


class Action(str, Enum):
    """What a :class:`Task` should do when run."""

    COMPRESS = "compress"
    EXTRACT = "extract"


class Task:
    """A single compress or extract run, configured fluently.

    >>> Task().with_action("compress").with_input("in.bmp").with_output("out.cim").run()
    ...                                                        # doctest: +SKIP

    :param y_block_size: block edge used for the luma channel.
    :param cb_block_size: block edge used for the Cb channel.
    :param cr_block_size: block edge used for the Cr channel.
    :param packed_block_size: how many low-frequency coefficients per axis are
        kept when writing. This is the codec's lossy knob.
    """

    def __init__(
        self,
        *,
        y_block_size: int = DEFAULT_Y_BLOCK_SIZE,
        cb_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        cr_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        packed_block_size: int = DEFAULT_PACKED_BLOCK_SIZE,
    ) -> None:
        """Create an unconfigured task with the default block geometry.

        Args:
            y_block_size: Block edge used for the luma channel.
            cb_block_size: Block edge used for the Cb channel.
            cr_block_size: Block edge used for the Cr channel.
            packed_block_size: How many low-frequency coefficients per axis are
                kept when writing. This is the codec's lossy knob.
        """
        self._input: FileSource = None
        self._output: FileSource = None
        self._action: Action | None = None
        self._coeff_removal: float | None = None
        self._y_block_size = y_block_size
        self._cb_block_size = cb_block_size
        self._cr_block_size = cr_block_size
        self._packed_block_size = packed_block_size

    # -- configuration ---------------------------------------------------

    def with_input(self, source: FileSource) -> Task:
        """Set where the input is read from.

        Args:
            source: Path to read, or ``None`` to read from stdin.

        Returns:
            This task, so calls can be chained.
        """
        self._input = source
        return self

    def with_output(self, destination: FileSource) -> Task:
        """Set where the result is written.

        For :meth:`extract` the suffix also selects the output raster format.

        Args:
            destination: Path to write, or ``None`` to write to stdout.

        Returns:
            This task, so calls can be chained.
        """
        self._output = destination
        return self

    def with_action(self, action: Action | str) -> Task:
        """Select which pipeline :meth:`run` will execute.

        Args:
            action: An :class:`Action`, or its string value.

        Returns:
            This task, so calls can be chained.

        Raises:
            ValueError: If ``action`` is not one of the known actions.
        """
        try:
            self._action = Action(action)
        except ValueError:
            valid = ", ".join(repr(a.value) for a in Action)
            raise ValueError(f"unknown action {action!r}; expected one of {valid}") from None
        return self

    def with_coeff_removal(self, coeff: float | None) -> Task:
        """Enable the second, independent lossy knob.

        Args:
            coeff: Threshold at or below which Hadamard matrix entries are
                zeroed during construction, or ``None`` to leave the matrix
                intact. See :class:`~walsh.transforms.WalshHadamardTransform`
                for how the comparison behaves.

        Returns:
            This task, so calls can be chained.
        """
        self._coeff_removal = coeff
        return self

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _get_padding_size(x: int, a: int) -> int:
        """Work out how much padding reaches the next multiple of ``a``.

        Args:
            x: The current size.
            a: The block size to align to.

        Returns:
            The number of elements to append, zero if ``x`` already fits.
        """
        return ((x - 1) // a + 1) * a - x

    def _slice(
        self, values: Sequence[float], width: int, height: int, block_size: int
    ) -> list[Block]:
        """Reshape a flat channel into zero-padded square blocks.

        Args:
            values: One channel's samples, row-major, ``width * height`` long.
            width: Image width in pixels.
            height: Image height in pixels.
            block_size: Edge length of the blocks to cut.

        Returns:
            The blocks in row-major order, each ``block_size`` square. The
            image is zero-padded up to a whole number of blocks first.
        """
        plane = np.asarray(values, dtype=np.float64).reshape(height, width)

        height_padding = self._get_padding_size(height, block_size)
        width_padding = self._get_padding_size(width, block_size)
        log.debug(
            "slicing %dx%d into %d-blocks (pad h=%d w=%d)",
            width,
            height,
            block_size,
            height_padding,
            width_padding,
        )
        plane = np.pad(plane, ((0, height_padding), (0, width_padding)))

        blocks: list[Block] = []
        for row in np.vsplit(plane, plane.shape[0] // block_size):
            blocks.extend(np.hsplit(row, row.shape[1] // block_size))

        log.debug("produced %d block(s) of %s", len(blocks), blocks[0].shape)
        return blocks

    @staticmethod
    def _merge(blocks: Sequence[Block], width: int, height: int) -> Block:
        """Reassemble blocks into a plane and crop the padding back off.

        The inverse of :meth:`_slice`.

        Args:
            blocks: Blocks in the row-major order :meth:`_slice` produced.
            width: Width to crop back to.
            height: Height to crop back to.

        Returns:
            The reassembled plane, of shape ``(height, width)``.
        """
        _, block_width = blocks[0].shape
        blocks_per_row = (width - 1) // block_width + 1
        blocks_per_column = len(blocks) // blocks_per_row
        rows = [
            np.hstack(blocks[i * blocks_per_row : (i + 1) * blocks_per_row])
            for i in range(blocks_per_column)
        ]
        return np.vstack(rows)[0:height, 0:width]

    # -- pipelines -------------------------------------------------------

    def compress(self) -> None:
        """Read a raster image, transform it, and write a spectral ``.cim``.

        The input format is chosen from the filename suffix, so this reads BMP
        or PPM without being told which.

        Raises:
            UnsupportedFileFormatError: If the input suffix is unknown, or the
                file is not valid for its format.
            OSError: If either file cannot be opened.
        """
        log.info("compressing %s -> %s", self._input, self._output)

        source_image = reader_for(self._input)
        source_image.load(self._input)

        width, height = source_image.get_dimensions()
        data = source_image.get_raw_data()

        color = RgbColorModel()
        y, cb, cr = zip(*(color.get_y_cb_cr(pixel) for pixel in data), strict=True)

        blocks = {
            "y": self._slice(y, width, height, self._y_block_size),
            "cb": self._slice(cb, width, height, self._cb_block_size),
            "cr": self._slice(cr, width, height, self._cr_block_size),
        }

        transform = WalshHadamardTransform(self._coeff_removal)
        spectral = {
            channel: transform.transform_sequence(channel_blocks)
            for channel, channel_blocks in blocks.items()
        }

        packed = self._packed_block_size
        customizable_image = CustomizableImage()
        customizable_image.set_dimensions(width, height)
        customizable_image.set_descriptions(
            BlockDescription(self._y_block_size, packed, len(spectral["y"])),
            BlockDescription(self._cb_block_size, packed, len(spectral["cb"])),
            BlockDescription(self._cr_block_size, packed, len(spectral["cr"])),
        )
        customizable_image.set_data(spectral["y"], spectral["cb"], spectral["cr"])
        customizable_image.save(self._output)

    def extract(self) -> None:
        """Read a spectral ``.cim``, invert the transform, and write a raster image.

        The output format is chosen from the filename suffix, so the picture can
        come back as a different format from the one it went in as.

        Raises:
            UnsupportedFileFormatError: If the output suffix is unknown, or
                the ``.cim`` file is truncated or malformed.
            OSError: If either file cannot be opened.
        """
        log.info("extracting %s -> %s", self._input, self._output)

        customizable_image = CustomizableImage.load(self._input)
        width, height = customizable_image.get_dimensions()
        transform = WalshHadamardTransform()

        channels = {
            "y": (customizable_image.get_y_data(), NEUTRAL_LUMA),
            "cb": (customizable_image.get_cb_data(), NEUTRAL_CHROMA),
            "cr": (customizable_image.get_cr_data(), NEUTRAL_CHROMA),
        }

        planes: dict[str, list[float]] = {}
        for channel, (spectral, neutral) in channels.items():
            if not spectral:
                log.debug("channel %s is empty, filling with %d", channel, neutral)
                planes[channel] = [float(neutral)] * (width * height)
                continue
            merged = self._merge(transform.inverse_transform_sequence(spectral), width, height)
            planes[channel] = np.asarray(merged).reshape(-1).tolist()

        color = YCbCrColorModel()
        pixels = [
            color.get_rgb(triple)
            for triple in zip(planes["y"], planes["cb"], planes["cr"], strict=True)
        ]

        output_image = reader_for(self._output)
        output_image.set_dimensions(width, height)
        output_image.set_raw_data(pixels)
        output_image.save(self._output)

    _ACTIONS: ClassVar[dict[Action, Callable[[Task], None]]] = {
        Action.COMPRESS: compress,
        Action.EXTRACT: extract,
    }

    def run(self) -> None:
        """Execute the configured action.

        Raises:
            ValueError: If :meth:`with_action` was never called.
            UnsupportedFileFormatError: If an input or output format is not
                supported.
            OSError: If a file cannot be read or written.
        """
        if self._action is None:
            raise ValueError("no action selected; call with_action() first")
        log.debug(
            "run action=%s input=%s output=%s coeff_removal=%s",
            self._action.value,
            self._input,
            self._output,
            self._coeff_removal,
        )
        Task._ACTIONS[self._action](self)
