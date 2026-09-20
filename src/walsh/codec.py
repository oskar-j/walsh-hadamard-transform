"""Orchestration: the compress and extract pipelines.

A :class:`Codec` is told what to do by :meth:`Codec.compress` or
:meth:`Codec.extract`, each naming its input and its output, and does it on
:meth:`Codec.run`::

    Codec().compress(input="photo.ppm", output="photo.cim").run()
    Codec().extract(input="photo.cim", output="restored.ppm").run()

``compress`` also accepts a picture as its output, which skips the ``.cim``
file: the picture goes through the whole codec in memory and what is written
is its lossy reconstruction, byte for byte what the two steps above produce::

    Codec().compress(input="photo.ppm", output="photo_compressed.ppm").run()
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from walsh.colors import rgb_to_ycbcr, ycbcr_to_rgb
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import (
    MAX_BLOCK_SIZE,
    MAX_BLOCKS_PER_CHANNEL,
    SUFFIXES,
    BlockDescription,
    CustomizableImage,
    FileSource,
    blocks_for,
    reader_for,
)
from walsh.transforms import (
    TRANSFORMS,
    Transform,
    WalshHadamardTransform,
    remove_small_coefficients,
    transform_for,
)

__all__ = ["Action", "Codec"]

log = logging.getLogger(__name__)

Block = npt.NDArray[np.float64]

#: Defaults carried over from the original implementation.
DEFAULT_Y_BLOCK_SIZE = 8
DEFAULT_CHROMA_BLOCK_SIZE = 16
DEFAULT_PACKED_BLOCK_SIZE = 4

#: Where writing a different file type from the one read is tracked. Until it
#: is done, :meth:`Codec.compress` refuses it and points here.
CROSS_FORMAT_ISSUE = "https://github.com/oskar-j/walsh-hadamard-transform/issues/51"

#: Neutral fill values used when a channel carries no blocks.
NEUTRAL_LUMA = 0
NEUTRAL_CHROMA = 128


class Action(str, Enum):
    """What a :class:`Codec` should do when run."""

    COMPRESS = "compress"
    EXTRACT = "extract"


class Codec:
    """A single compress or extract run, configured fluently.

    >>> Codec().compress(input="in.bmp", output="out.cim").run()  # doctest: +SKIP
    >>> Codec().extract(input="out.cim", output="back.bmp").run()  # doctest: +SKIP

    :param y_block_size: block edge used for the luma channel.
    :param cb_block_size: block edge used for the Cb channel.
    :param cr_block_size: block edge used for the Cr channel.
    :param packed_block_size: how many low-frequency coefficients per axis are
        kept when writing. This is the codec's lossy knob.
    :param transform: the block transform to use, for experiments. Not
        recorded in the ``.cim``; see :meth:`__init__`.
    """

    def __init__(
        self,
        *,
        y_block_size: int = DEFAULT_Y_BLOCK_SIZE,
        cb_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        cr_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        packed_block_size: int = DEFAULT_PACKED_BLOCK_SIZE,
        transform: Transform | str | None = None,
    ) -> None:
        """Create an unconfigured codec with the default block geometry.

        Args:
            y_block_size: Block edge used for the luma channel.
            cb_block_size: Block edge used for the Cb channel.
            cr_block_size: Block edge used for the Cr channel.
            packed_block_size: How many low-frequency coefficients per axis are
                kept when writing. This is the codec's lossy knob.
            transform: The block transform both directions run: a name,
                ``"walsh"``, ``"dct"`` or ``"haar"`` in any case (see
                :data:`~walsh.transforms.TRANSFORMS`), or an instance of a
                :class:`~walsh.transforms.Transform` subclass. ``None``, the
                default, is the Walsh-Hadamard transform, and the only one
                whose output is bit-exact across platforms. The others
                exist for experiments: a DCT or a Haar transform dropped in
                reuses the colour conversion, the padding, the crop to the
                packed corner and the container, so the comparison is between
                transforms and nothing else. **The ``.cim`` does not record
                which transform wrote it.** A file written with anything but
                the default is a ``.cim`` in name only: it must be extracted
                by a ``Codec`` given the same transform, and the ``walsh``
                command, which never takes one, will decode it without
                complaint into the wrong picture.

        Raises:
            TypeError: If ``transform`` is neither ``None``, a string nor a
                :class:`~walsh.transforms.Transform` instance, such as the
                class itself or a bare function.
            ValueError: If ``transform`` is a string that names no known
                transform, in which case the message lists the names that
                are. Also if a block edge is not a power of two, or exceeds
                ``MAX_BLOCK_SIZE``, or the packed size is not between 1 and the
                smallest block edge. Each of those used to fail late and
                quietly: an edge of 0 died in the padding arithmetic with a
                bare ``ZeroDivisionError``; an edge above the ceiling
                saturated the ``int16`` coefficients so a white picture came
                back mid-grey at exit 0; and a packed size above the edge was
                clamped by numpy on write but recorded unclamped in the
                header, producing a file the reader refuses forever, also at
                exit 0. The packed size need not be a power of two.
        """
        self._check_block_sizes(y_block_size, cb_block_size, cr_block_size, packed_block_size)
        self._transform = self._resolve_transform(transform)
        self._input: FileSource = None
        self._output: FileSource = None
        self._action: Action | None = None
        self._coeff_removal: float | None = None
        self._input_size: tuple[int, int] | None = None
        self._y_block_size = y_block_size
        self._cb_block_size = cb_block_size
        self._cr_block_size = cr_block_size
        self._packed_block_size = packed_block_size

    @staticmethod
    def _resolve_transform(transform: Transform | str | None) -> Transform:
        """Turn the constructor's ``transform`` argument into an instance.

        Args:
            transform: ``None``, a known name, or a transform instance.

        Returns:
            The Walsh-Hadamard transform for ``None``, a fresh instance for a
            name, and the instance itself otherwise.

        Raises:
            ValueError: If a name is not one of the known transforms.
            TypeError: If it is none of the three.
        """
        if transform is None:
            return WalshHadamardTransform()
        if isinstance(transform, str):
            return transform_for(transform)
        if isinstance(transform, Transform):
            return transform
        known = ", ".join(repr(name) for name in sorted(TRANSFORMS))
        raise TypeError(
            f"transform must be a Transform instance or one of {known}, got {transform!r}"
        )

    @staticmethod
    def _check_block_sizes(y: int, cb: int, cr: int, packed: int) -> None:
        """Reject block geometry the pipeline cannot honour, by name.

        Args:
            y: Luma block edge.
            cb: Cb block edge.
            cr: Cr block edge.
            packed: Coefficients kept per axis.

        Raises:
            ValueError: See :meth:`__init__`.
        """
        for name, edge in (("y_block_size", y), ("cb_block_size", cb), ("cr_block_size", cr)):
            if edge < 1 or edge & (edge - 1):
                raise ValueError(f"{name} must be a positive power of two, got {edge}")
            if edge > MAX_BLOCK_SIZE:
                raise ValueError(
                    f"{name} must be at most {MAX_BLOCK_SIZE}, got {edge}: above that a "
                    f"block's coefficients overflow the .cim container's int16 fields"
                )
        smallest = min(y, cb, cr)
        if packed < 1 or packed > smallest:
            raise ValueError(
                f"packed_block_size must be between 1 and the smallest block edge "
                f"({smallest}), got {packed}"
            )

    # -- configuration ---------------------------------------------------

    def compress(self, input: FileSource, output: FileSource) -> Codec:
        """Plan a compression of ``input``; :meth:`run` carries it out.

        What is written depends on what ``output`` is named. A picture suffix
        (``.ppm``, ``.png``, any key of :data:`~walsh.image.SUFFIXES`) skips
        the ``.cim`` file: the picture is compressed and restored in memory,
        and its lossy reconstruction is written in that format. The file is as
        large as any other picture of its size, since it is the *result* of
        the compression and not the compressed data, and it is byte for byte
        what compressing to a ``.cim`` and extracting that would have
        written, because the decoder is handed the very bytes the file would
        have held. Any other name, ``.cim`` by convention, gets the spectral
        container itself.

        Args:
            input: Picture to read, its format taken from the suffix, or
                ``None`` to read a BMP from ``sys.stdin``, which the CLI
                never does; see :data:`~walsh.image.FileSource`.
            output: Where to write: a ``.cim`` path, a picture path of the
                same file type as ``input``, or ``None`` to write the
                container to stdout.

        Returns:
            This codec, so calls can be chained.

        Raises:
            NotImplementedError: If ``output`` is a picture of a different
                file type from ``input``, such as ``.ppm`` to ``.png``. That
                is planned for 0.6.0; until then compress to a ``.cim`` and
                extract it, which crosses formats freely. Suffixes that share
                a reader, such as ``.tif`` and ``.tiff``, are one type.
            UnsupportedFileFormatError: If ``output`` is a picture and the
                suffix of ``input`` is not a format this package reads.
        """
        if self._writes_a_picture(output) and type(reader_for(input)) is not type(
            reader_for(output)
        ):
            suffix = Path(str(output)).suffix.lower()
            raise NotImplementedError(
                f"cannot compress {str(input)!r} straight to {str(output)!r}: writing a "
                f"different file type from the one read is not implemented yet and is "
                f"planned for 0.6.0 ({CROSS_FORMAT_ISSUE}). Until then keep the file type, "
                f"or compress to a .cim and extract that to {suffix}"
            )
        self._action, self._input, self._output = Action.COMPRESS, input, output
        return self

    def extract(self, input: FileSource, output: FileSource) -> Codec:
        """Plan the restoring of a picture from a ``.cim``; :meth:`run` does it.

        Args:
            input: The ``.cim`` to read, or ``None`` to read it from
                ``sys.stdin``.
            output: Picture to write, its format taken from the suffix, so it
                need not be the format the picture went in as. ``None`` writes
                a BMP to stdout.

        Returns:
            This codec, so calls can be chained.
        """
        self._action, self._input, self._output = Action.EXTRACT, input, output
        return self

    @staticmethod
    def _writes_a_picture(destination: FileSource) -> bool:
        """Tell whether a compression's output is a picture or the container.

        Args:
            destination: Where :meth:`compress` was told to write.

        Returns:
            ``True`` if the name ends in a suffix this package writes pictures
            for. Anything else (``.cim``, another suffix, none, or stdout)
            means the container, as it did before a picture could be named
            here.
        """
        return destination is not None and Path(destination).suffix.lower() in SUFFIXES

    def with_input_size(self, width: int | None, height: int | None) -> Codec:
        """Declare how large the input picture is, for input that cannot say.

        Needed by exactly one kind of input: a pickled flat list of pixels,
        ``[(r, g, b), ...]``, which does not carry its size. For any other
        input the declaration is checked against the file, so it is honoured
        or verified and never silently dropped. It applies to
        :meth:`compress` only.

        Args:
            width: Width of the input in pixels, or ``None``.
            height: Height of the input in pixels, or ``None``. Both or
                neither; two ``None`` clear a previous declaration.

        Returns:
            This codec, so calls can be chained.

        Raises:
            ValueError: If only one is given, or either is not a positive
                integer.
        """
        if width is None and height is None:
            self._input_size = None
            return self
        if width is None or height is None:
            raise ValueError("width and height must be declared together")
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"input {name} must be a positive integer, got {value!r}")
        self._input_size = (width, height)
        return self

    def with_coeff_removal(self, coeff: float | None) -> Codec:
        """Enable the second, independent lossy knob.

        Args:
            coeff: Magnitude below which spectral coefficients are zeroed,
                or ``None`` to keep every coefficient. Strict, so a
                coefficient exactly equal to ``coeff`` is kept. It acts on the
                *spectrum* of each block, never on the Hadamard matrix, whose
                entries all share one magnitude; see
                :func:`~walsh.transforms.remove_small_coefficients`. The codec
                applies it to the output of whichever transform it was given,
                so it works the same for a custom one. The value is absolute,
                so its effect scales with the block size, and it is consumed
                only by :meth:`compress`: :meth:`extract` never thresholds.

        Returns:
            This codec, so calls can be chained.

        Raises:
            ValueError: If ``coeff`` is negative. It is compared against a
                magnitude, so a negative value could only be a mistake: it
                would silently keep everything.
        """
        if coeff is not None and coeff < 0:
            raise ValueError(f"coeff must be non-negative, got {coeff}")
        self._coeff_removal = coeff
        return self

    # -- helpers ---------------------------------------------------------

    def _same_shape(self, result: Block, given: Block, method: str) -> Block:
        """Return ``result`` as an array, insisting it has the shape of ``given``.

        The built-in transform cannot fail this; it is for a custom one, where
        a wrong shape would otherwise surface far away as a container error.

        Args:
            result: What the transform returned.
            given: The stack it was given.
            method: The method that returned it, for the message.

        Returns:
            ``result`` as a ``float64`` array.

        Raises:
            ValueError: If the shapes differ, naming the transform's class.
        """
        array = np.asarray(result, dtype=np.float64)
        if array.shape != given.shape:
            raise ValueError(
                f"{type(self._transform).__name__}.{method} returned shape {array.shape} "
                f"for a stack of shape {given.shape}"
            )
        return array

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

    @staticmethod
    def _count_blocks(width: int, height: int, block_size: int) -> int:
        """Work out how many blocks a plane of this size will be cut into.

        Matches what :meth:`_slice` produces, without building anything: the
        image is padded up to a whole number of blocks on each axis.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            block_size: Edge length of the blocks.

        Returns:
            The block count, row-major blocks per row times blocks per column.
            Delegates to :func:`walsh.image.blocks_for`, which the ``.cim``
            reader also uses to check a header, so encoder and reader cannot
            disagree about it.
        """
        return blocks_for(width, height, block_size)

    def _check_fits_the_container(self, width: int, height: int) -> None:
        """Reject an image with more blocks than ``.cim`` can count.

        The container stores each channel's block count in a 16-bit field, so
        at the default 8-pixel luma block the codec caps out near 4.19
        megapixels -- below any phone photo. The overflow used to surface from
        ``struct.pack`` while writing the header, naming neither the channel,
        the limit, nor the option that raises it, and only after the whole
        image had been read, converted and transformed. The count follows from
        the dimensions alone, so it is checked here instead, before any of that
        work happens.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Raises:
            UnsupportedFileFormatError: If any channel would need more than
                ``MAX_BLOCKS_PER_CHANNEL`` blocks. The message names the
                channel, its count, the limit, and the block size that would
                bring the image inside it.
        """
        channels = (
            ("luma", self._y_block_size),
            ("Cb", self._cb_block_size),
            ("Cr", self._cr_block_size),
        )
        for channel, block_size in channels:
            blocks = self._count_blocks(width, height, block_size)
            if blocks <= MAX_BLOCKS_PER_CHANNEL:
                continue
            # The count falls by the square of the block size, so this is the
            # smallest power of two that brings the image inside the limit.
            sufficient = block_size
            while self._count_blocks(width, height, sufficient) > MAX_BLOCKS_PER_CHANNEL:
                sufficient *= 2
            option = "--y-block-size" if channel == "luma" else "--chroma-block-size"
            ceiling = MAX_BLOCKS_PER_CHANNEL * block_size * block_size
            in_words = (
                f"{ceiling / 1_000_000:.1f} megapixels"
                if ceiling >= 1_000_000
                else f"{ceiling} pixels"
            )
            raise UnsupportedFileFormatError(
                f"image is too large for the .cim container: a {width}x{height} image "
                f"needs {blocks} {channel} blocks of {block_size} pixels, and the format "
                f"stores at most {MAX_BLOCKS_PER_CHANNEL} per channel, which is "
                f"{in_words} at this block size. "
                f"Retry with {option} {sufficient} or larger, or scale the image down."
            )

    def _slice(self, values: npt.ArrayLike, width: int, height: int, block_size: int) -> Block:
        """Reshape a flat channel into square blocks, padding the edge outwards.

        Args:
            values: One channel's samples, row-major, ``width * height`` long.
            width: Image width in pixels.
            height: Image height in pixels.
            block_size: Edge length of the blocks to cut.

        Returns:
            The blocks as one ``(count, block_size, block_size)`` array in
            row-major order, cut by a single reshape. The image is first
            padded up to a whole number of blocks by replicating its last row
            and column, so the fill carries no content of its own; see the
            note in the body for why that matters.
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
        # Replicate the edge rather than padding with zeros. The padding shares
        # its blocks with real pixels, and the transform is low-pass, so
        # whatever fills it is smeared back across the last few real columns
        # and rows. Zero is black in luma and fully saturated in chroma, which
        # is why it showed up as a coloured seam; the edge sample is the
        # cheapest fill that cannot introduce an edge that was not there. This
        # is what JPEG does for the same reason.
        plane = np.pad(plane, ((0, height_padding), (0, width_padding)), mode="edge")

        rows, columns = plane.shape[0] // block_size, plane.shape[1] // block_size
        # Split each axis into (block index, offset within block), then bring
        # the two block indices to the front so the blocks come out row-major.
        grid = plane.reshape(rows, block_size, columns, block_size).swapaxes(1, 2)
        blocks = grid.reshape(rows * columns, block_size, block_size)

        log.debug("produced %d block(s) of %s", len(blocks), blocks.shape[1:])
        return blocks

    @staticmethod
    def _merge(blocks: Block | Sequence[Block], width: int, height: int) -> Block:
        """Reassemble blocks into a plane and crop the padding back off.

        The inverse of :meth:`_slice`.

        Args:
            blocks: The blocks in the row-major order :meth:`_slice` produced,
                as one ``(count, edge, edge)`` array or a sequence of blocks.
                An array passes through without a copy; a sequence is stacked.
            width: Width to crop back to.
            height: Height to crop back to.

        Returns:
            The reassembled plane, of shape ``(height, width)``.

        Raises:
            ValueError: If the number of blocks is not what a plane of these
                dimensions is cut into. The ``.cim`` reader guarantees this
                for anything it accepts; the check is for direct callers.
        """
        stacked = np.asarray(blocks, dtype=np.float64)
        _, block_height, block_width = stacked.shape
        blocks_per_row = (width - 1) // block_width + 1
        blocks_per_column = (height - 1) // block_height + 1
        if len(blocks) != blocks_per_row * blocks_per_column:
            raise ValueError(
                f"{len(blocks)} blocks cannot tile a {width}x{height} plane in "
                f"{block_height}x{block_width} blocks, which takes "
                f"{blocks_per_row * blocks_per_column}"
            )
        # The inverse of the reshape in _slice: lay the blocks out on their
        # grid, then interleave the block index with the offset within it.
        grid = stacked.reshape(blocks_per_column, blocks_per_row, block_height, block_width)
        plane = grid.swapaxes(1, 2).reshape(
            blocks_per_column * block_height, blocks_per_row * block_width
        )
        return plane[:height, :width]

    # -- pipelines -------------------------------------------------------

    def _encode(self) -> CustomizableImage:
        """Read the input picture and transform it into a spectral container.

        The input format is chosen from the filename suffix.

        Returns:
            The container, not yet written anywhere. Its blocks are cropped to
            the packed corner but still unrounded: the rounding to ``int16``
            happens on the way out of it, to a file or to bytes.

        Raises:
            UnsupportedFileFormatError: If the input suffix is unknown, the
                file is not valid for its format, or the image needs more
                blocks than the ``.cim`` container can count.
            ValueError: If the picture is not the size that was declared.
            OSError: If the input cannot be opened.
        """
        source_image = reader_for(self._input)
        if self._input_size is not None:
            source_image.declare_size(*self._input_size)
        source_image.load(self._input)

        width, height = source_image.get_dimensions()
        if self._input_size is not None and self._input_size != (width, height):
            raise ValueError(
                f"{self._input} is {width}x{height}, not the "
                f"{self._input_size[0]}x{self._input_size[1]} declared"
            )
        self._check_fits_the_container(width, height)
        ycbcr = rgb_to_ycbcr(source_image.get_array().reshape(-1, 3))

        blocks = {
            "y": self._slice(ycbcr[:, 0], width, height, self._y_block_size),
            "cb": self._slice(ycbcr[:, 1], width, height, self._cb_block_size),
            "cr": self._slice(ycbcr[:, 2], width, height, self._cr_block_size),
        }

        # Each channel is one stack and the transform takes a stack as it is:
        # no list of blocks is built, re-stacked or split anywhere in between.
        # Coefficient removal is applied here rather than by the transform, so
        # it works for whichever transform the codec was given.
        spectral: dict[str, Block] = {}
        for channel, channel_blocks in blocks.items():
            spectrum = self._same_shape(
                self._transform.transform_stack(channel_blocks), channel_blocks, "transform_stack"
            )
            if self._coeff_removal is not None:
                spectrum = remove_small_coefficients(spectrum, self._coeff_removal)
            spectral[channel] = spectrum

        packed = self._packed_block_size
        customizable_image = CustomizableImage()
        customizable_image.set_dimensions(width, height)
        customizable_image.set_descriptions(
            BlockDescription(self._y_block_size, packed, len(spectral["y"])),
            BlockDescription(self._cb_block_size, packed, len(spectral["cb"])),
            BlockDescription(self._cr_block_size, packed, len(spectral["cr"])),
        )
        customizable_image.set_data(spectral["y"], spectral["cb"], spectral["cr"])
        return customizable_image

    def _decode(self, customizable_image: CustomizableImage) -> None:
        """Invert the transform and write the picture to the output.

        The output format is chosen from the filename suffix.

        Args:
            customizable_image: A container as the reader hands it over, its
                blocks zero-padded back to full size.

        Raises:
            UnsupportedFileFormatError: If the output suffix is unknown.
            OSError: If the output cannot be written.
        """
        width, height = customizable_image.get_dimensions()

        neutral = {"y": NEUTRAL_LUMA, "cb": NEUTRAL_CHROMA, "cr": NEUTRAL_CHROMA}

        planes: dict[str, npt.NDArray[np.float64]] = {}
        for channel, fill in neutral.items():
            spectral = customizable_image.get_stack(channel)
            if len(spectral) == 0:
                log.debug("channel %s is empty, filling with %d", channel, fill)
                planes[channel] = np.full(width * height, float(fill))
                continue
            restored = self._same_shape(
                self._transform.inverse_transform_stack(spectral),
                spectral,
                "inverse_transform_stack",
            )
            merged = self._merge(restored, width, height)
            planes[channel] = merged.reshape(-1)

        ycbcr = np.stack([planes["y"], planes["cb"], planes["cr"]], axis=1)
        pixels = ycbcr_to_rgb(ycbcr).reshape(height, width, 3)

        output_image = reader_for(self._output)
        output_image.set_array(pixels)
        output_image.save(self._output)

    def _compress(self) -> None:
        """Run a compression: to a ``.cim``, or straight through to a picture.

        With a picture as the output the container never reaches a disk, but
        it is still serialised and parsed back, in memory. That is deliberate.
        The container in hand holds unrounded blocks cropped to their packed
        corner, while a decoder is owed what a file would give it: ``int16``
        coefficients, rounded and clipped, zero-padded to full blocks. Going
        through the bytes is what makes this output identical to the two-step
        one by construction, for a few hundred kilobytes of copying.

        Raises:
            UnsupportedFileFormatError: If a suffix is unknown, the input is
                not valid for its format, or the image needs more blocks than
                the ``.cim`` container can count.
            ValueError: If the picture is not the size that was declared.
            OSError: If either file cannot be opened.
        """
        log.info("compressing %s -> %s", self._input, self._output)
        customizable_image = self._encode()
        if self._writes_a_picture(self._output):
            self._decode(CustomizableImage.from_bytes(customizable_image.to_bytes()))
        else:
            customizable_image.save(self._output)

    def _extract(self) -> None:
        """Run an extraction: read a ``.cim`` and write the picture it holds.

        Raises:
            ValueError: If an input size was declared, which means nothing
                here.
            UnsupportedFileFormatError: If the output suffix is unknown, or
                the ``.cim`` file is truncated or malformed.
            OSError: If either file cannot be opened.
        """
        if self._input_size is not None:
            raise ValueError(
                "an input size applies to compress only: a .cim records its own dimensions"
            )
        log.info("extracting %s -> %s", self._input, self._output)
        self._decode(CustomizableImage.load(self._input))

    _ACTIONS: ClassVar[dict[Action, Callable[[Codec], None]]] = {
        Action.COMPRESS: _compress,
        Action.EXTRACT: _extract,
    }

    def run(self) -> None:
        """Execute the configured action.

        Raises:
            ValueError: If neither :meth:`compress` nor :meth:`extract` was
                called first.
            UnsupportedFileFormatError: If an input or output format is not
                supported.
            OSError: If a file cannot be read or written.
        """
        if self._action is None:
            raise ValueError("nothing to run; call compress() or extract() first")
        log.debug(
            "run action=%s input=%s output=%s coeff_removal=%s transform=%s",
            self._action.value,
            self._input,
            self._output,
            self._coeff_removal,
            type(self._transform).__name__,
        )
        Codec._ACTIONS[self._action](self)
