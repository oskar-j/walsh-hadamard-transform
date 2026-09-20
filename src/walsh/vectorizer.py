"""A picture as coefficient vectors: look at them, measure them, save them.

:class:`~walsh.codec.Codec` goes from one file to another. A
:class:`Vectorizer` stops in the middle, where the picture is a table of
numbers, and hands that table over::

    vectorizer = Vectorizer(transform="walsh").parse(file_name="photo.png").compute()
    vectorizer.vectors          # int16, one row per block
    print(vectorizer.describe())
    vectorizer.save(output_file_name="photo.cim")

Each block of the picture becomes one *vector*: the ``packed_block_size``
squared coefficients the codec keeps of it, row-major, low frequencies first.
Every channel keeps the same number per block, so all of them fit one array of
shape ``(blocks, packed_block_size ** 2)``: the luma blocks first, then Cb,
then Cr, each in row-major order over the picture. That is the order of the
``.cim`` file, and the array is ``int16`` because the file is, so
``vectors.tobytes()`` is exactly the file after its 26-byte header.

The vectors are the object's state, not a copy of it. Change them and
:meth:`Vectorizer.describe`, :meth:`Vectorizer.reconstruct` and
:meth:`Vectorizer.save` all follow, which is what makes this a bench for
experiments: zero a column and see what the picture loses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from walsh.codec import (
    DEFAULT_CHROMA_BLOCK_SIZE,
    DEFAULT_PACKED_BLOCK_SIZE,
    DEFAULT_Y_BLOCK_SIZE,
    Codec,
)
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import (
    COEFF_DTYPE,
    SUFFIXES,
    BlockDescription,
    CustomizableImage,
    FileSource,
    PixelArray,
    reader_for,
)
from walsh.transforms import Transform

__all__ = ["CompressionStats", "Vectorizer"]

#: The coefficient vectors: ``int16``, shape ``(blocks, packed_block_size ** 2)``.
Vectors = npt.NDArray[np.int16]

_CHANNELS = ("y", "cb", "cr")
_CONTAINER_SUFFIX = ".cim"
_PEAK = 255.0


@dataclass(frozen=True)
class CompressionStats:
    """What :meth:`Vectorizer.describe` reports. ``print()`` it for a table.

    Attributes:
        width: Picture width in pixels.
        height: Picture height in pixels.
        transform: Name of the transform class in use.
        y_block_size: Luma block edge.
        cb_block_size: Cb block edge.
        cr_block_size: Cr block edge.
        packed_block_size: Coefficients kept per axis of every block.
        y_vectors: Luma blocks, which are the first rows of the vectors.
        cb_vectors: Cb blocks, the rows after those.
        cr_vectors: Cr blocks, the last rows.
        vector_length: Coefficients in one vector.
        coefficients: Coefficients stored in all.
        nonzero_coefficients: How many of them are not zero. Coefficient
            removal lowers this without shrinking the file.
        kept_share: Coefficients stored as a share of the samples the
            picture has, ``width * height * 3``, in percent.
        raw_bytes: The picture as plain 8-bit RGB, ``width * height * 3``.
        source_bytes: Size of the file that was parsed, or ``None`` when
            there is none to measure: after ``load``, or reading stdin.
        compressed_bytes: Size of the ``.cim`` these vectors make.
        reduction_percent: How much smaller the ``.cim`` is than the raw
            pixels, in percent.
        source_reduction_percent: The same against ``source_bytes``, which is
            negative when the source was itself compressed, as a PNG is. Or
            ``None``.
        bits_per_pixel: ``compressed_bytes`` in bits, per pixel.
        psnr_db: Peak signal-to-noise ratio of the reconstruction against the
            parsed picture, in decibels over all three channels; ``inf`` when
            they are equal, ``None`` after ``load``, when there is no original
            to compare with.
        mean_squared_error: What ``psnr_db`` is computed from, or ``None``.
        max_error: The largest difference in any sample, or ``None``.
    """

    width: int
    height: int
    transform: str
    y_block_size: int
    cb_block_size: int
    cr_block_size: int
    packed_block_size: int
    y_vectors: int
    cb_vectors: int
    cr_vectors: int
    vector_length: int
    coefficients: int
    nonzero_coefficients: int
    kept_share: float
    raw_bytes: int
    source_bytes: int | None
    compressed_bytes: int
    reduction_percent: float
    source_reduction_percent: float | None
    bits_per_pixel: float
    psnr_db: float | None
    mean_squared_error: float | None
    max_error: int | None

    def __str__(self) -> str:
        """Lay the figures out as an aligned table.

        Returns:
            One line per figure. Those that need the original picture say so
            when there is none.
        """
        vectors = self.y_vectors + self.cb_vectors + self.cr_vectors
        nonzero = 100 * self.nonzero_coefficients / max(self.coefficients, 1)
        rows = [
            ("picture", f"{self.width} x {self.height}"),
            ("transform", self.transform),
            (
                "blocks",
                f"Y {self.y_block_size}, Cb {self.cb_block_size}, Cr {self.cr_block_size}; "
                f"{self.packed_block_size} x {self.packed_block_size} kept of each",
            ),
            (
                "vectors",
                f"{vectors:,} of {self.vector_length} "
                f"(Y {self.y_vectors:,}, Cb {self.cb_vectors:,}, Cr {self.cr_vectors:,})",
            ),
            (
                "coefficients",
                f"{self.coefficients:,}, {self.kept_share:.2f}% of the picture's samples; "
                f"{self.nonzero_coefficients:,} non-zero ({nonzero:.1f}%)",
            ),
            ("raw pixels", f"{self.raw_bytes:,} B"),
            ("source file", "none" if self.source_bytes is None else f"{self.source_bytes:,} B"),
            (
                "compressed",
                f"{self.compressed_bytes:,} B, {self.bits_per_pixel:.2f} bits per pixel",
            ),
            ("reduction", self._reduction()),
            ("PSNR", self._quality()),
        ]
        return "\n".join(f"{name:<13} {value}" for name, value in rows)

    def _reduction(self) -> str:
        """Word the size reduction, against the source file too when known.

        Returns:
            The reduction line of the table.
        """
        text = f"{self.reduction_percent:.1f}% smaller than the raw pixels"
        if self.source_reduction_percent is not None:
            change = self.source_reduction_percent
            direction = "smaller" if change >= 0 else "larger"
            text += f", {abs(change):.1f}% {direction} than the source file"
        return text

    def _quality(self) -> str:
        """Word the quality figures, or why there are none.

        Returns:
            The PSNR line of the table.
        """
        if self.psnr_db is None:
            return "unknown: loaded from a .cim, so there is no original to compare with"
        if math.isinf(self.psnr_db):
            return "infinite: the reconstruction equals the original"
        return (
            f"{self.psnr_db:.2f} dB (mean squared error {self.mean_squared_error:.2f}, "
            f"largest error {self.max_error} of 255)"
        )


class Vectorizer:
    """A picture held as the coefficient vectors the codec keeps of it.

    Two ways in, and they meet at the vectors. :meth:`parse` reads a picture
    and :meth:`compute` transforms it; :meth:`load` reads a ``.cim``, which
    *is* vectors, so there is nothing left to compute. From there
    :attr:`vectors` is the array, :meth:`describe` measures it,
    :meth:`reconstruct` turns it back into pixels and :meth:`save` writes it
    as a ``.cim``. Every method that does not return something else returns
    the vectorizer, so calls chain.
    """

    def __init__(
        self,
        *,
        transform: Transform | str | None = None,
        y_block_size: int = DEFAULT_Y_BLOCK_SIZE,
        cb_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        cr_block_size: int = DEFAULT_CHROMA_BLOCK_SIZE,
        packed_block_size: int = DEFAULT_PACKED_BLOCK_SIZE,
        coeff_removal: float | None = None,
    ) -> None:
        """Create a vectorizer with nothing in it yet.

        The arguments are those of :class:`~walsh.codec.Codec`, which does the
        arithmetic, and mean what they mean there.

        Args:
            transform: The block transform: a name (``"walsh"``, ``"dct"``,
                ``"haar"``), a :class:`~walsh.transforms.Transform` instance,
                or ``None`` for Walsh-Hadamard. **A ``.cim`` does not record
                the transform that wrote it**, so after :meth:`load` this is
                taken on trust: :meth:`reconstruct` inverts with whatever was
                given here.
            y_block_size: Block edge used for the luma channel.
            cb_block_size: Block edge used for the Cb channel.
            cr_block_size: Block edge used for the Cr channel.
            packed_block_size: Coefficients kept per axis, so the length of a
                vector is its square. After :meth:`load` the geometry is the
                file's, whatever was given here.
            coeff_removal: Magnitude below which coefficients are zeroed by
                :meth:`compute`, or ``None`` to keep them all.

        Raises:
            TypeError: If ``transform`` is none of the three.
            ValueError: If the transform name is unknown, the block geometry
                is one the codec cannot honour, or ``coeff_removal`` is
                negative.
        """
        self._codec = Codec(
            transform=transform,
            y_block_size=y_block_size,
            cb_block_size=cb_block_size,
            cr_block_size=cr_block_size,
            packed_block_size=packed_block_size,
        ).with_coeff_removal(coeff_removal)
        self._transform_name = type(self._codec.transform).__name__
        self._original: PixelArray | None = None
        self._source_bytes: int | None = None
        self._size: tuple[int, int] | None = None
        self._descriptions: dict[str, BlockDescription] = {}
        self._vectors: Vectors | None = None

    # -- ways in ---------------------------------------------------------

    def parse(
        self, file_name: FileSource, *, width: int | None = None, height: int | None = None
    ) -> Vectorizer:
        """Read a picture. :meth:`compute` turns it into vectors.

        Anything held from before, vectors included, is dropped.

        Args:
            file_name: Picture to read, its format taken from the suffix, or
                ``None`` to read a BMP from ``sys.stdin``.
            width: Width of the picture, needed only for a pickled flat list
                of pixels, which does not carry its size. For any other input
                it is checked against the file.
            height: Height of the picture. Both or neither.

        Returns:
            This vectorizer, so calls can be chained.

        Raises:
            ValueError: If ``file_name`` is a ``.cim``, which :meth:`load`
                reads; if only one of ``width`` and ``height`` is given, or
                either is not a positive integer; or if the picture is not the
                size declared.
            UnsupportedFileFormatError: If the suffix is unknown, or the file
                is not valid for its format.
            OSError: If the file cannot be read.
        """
        if file_name is not None and Path(file_name).suffix.lower() == _CONTAINER_SUFFIX:
            raise ValueError(
                f"{str(file_name)!r} is a .cim, which already holds vectors: use load() for it"
            )
        if (width is None) != (height is None):
            raise ValueError("width and height must be declared together")

        image = reader_for(file_name)
        if width is not None and height is not None:
            image.declare_size(width, height)
        image.load(file_name)
        if width is not None and (width, height) != image.get_dimensions():
            found = "x".join(str(edge) for edge in image.get_dimensions())
            raise ValueError(f"{file_name} is {found}, not the {width}x{height} declared")

        self._forget()
        self._original = image.get_array()
        self._source_bytes = None if file_name is None else Path(file_name).stat().st_size
        return self

    def load(self, file_name: FileSource) -> Vectorizer:
        """Read a ``.cim``: the vectors, as some earlier compression left them.

        Anything held from before is dropped, and there is no original
        picture afterwards, so :meth:`describe` has no PSNR to report.

        Args:
            file_name: The ``.cim`` to read, or ``None`` to read it from
                ``sys.stdin``.

        Returns:
            This vectorizer, so calls can be chained.

        Raises:
            UnsupportedFileFormatError: If the file is truncated, or is not a
                ``.cim`` at all.
            OSError: If the file cannot be read.
        """
        container = CustomizableImage.load(file_name)
        self._forget()
        self._take(container)
        return self

    def compute(self) -> Vectorizer:
        """Transform the parsed picture into vectors.

        After :meth:`load` the vectors came with the file, so there is nothing
        to compute and this does nothing. After :meth:`parse` it may be called
        again, which computes afresh and discards any change made to
        :attr:`vectors` since.

        Returns:
            This vectorizer, so calls can be chained.

        Raises:
            ValueError: If nothing has been parsed or loaded.
            UnsupportedFileFormatError: If the picture needs more blocks than
                a ``.cim`` can count.
        """
        if self._original is not None:
            self._take(self._codec.encode(self._original))
        elif self._vectors is None:
            raise ValueError("nothing to compute; call parse() or load() first")
        return self

    # -- the vectors -----------------------------------------------------

    @property
    def vectors(self) -> Vectors:
        """The coefficient vectors, live: see the module docstring.

        ``_vectors`` is the same array, and ``None`` until there is one.

        Returns:
            ``int16`` of shape ``(blocks, packed_block_size ** 2)``. It is the
            object's own state, not a copy, so writing into it changes what is
            described, reconstructed and saved.

        Raises:
            ValueError: If there are no vectors yet, or the array has since
                been replaced by one of another shape or dtype.
        """
        return self._checked()

    def reconstruct(self) -> PixelArray:
        """Turn the vectors back into a picture.

        Returns:
            ``(height, width, 3)`` RGB ``uint8``: what extracting the saved
            ``.cim`` would give, byte for byte.

        Raises:
            ValueError: If there are no vectors yet.
        """
        return self._codec.decode(self._container())

    def describe(self) -> CompressionStats:
        """Measure the compression: sizes, reduction, and PSNR where possible.

        Returns:
            The figures. ``print()`` them for a table.

        Raises:
            ValueError: If there are no vectors yet.
        """
        vectors = self._checked()
        width, height = self._dimensions()
        raw_bytes = width * height * 3
        compressed_bytes = len(self._container().to_bytes())

        psnr = mse = max_error = None
        if self._original is not None:
            difference = self._original.astype(np.int16) - self.reconstruct().astype(np.int16)
            mse = float(np.mean(np.square(difference, dtype=np.float64)))
            max_error = int(np.abs(difference).max())
            psnr = math.inf if mse == 0 else 10 * math.log10(_PEAK**2 / mse)

        blocks = {channel: self._descriptions[channel] for channel in _CHANNELS}
        return CompressionStats(
            width=width,
            height=height,
            transform=self._transform_name,
            y_block_size=blocks["y"].original_block_size,
            cb_block_size=blocks["cb"].original_block_size,
            cr_block_size=blocks["cr"].original_block_size,
            packed_block_size=blocks["y"].packed_block_size,
            y_vectors=blocks["y"].number_of_blocks,
            cb_vectors=blocks["cb"].number_of_blocks,
            cr_vectors=blocks["cr"].number_of_blocks,
            vector_length=vectors.shape[1],
            coefficients=vectors.size,
            nonzero_coefficients=int(np.count_nonzero(vectors)),
            kept_share=100 * vectors.size / raw_bytes,
            raw_bytes=raw_bytes,
            source_bytes=self._source_bytes,
            compressed_bytes=compressed_bytes,
            reduction_percent=100 * (1 - compressed_bytes / raw_bytes),
            source_reduction_percent=(
                None
                if self._source_bytes is None
                else 100 * (1 - compressed_bytes / self._source_bytes)
            ),
            bits_per_pixel=8 * compressed_bytes / (width * height),
            psnr_db=psnr,
            mean_squared_error=mse,
            max_error=max_error,
        )

    def save(self, output_file_name: FileSource) -> Vectorizer:
        """Write the vectors as a ``.cim``.

        Args:
            output_file_name: Path to write, or ``None`` to write to stdout.

        Returns:
            This vectorizer, so calls can be chained.

        Raises:
            ValueError: If there are no vectors yet, or the name ends in a
                picture suffix. What is written is always the container, and
                a ``.cim`` under a picture's name opens in nothing;
                :meth:`reconstruct` gives the picture, and
                :class:`~walsh.codec.Codec` writes one.
            OSError: If the file cannot be written.
        """
        if output_file_name is not None and Path(output_file_name).suffix.lower() in SUFFIXES:
            raise ValueError(
                f"save() writes the .cim container, and {str(output_file_name)!r} names a "
                f"picture; name it .cim, or use Codec().extract() to write the picture"
            )
        self._container().save(output_file_name)
        return self

    # -- helpers ---------------------------------------------------------

    def _forget(self) -> None:
        """Drop everything a previous parse, load or compute left behind."""
        self._original = None
        self._source_bytes = None
        self._size = None
        self._descriptions = {}
        self._vectors = None

    def _take(self, container: CustomizableImage) -> None:
        """Adopt a container's geometry and flatten its blocks into vectors.

        The blocks go through the container's bytes first, so the vectors are
        the ``int16`` values a file would hold, however the container was
        made: fresh from the codec its blocks are still unrounded floats.

        Args:
            container: A container from the codec or from a file.

        Raises:
            UnsupportedFileFormatError: If the channels keep different numbers
                of coefficients per block. The format allows it and nothing
                in this package writes it, but such a file has vectors of two
                lengths, which do not make one array.
        """
        descriptions = container.get_descriptions()
        kept = {d.packed_block_size for d in descriptions.values() if d.number_of_blocks}
        if len(kept) > 1:
            raise UnsupportedFileFormatError(
                f"this .cim keeps {sorted(kept)} coefficients per axis in different channels, "
                f"so its vectors are of different lengths and do not make one array"
            )
        packed = kept.pop() if kept else descriptions["y"].packed_block_size

        data = container.to_bytes()
        stored = np.frombuffer(data, dtype=COEFF_DTYPE, offset=CustomizableImage.HEADER_SIZE)
        self._size = container.get_dimensions()
        self._descriptions = descriptions
        self._vectors = stored.reshape(-1, packed * packed).astype(np.int16, copy=True)

    def _dimensions(self) -> tuple[int, int]:
        """Return the picture's size, which vectors never come without.

        Returns:
            ``(width, height)``.

        Raises:
            ValueError: If there are no vectors yet.
        """
        if self._size is None:
            raise ValueError("there are no vectors yet; call parse() and compute(), or load()")
        return self._size

    def _checked(self) -> Vectors:
        """Return the vectors, insisting they still fit the geometry.

        ``_vectors`` is open to being replaced as well as written into, so its
        shape and dtype are verified wherever it is used.

        Returns:
            The vectors.

        Raises:
            ValueError: If there are none yet, or the array is not ``int16``
                of the shape the block counts and the packed size require.
        """
        self._dimensions()
        vectors = self._vectors
        count = sum(self._descriptions[channel].number_of_blocks for channel in _CHANNELS)
        packed = self._descriptions["y"].packed_block_size
        expected = (count, packed * packed)
        if (
            not isinstance(vectors, np.ndarray)
            or vectors.dtype != np.int16
            or vectors.shape != expected
        ):
            found = (
                f"{vectors.dtype} {vectors.shape}"
                if isinstance(vectors, np.ndarray)
                else type(vectors).__name__
            )
            raise ValueError(f"the vectors must be int16 of shape {expected}, and are {found}")
        return vectors

    def _container(self) -> CustomizableImage:
        """Build the ``.cim`` container the vectors make.

        Returns:
            A container holding the current vectors.

        Raises:
            ValueError: If there are no vectors yet, or they no longer fit.
        """
        vectors = self._checked()
        width, height = self._dimensions()
        container = CustomizableImage()
        container.set_dimensions(width, height)
        container.set_descriptions(*(self._descriptions[channel] for channel in _CHANNELS))

        stacks, start = [], 0
        for channel in _CHANNELS:
            description = self._descriptions[channel]
            packed = description.packed_block_size
            stop = start + description.number_of_blocks
            stacks.append(vectors[start:stop].reshape(-1, packed, packed).astype(np.float64))
            start = stop
        container.set_data(*stacks)
        return container
