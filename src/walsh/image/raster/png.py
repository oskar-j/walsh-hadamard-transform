"""PNG images: 8-bit truecolour, with or without an opaque alpha channel.

Every PNG holds its pixel rows inside a zlib (DEFLATE) stream; there is no
variant without one, and an "uncompressed" PNG is a stream of stored blocks
that still has to be inflated. That is no obstacle here, because the inflater
is :mod:`zlib` in the standard library, so this format keeps the package free
of an image library. The compression is lossless: a PNG hands the codec
exactly the pixels a PPM of the same picture would.

Like :mod:`~walsh.image.raster.tiff`, this reader handles one profile and
rejects the rest by name rather than guessing: eight bits per sample, colour
type 2 (RGB) or 6 (RGBA), not interlaced. RGBA is read **only when every pixel
is opaque**, which is what most everyday PNGs are; real transparency is
refused, because dropping it means choosing a background to flatten onto, and
the same goes for an RGB file whose ``tRNS`` chunk keys out a colour that some
pixel actually has. Palette, greyscale, 16-bit and Adam7 files are refused by
name. Ancillary chunks (``gAMA``, ``sRGB``, ``iCCP``, text, time, ``acTL``)
are skipped once their CRC checks out: samples are taken as they are, as every
other reader here takes them. An unknown *critical* chunk is refused, as the
format requires.

Each row of a PNG starts with a byte naming one of five filters, and encoders
choose per row, so all five have to be undone. Three of them predict a byte
from the pixel to its left, which in turn was predicted from the one before:
read naively that is a Python loop over every byte. It is not done that way
here. A pixel depends only on its left, upper and upper-left neighbours, so
every pixel on one anti-diagonal is independent of the others once the two
diagonals before it are known, and the image is unfiltered one *diagonal* at a
time, each as a single array operation: ``width + height`` steps instead of
``width * height``. See :func:`_undo_by_wavefront`.

Writing produces eight-bit RGB, not interlaced, as ``IHDR``, ``IDAT`` and
``IEND``. The filter is chosen per row by the usual minimum-sum-of-absolute-
differences rule, an array pass over a band of rows at a time, which costs a
few milliseconds on the samples and makes the files 14% smaller there and half
the size on a smooth picture. On the sample it is row for row the choice
libpng made. The filtered bytes are a function of the pixels alone; the bytes
of the file are not, because DEFLATE output may differ from one zlib build to
the next. A PNG written here is therefore pinned by the pixels it decodes to,
never by its bytes.
"""

from __future__ import annotations

import logging
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write, read_up_to
from walsh.image.base import CHANNELS, PixelArray, RasterImage

__all__ = ["PNG_SIGNATURE", "PNGImage"]

log = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

COLOUR_RGB = 2
COLOUR_RGBA = 6
BIT_DEPTH = 8

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVERAGE = 3
FILTER_PAETH = 4

_HEADER_FORMAT = ">IIBBBBB"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

#: Colour type to the bit depths the format allows it, which is how a header
#: that is malformed is told from one that is merely outside this profile.
_ALLOWED_DEPTHS = {
    0: (1, 2, 4, 8, 16),
    COLOUR_RGB: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    COLOUR_RGBA: (8, 16),
}
_HUMAN_COLOUR = {
    0: "greyscale",
    COLOUR_RGB: "RGB",
    3: "palette",
    4: "greyscale with alpha",
    COLOUR_RGBA: "RGBA",
}

#: Samples per pixel for the colour types this profile reads.
_SAMPLES = {COLOUR_RGB: 3, COLOUR_RGBA: 4}

#: Dimensions and chunk lengths are 31-bit by the format's own rule.
_FIELD_LIMIT = 2**31 - 1

#: The most compressed bytes one ``IDAT`` chunk is given on write. A module
#: constant so a test can lower it and see several chunks written and read.
_IDAT_LIMIT = _FIELD_LIMIT

#: The most bytes asked of the inflater in one call. The budget left in an
#: image can exceed what zlib's C interface accepts as a length: a header may
#: declare 2**31 - 1 pixels each way, and did, in the test that found this.
_INFLATE_STEP = 1 << 20

#: Rows unfiltered together are at least this many, and otherwise as many as
#: the image is wide. The wavefront allocates ``(width + rows) * rows`` pixels,
#: so square bands keep that within about twice the band, while the floor keeps
#: a narrow, tall image from being cut into thousands of tiny bands.
_MIN_BAND_ROWS = 256

#: Rows filtered together on write, which bounds the writer's working memory.
_FILTER_BAND_ROWS = 256

_OPAQUE = 255

#: A band of pixel rows, ``(rows, width, samples)``.
_Rows = npt.NDArray[np.uint8]
_Wide = npt.NDArray[np.int16]


@dataclass(frozen=True)
class _Header:
    """What ``IHDR`` declares, once it is known to be within the profile."""

    width: int
    height: int
    samples: int

    @property
    def row_bytes(self) -> int:
        """Bytes in one row of pixels, without its filter byte.

        Returns:
            ``width * samples``.
        """
        return self.width * self.samples

    @property
    def expected(self) -> int:
        """Bytes the inflated pixel stream must hold.

        Returns:
            One filter byte and ``row_bytes`` for each of ``height`` rows.
        """
        return self.height * (1 + self.row_bytes)


def _parse_header(data: bytes) -> _Header:
    """Validate ``IHDR`` and reject any PNG outside the one profile read.

    A header the format itself forbids is reported as invalid; one that is
    legal but unsupported is reported as such, by name.

    Args:
        data: The chunk's payload.

    Returns:
        The dimensions and the samples per pixel.

    Raises:
        UnsupportedFileFormatError: If the header is malformed, or describes a
            palette, greyscale, 16-bit or interlaced image.
    """
    if len(data) != _HEADER_SIZE:
        raise UnsupportedFileFormatError(
            f"corrupt PNG: IHDR holds {len(data)} bytes, not {_HEADER_SIZE}"
        )
    width, height, depth, colour, compression, filtering, interlace = struct.unpack(
        _HEADER_FORMAT, data
    )
    if not (1 <= width <= _FIELD_LIMIT and 1 <= height <= _FIELD_LIMIT):
        raise UnsupportedFileFormatError(f"invalid PNG dimensions {width}x{height}")
    if colour not in _ALLOWED_DEPTHS:
        raise UnsupportedFileFormatError(f"invalid PNG header: unknown colour type {colour}")
    name = _HUMAN_COLOUR[colour]
    if depth not in _ALLOWED_DEPTHS[colour]:
        raise UnsupportedFileFormatError(
            f"invalid PNG header: {depth} bits per sample is not allowed for "
            f"colour type {colour} ({name})"
        )
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise UnsupportedFileFormatError(
            f"invalid PNG header: compression method {compression}, filter method "
            f"{filtering} and interlace method {interlace}; the format defines 0, 0 and 0 or 1"
        )

    if colour not in _SAMPLES:
        raise UnsupportedFileFormatError(
            f"only RGB and RGBA PNG is supported, got colour type {colour} ({name})"
        )
    if depth != BIT_DEPTH:
        raise UnsupportedFileFormatError(
            f"only {BIT_DEPTH} bits per sample is supported, got {depth}"
        )
    if interlace:
        raise UnsupportedFileFormatError(
            "interlaced (Adam7) PNG is not supported; save it without interlacing"
        )
    return _Header(width, height, _SAMPLES[colour])


def _chunks(file: BinaryIO) -> Iterator[tuple[bytes, bytes]]:
    """Yield every chunk up to and including ``IEND``, each with its CRC checked.

    A payload is read through :func:`~walsh.image._io.read_up_to`, so a chunk
    that declares two gigabytes costs only what the file actually holds.

    Args:
        file: Stream positioned just after the signature.

    Yields:
        ``(type, payload)`` for each chunk, in file order.

    Raises:
        UnsupportedFileFormatError: If the file ends before ``IEND``, or a
            chunk is cut short, fails its CRC, or has a type that is not four
            letters.
    """
    while True:
        head = file.read(8)
        if not head:
            raise UnsupportedFileFormatError("truncated PNG: the file ends without an IEND chunk")
        if len(head) < 8:
            raise UnsupportedFileFormatError("truncated PNG: the file ends inside a chunk header")
        length, kind = struct.unpack(">I4s", head)
        if not kind.isalpha():
            raise UnsupportedFileFormatError(
                f"corrupt PNG: chunk type {kind!r} is not four letters"
            )
        name = kind.decode("ascii")
        if length > _FIELD_LIMIT:
            raise UnsupportedFileFormatError(
                f"corrupt PNG: chunk {name} declares {length} bytes, above the format's limit"
            )

        data = read_up_to(file, length)
        checksum = file.read(4)
        if len(data) < length or len(checksum) < 4:
            raise UnsupportedFileFormatError(f"truncated PNG: chunk {name} is cut short")
        if zlib.crc32(data, zlib.crc32(kind)) != struct.unpack(">I", checksum)[0]:
            raise UnsupportedFileFormatError(f"corrupt PNG: chunk {name} fails its CRC")

        yield kind, data
        if kind == b"IEND":
            return


class _PixelStream:
    """The zlib stream spread over the ``IDAT`` chunks, inflated as they arrive.

    Inflation is clamped to the bytes the header declares, in the way the TIFF
    reader clamps its strips: DEFLATE can expand a thousandfold, so a small
    file whose stream runs on past the image would otherwise cost whatever it
    liked. Nothing is allocated from the header either, only from what the
    stream delivers, so a header that declares more than the file holds costs
    nothing before it is reported. A surplus beyond the image is ignored, as
    libpng ignores it.
    """

    def __init__(self, expected: int) -> None:
        """Start an empty stream.

        Args:
            expected: Bytes of filtered rows the header declares.
        """
        self._expected = expected
        self._inflater = zlib.decompressobj()
        self._data = bytearray()
        self.started = False

    def feed(self, data: bytes) -> None:
        """Inflate one ``IDAT`` payload, stopping once the image is complete.

        Args:
            data: The chunk's payload, a slice of the zlib stream.

        Raises:
            UnsupportedFileFormatError: If the data is not a valid stream.
        """
        self.started = True
        try:
            while data and len(self._data) < self._expected:
                budget = min(self._expected - len(self._data), _INFLATE_STEP)
                self._data += self._inflater.decompress(data, budget)
                data = self._inflater.unconsumed_tail
        except zlib.error as error:
            raise UnsupportedFileFormatError(f"corrupt PNG pixel data: {error}") from None

    def finish(self) -> npt.NDArray[np.uint8]:
        """Hand over the filtered rows once every chunk has been fed.

        Returns:
            Exactly the declared number of bytes, writable.

        Raises:
            UnsupportedFileFormatError: If there was no ``IDAT`` chunk, or the
                stream ended before the image did.
        """
        if not self.started:
            raise UnsupportedFileFormatError("PNG has no IDAT chunk, so no pixels")
        if len(self._data) < self._expected:
            raise UnsupportedFileFormatError(
                f"truncated PNG pixel data: expected {self._expected} bytes, got {len(self._data)}"
            )
        return np.frombuffer(self._data, dtype=np.uint8)


def _transparent_colour(data: bytes, header: _Header) -> tuple[int, ...] | None:
    """Read the colour a ``tRNS`` chunk keys out, where that means anything.

    Args:
        data: The chunk's payload.
        header: The image's header.

    Returns:
        The ``(red, green, blue)`` that marks a pixel transparent in an RGB
        image, or ``None`` for RGBA, where the format forbids the chunk and
        the alpha channel decides.

    Raises:
        UnsupportedFileFormatError: If an RGB image's chunk is not three
            16-bit samples.
    """
    if header.samples != CHANNELS:
        return None
    if len(data) != 6:
        raise UnsupportedFileFormatError(
            f"corrupt PNG: tRNS holds {len(data)} bytes, and an RGB image's holds 6"
        )
    return struct.unpack(">3H", data)


def _predictors(left: _Wide, above: _Wide, corner: _Wide) -> tuple[_Wide, ...]:
    """Compute what each of the five filters predicts, for many bytes at once.

    Args:
        left: The byte one pixel to the left of each byte predicted, zero
            where that is outside the image.
        above: The byte directly above, likewise.
        corner: The byte above and to the left, likewise.

    Returns:
        The predictions, indexed by filter type: nothing, the left byte, the
        byte above, the mean of the two rounded down, and Paeth's choice of
        whichever of the three is nearest to ``left + above - corner``, with
        ties going to left, then above.
    """
    to_left = np.abs(above - corner)
    to_above = np.abs(left - corner)
    to_corner = np.abs(left + above - 2 * corner)
    paeth = np.where(
        (to_left <= to_above) & (to_left <= to_corner),
        left,
        np.where(to_above <= to_corner, above, corner),
    )
    return (np.zeros_like(left), left, above, (left + above) >> 1, paeth)


def _undo_by_rows(rows: _Rows, kinds: npt.NDArray[np.uint8], above: _Rows) -> None:
    """Unfilter a band that uses no filter with a sequential dependency.

    ``Sub`` adds the pixel to the left, which is a running sum along the row
    and wraps modulo 256 exactly as ``uint8`` does; ``Up`` adds the row above.

    Args:
        rows: The band's filtered rows, replaced in place by its pixels.
        kinds: Each row's filter type, none of them Average or Paeth.
        above: The pixels of the row before the band.
    """
    sub = kinds == FILTER_SUB
    rows[sub] = np.cumsum(rows[sub], axis=1, dtype=np.uint8)
    for row in np.flatnonzero(kinds == FILTER_UP):
        rows[row] += rows[row - 1] if row else above


def _undo_by_wavefront(rows: _Rows, kinds: npt.NDArray[np.uint8], above: _Rows) -> None:
    """Unfilter a band one anti-diagonal at a time.

    Average and Paeth predict a byte from its left neighbour, which was itself
    predicted, so a row cannot be undone in one operation. But a pixel needs
    only its left, upper and upper-left neighbours, all of which lie on the
    two anti-diagonals before its own. The band is therefore copied into a
    skewed grid in which each anti-diagonal is one contiguous row, and undone
    a diagonal at a time: ``width + rows`` array operations in place of
    ``width * rows`` byte operations.

    With ``above`` as row 0 and the band as rows 1 onwards, pixel ``x`` of row
    ``r`` sits at ``skewed[x + r + 1, r]``. Its left neighbour is then at
    ``[d - 1, r]``, the one above at ``[d - 1, r - 1]`` and the corner at
    ``[d - 2, r - 1]``. Slots left of the image are never written and stay
    zero, which is what the format says a missing neighbour is.

    Args:
        rows: The band's filtered rows, replaced in place by its pixels.
        kinds: Each row's filter type.
        above: The pixels of the row before the band.
    """
    count, width, samples = rows.shape
    skewed = np.zeros((width + count + 1, count + 1, samples), dtype=np.int16)
    skewed[1 : width + 1, 0] = above
    for row in range(count):
        skewed[row + 2 : row + 2 + width, row + 1] = rows[row]

    kind = kinds[:, np.newaxis]
    for diagonal in range(2, width + count + 1):
        first = max(1, diagonal - width)
        last = min(count, diagonal - 1)
        here = skewed[diagonal, first : last + 1]
        predictions = _predictors(
            skewed[diagonal - 1, first : last + 1],
            skewed[diagonal - 1, first - 1 : last],
            skewed[diagonal - 2, first - 1 : last],
        )
        here += np.choose(kind[first - 1 : last], predictions)
        here &= 0xFF

    for row in range(count):
        rows[row] = skewed[row + 2 : row + 2 + width, row + 1]


def _undo_filters(rows: _Rows, kinds: npt.NDArray[np.uint8]) -> None:
    """Turn filtered rows into pixels, in place, a band of rows at a time.

    Args:
        rows: Every filtered row, ``(height, width, samples)``.
        kinds: Each row's filter type, already known to be 0-4.
    """
    height, width, samples = rows.shape
    band = min(height, max(width, _MIN_BAND_ROWS))
    above = np.zeros((width, samples), dtype=np.uint8)
    for start in range(0, height, band):
        chunk = rows[start : start + band]
        chunk_kinds = kinds[start : start + band]
        if bool(np.any(chunk_kinds >= FILTER_AVERAGE)):
            _undo_by_wavefront(chunk, chunk_kinds, above)
        else:
            _undo_by_rows(chunk, chunk_kinds, above)
        above = chunk[-1]


def _opaque_rgb(pixels: _Rows, transparent: tuple[int, ...] | None) -> PixelArray:
    """Reduce the decoded pixels to RGB, refusing real transparency.

    Args:
        pixels: ``(height, width, 3)`` RGB or ``(height, width, 4)`` RGBA.
        transparent: The colour a ``tRNS`` chunk keys out of an RGB image, if
            there was one.

    Returns:
        ``(height, width, 3)`` RGB, C-contiguous.

    Raises:
        UnsupportedFileFormatError: If any pixel is less than fully opaque.
            Dropping transparency means choosing a background to flatten
            onto, and nothing in the file says which.
    """
    total = pixels.shape[0] * pixels.shape[1]
    if pixels.shape[2] != CHANNELS:
        translucent = int(np.count_nonzero(pixels[:, :, CHANNELS] != _OPAQUE))
        if translucent:
            raise UnsupportedFileFormatError(
                f"PNG has real transparency: alpha is below {_OPAQUE} in {translucent} of "
                f"{total} pixels; only RGBA that is opaque throughout is supported"
            )
        return np.ascontiguousarray(pixels[:, :, :CHANNELS])

    if transparent is not None and max(transparent) <= _OPAQUE:
        keyed = int(np.count_nonzero(np.all(pixels == np.array(transparent, np.uint8), axis=2)))
        if keyed:
            raise UnsupportedFileFormatError(
                f"PNG has real transparency: its tRNS chunk makes the colour "
                f"{transparent} transparent and {keyed} of {total} pixels have it"
            )
    return pixels


def _filter_rows(pixels: PixelArray) -> npt.NDArray[np.uint8]:
    """Filter every row for writing, choosing each row's filter type.

    The encoder has every original pixel to hand, so unlike decoding there is
    no sequential dependency: each filter is one pass over many rows at once.
    A row gets the filter whose output has the smallest sum of absolute
    values, reading the bytes as signed, which is the heuristic the PNG
    specification suggests and libpng uses. Ties go to the lower type, so the
    choice is a function of the pixels alone.

    The rows are taken a band at a time only to bound the working memory,
    which is some fifteen times the pixels it covers; a row's filter depends
    on the row above it and on nothing else, so the bands change no byte.

    Args:
        pixels: ``(height, width, 3)`` RGB.

    Returns:
        ``(height, 1 + width * 3)``: each row's filter type, then its bytes.
    """
    height, width, samples = pixels.shape
    out = np.empty((height, 1 + width * samples), dtype=np.uint8)
    for start in range(0, height, _FILTER_BAND_ROWS):
        band = pixels[start : start + _FILTER_BAND_ROWS]
        padded = np.zeros((len(band) + 1, width + 1, samples), dtype=np.int16)
        padded[1:, 1:] = band
        if start:
            padded[0, 1:] = pixels[start - 1]

        target = out[start : start + len(band)]
        lowest = np.full(len(band), np.iinfo(np.int64).max, dtype=np.int64)
        predictions = _predictors(padded[1:, :-1], padded[:-1, 1:], padded[:-1, :-1])
        for kind, prediction in enumerate(predictions):
            filtered = ((padded[1:, 1:] - prediction) & 0xFF).astype(np.uint8)
            filtered = filtered.reshape(len(band), -1)
            cost = np.abs(filtered.view(np.int8).astype(np.int16)).sum(axis=1, dtype=np.int64)
            better = cost < lowest
            target[better, 0] = kind
            target[better, 1:] = filtered[better]
            lowest[better] = cost[better]
    return out


def _chunk(kind: bytes, data: bytes) -> bytes:
    """Frame one chunk: length, type, payload, and the CRC of the last two.

    Args:
        kind: The four-letter chunk type.
        data: The payload.

    Returns:
        The chunk as it appears in the file.
    """
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


class PNGImage(RasterImage):
    """An eight-bit RGB PNG.

    Pixels are stored red-green-blue and top-down, which already matches this
    package's in-memory contract, so nothing is reordered; the work is in the
    container, which is compressed and filtered.
    """

    def load(self, filename: FileSource) -> None:
        """Read a PNG from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``,
                which need not be seekable: a PNG is read front to back.

        Raises:
            UnsupportedFileFormatError: If the file is not a PNG, is damaged,
                is one this profile does not cover, or has real transparency.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            signature = file.read(len(PNG_SIGNATURE))
            if signature != PNG_SIGNATURE:
                raise UnsupportedFileFormatError(
                    f"expected the PNG signature {PNG_SIGNATURE!r}, got {signature!r}"
                )
            chunks = _chunks(file)
            kind, data = next(chunks)
            if kind != b"IHDR":
                raise UnsupportedFileFormatError(
                    f"corrupt PNG: the first chunk is {kind.decode('ascii')}, not IHDR"
                )
            header = _parse_header(data)
            transparent, filtered = self._read_body(chunks, header)

        table = filtered.reshape(header.height, 1 + header.row_bytes)
        kinds = np.ascontiguousarray(table[:, 0])
        if int(kinds.max()) > FILTER_PAETH:
            row = int(np.argmax(kinds > FILTER_PAETH))
            raise UnsupportedFileFormatError(
                f"corrupt PNG: row {row} uses filter type {int(kinds[row])}, and there are five"
            )
        pixels = np.ascontiguousarray(table[:, 1:]).reshape(
            header.height, header.width, header.samples
        )
        _undo_filters(pixels, kinds)

        self.set_array(_opaque_rgb(pixels, transparent))
        log.debug("loaded PNG %dx%d from %s", self._width, self._height, filename)

    @staticmethod
    def _read_body(
        chunks: Iterator[tuple[bytes, bytes]], header: _Header
    ) -> tuple[tuple[int, ...] | None, npt.NDArray[np.uint8]]:
        """Walk the chunks after ``IHDR`` and inflate the pixel stream.

        Args:
            chunks: The rest of the file's chunks.
            header: The image's header.

        Returns:
            The colour a ``tRNS`` chunk keys out, if any, and the filtered
            rows, one filter byte before each.

        Raises:
            UnsupportedFileFormatError: If a chunk this reader cannot ignore
                is unknown, the ``IDAT`` chunks are interrupted, there is a
                second ``IHDR``, or the pixel data is damaged or short.
        """
        stream = _PixelStream(header.expected)
        transparent: tuple[int, ...] | None = None
        interrupted = False
        for kind, data in chunks:
            if kind == b"IDAT":
                if interrupted:
                    raise UnsupportedFileFormatError(
                        "corrupt PNG: its IDAT chunks are not consecutive"
                    )
                stream.feed(data)
                continue
            interrupted = stream.started
            if kind == b"tRNS":
                transparent = _transparent_colour(data, header)
            elif kind == b"IHDR":
                raise UnsupportedFileFormatError("corrupt PNG: a second IHDR chunk")
            elif kind not in (b"PLTE", b"IEND") and not kind[0] & 0x20:
                # An upper-case first letter marks a chunk a decoder may not
                # skip; PLTE is one, but in a truecolour file it only suggests
                # a palette to quantise to, and changes no pixel.
                raise UnsupportedFileFormatError(
                    f"PNG uses a critical chunk this reader does not know: {kind.decode('ascii')}"
                )
        return transparent, stream.finish()

    def save(self, filename: FileSource) -> None:
        """Write this image as an eight-bit RGB PNG, not interlaced.

        The bytes are not reproducible across zlib builds, only the pixels
        are; see the module docstring.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If the pixel count does not match the dimensions, or
                the image is empty, which a PNG cannot be.
            OSError: If the file cannot be written.
        """
        pixels = self.get_array()
        if not pixels.size:
            raise ValueError(
                f"a PNG cannot hold an empty image, and this one is {self._width}x{self._height}"
            )
        packed = zlib.compress(_filter_rows(pixels).tobytes())
        header = struct.pack(
            _HEADER_FORMAT, self._width, self._height, BIT_DEPTH, COLOUR_RGB, 0, 0, 0
        )

        with open_binary_write(filename) as file:
            file.write(PNG_SIGNATURE)
            file.write(_chunk(b"IHDR", header))
            for start in range(0, len(packed), _IDAT_LIMIT):
                file.write(_chunk(b"IDAT", packed[start : start + _IDAT_LIMIT]))
            file.write(_chunk(b"IEND", b""))
