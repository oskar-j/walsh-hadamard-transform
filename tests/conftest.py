from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walsh.image import BMP_HEADER_FORMAT, BMP_PIXEL_OFFSET, BMP_SIGNATURE, align

Pixel = tuple[int, int, int]

#: What the `sample` fixture hands out: a checked-in file's name to its path.
Sample = Callable[[str], Path]


def write_bmp(path: Path, width: int, height: int, pixels: list[Pixel]) -> Path:
    """Write a minimal 24-bit BMP from RGB pixels given top row first.

    The file itself gets BMP's own layout -- blue-green-red samples, rows
    bottom-up -- so this is the inverse of what `BMPImage.load` does.
    """
    stride = align(width * 3, 4)
    padding = b"\x00" * (stride - width * 3)
    header = struct.pack(
        BMP_HEADER_FORMAT,
        BMP_SIGNATURE,
        BMP_PIXEL_OFFSET + stride * height,
        0,
        0,
        BMP_PIXEL_OFFSET,
        40,
        width,
        height,
        1,
        24,
        0,
        stride * height,
        2835,
        2835,
        0,
        0,
    )
    body = bytearray()
    for row in reversed(range(height)):
        for r, g, b in pixels[row * width : (row + 1) * width]:
            body += struct.pack("<BBB", b, g, r)
        body += padding
    path.write_bytes(header + bytes(body))
    return path


def write_ppm(
    path: Path,
    width: int,
    height: int,
    pixels: list[Pixel],
    *,
    maxval: int = 255,
    ascii_form: bool = False,
) -> Path:
    """Write a PPM from RGB pixels given top row first.

    PPM's on-disk layout already matches the in-memory contract, so no
    conversion is needed. Set `ascii_form` for the P3 variant.
    """
    if ascii_form:
        header = f"P3\n{width} {height}\n{maxval}\n".encode()
        body = " ".join(str(c) for pixel in pixels for c in pixel).encode() + b"\n"
    else:
        header = f"P6\n{width} {height}\n{maxval}\n".encode()
        body = bytes(channel for pixel in pixels for channel in pixel)
    path.write_bytes(header + body)
    return path


def build_tiff(
    width: int,
    height: int,
    pixels: list[Pixel],
    *,
    order: bytes = b"II",
    compression: int = 1,
    photometric: int = 2,
    bits: tuple[int, int, int] = (8, 8, 8),
    samples: int = 3,
    planar: int = 1,
    orientation: int = 1,
    rows_per_strip: int | None = None,
    omit: set[int] | None = None,
    extra_entries: list[tuple[int, int, int, int]] | None = None,
) -> bytes:
    """Build an uncompressed TIFF, with hooks for the unsupported profiles.

    The keyword arguments exist so tests can produce files this reader must
    reject -- compressed, palette, 16-bit, planar, rotated -- which Pillow will
    not emit on request. ``extra_entries`` appends ``(tag, type, count,
    value)`` directory entries verbatim, the value being the four-byte
    value-or-offset field; a type this reader keeps opaque may point anywhere
    in the file, since its bytes are read and discarded.
    """
    prefix = ">" if order == b"MM" else "<"
    omit = omit or set()
    body = bytes(channel for pixel in pixels for channel in pixel)
    rows = rows_per_strip or height
    row_bytes = width * 3

    strips = [
        body[i * rows * row_bytes : (i + 1) * rows * row_bytes]
        for i in range((height + rows - 1) // rows)
    ]
    strips = [s for s in strips if s]

    entries: list[tuple[int, int, int, object]] = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 3, "bits"),
        (259, 3, 1, compression),
        (262, 3, 1, photometric),
        (273, 4, len(strips), "offsets"),
        (274, 3, 1, orientation),
        (277, 3, 1, samples),
        (278, 4, 1, rows),
        (279, 4, len(strips), "counts"),
        (284, 3, 1, planar),
    ]
    entries = [e for e in entries if e[0] not in omit]
    entries += [(tag, kind, count, value) for tag, kind, count, value in extra_entries or []]

    directory_size = 2 + len(entries) * 12 + 4
    cursor = 8 + directory_size
    extra = bytearray()

    def place(values: list[int], code: str, size: int) -> int:
        nonlocal cursor
        if len(values) * size <= 4:
            return values[0] if len(values) == 1 else 0
        offset = cursor
        extra.extend(struct.pack(f"{prefix}{len(values)}{code}", *values))
        cursor += len(values) * size
        return offset

    bits_value = place(list(bits), "H", 2)
    data_start = cursor + sum(len(s) for s in ())  # placeholder; offsets computed below
    # Strip offsets need the final data position, so reserve their slot first.
    offsets_need_table = len(strips) * 4 > 4
    counts_need_table = len(strips) * 4 > 4
    reserved = (len(strips) * 4 if offsets_need_table else 0) + (
        len(strips) * 4 if counts_need_table else 0
    )
    data_start = cursor + reserved

    positions, position = [], data_start
    for strip in strips:
        positions.append(position)
        position += len(strip)

    offsets_value = place(positions, "I", 4)
    counts_value = place([len(s) for s in strips], "I", 4)

    resolved = {"bits": bits_value, "offsets": offsets_value, "counts": counts_value}

    out = bytearray(struct.pack(f"{prefix}2sHI", order, 42, 8))
    out += struct.pack(f"{prefix}H", len(entries))
    for tag, field_type, count, value in entries:
        real = resolved[value] if isinstance(value, str) else value
        out += struct.pack(f"{prefix}HHI", tag, field_type, count)
        inline_short = field_type == 3 and count == 1
        out += struct.pack(f"{prefix}H2x" if inline_short else f"{prefix}I", real)
    out += struct.pack(f"{prefix}I", 0)
    out += extra
    for strip in strips:
        out += strip
    return bytes(out)


def build_tiff_with_strip_table(
    width: int,
    height: int,
    offsets: list[int],
    counts: list[int],
    payload: bytes,
) -> bytes:
    """Build a TIFF whose strip tables are given verbatim, however wrong.

    `build_tiff` derives consistent strip tables from the pixels. This one
    takes them as they are, so a test can describe strips that overlap, repeat
    one region, or run past the pixels the header declares -- geometry no
    encoder produces and `build_tiff` cannot express.
    """
    tags = [
        (256, 4, 1),
        (257, 4, 1),
        (258, 3, 3),
        (259, 3, 1),
        (262, 3, 1),
        (273, 4, len(offsets)),
        (274, 3, 1),
        (277, 3, 1),
        (278, 4, 1),
        (279, 4, len(counts)),
        (284, 3, 1),
    ]
    heap = 8 + 2 + 12 * len(tags) + 4
    bits_at = heap
    offs_at = bits_at + 6
    cnts_at = offs_at + 4 * len(offsets)
    data_at = cnts_at + 4 * len(counts)

    inline = {256: width, 257: height, 259: 1, 262: 2, 274: 1, 277: 3, 278: height, 284: 1}
    table = {258: bits_at, 273: offs_at, 279: cnts_at}

    out = bytearray(struct.pack("<2sHI", b"II", 42, 8) + struct.pack("<H", len(tags)))
    for tag, field_type, count in tags:
        value = table[tag] if tag in table else inline[tag]
        out += struct.pack("<HHI", tag, field_type, count)
        out += struct.pack("<H2x" if (field_type == 3 and count == 1) else "<I", value)
    out += struct.pack("<I", 0)
    out += struct.pack("<3H", 8, 8, 8)
    out += struct.pack(f"<{len(offsets)}I", *(data_at + o for o in offsets))
    out += struct.pack(f"<{len(counts)}I", *counts)
    out += payload
    return bytes(out)


def write_tiff(path: Path, width: int, height: int, pixels: list[Pixel], **kwargs: Any) -> Path:
    """Write an uncompressed TIFF from RGB pixels given top row first."""
    path.write_bytes(build_tiff(width, height, pixels, **kwargs))
    return path


def write_pam(
    path: Path,
    width: int,
    height: int,
    pixels: list[Pixel],
    *,
    maxval: int = 255,
    depth: int = 3,
    tupltype: str | None = "RGB",
    header_lines: list[str] | None = None,
) -> Path:
    """Write a PAM from RGB pixels given top row first.

    With the defaults the header is byte-identical to what Netpbm's own
    `pamtopam` writes. `depth` and `tupltype` change only what the header
    claims, not the samples written, which is what the profile-rejection tests
    need. `header_lines` replaces the generated field lines entirely, for
    malformed or unusually laid out headers.
    """
    if header_lines is None:
        header_lines = [f"WIDTH {width}", f"HEIGHT {height}", f"DEPTH {depth}", f"MAXVAL {maxval}"]
        if tupltype is not None:
            header_lines.append(f"TUPLTYPE {tupltype}")
    header = "P7\n" + "".join(line + "\n" for line in header_lines) + "ENDHDR\n"
    body = bytes(channel for pixel in pixels for channel in pixel)
    path.write_bytes(header.encode() + body)
    return path


def write_npy(
    path: Path,
    width: int,
    height: int,
    pixels: list[Pixel],
    *,
    channels: int | None = 3,
    dtype: str = "uint8",
    alpha: int = 255,
    fortran_order: bool = False,
) -> Path:
    """Write a `.npy` from RGB pixels given top row first.

    With the defaults this is exactly what `np.save` writes for an
    `(height, width, 3)` `uint8` array, so a file this package writes should
    match it byte for byte. The keywords produce the off-profile shapes the
    reader must handle or refuse: `channels=None` for a 2-D greyscale array,
    `1` for `(h, w, 1)`, `4` to append an `alpha` plane, any `dtype` numpy
    knows, and Fortran memory order. Greyscale takes the red channel.
    """
    rgb = np.asarray(pixels, dtype=np.uint8).reshape(height, width, 3)
    if channels is None:
        array = rgb[:, :, 0]
    elif channels == 1:
        array = rgb[:, :, :1]
    elif channels == 4:
        array = np.concatenate([rgb, np.full((height, width, 1), alpha, dtype=np.uint8)], axis=2)
    else:
        array = (
            rgb[:, :, :channels]
            if channels <= 3
            else np.repeat(rgb, channels, axis=2)[:, :, :channels]
        )
    array = array.astype(dtype)
    if fortran_order:
        array = np.asfortranarray(array)
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    return path


def png_predict(kind: int, left: int, above: int, corner: int) -> int:
    """What PNG filter ``kind`` predicts for one byte, straight from the spec.

    Kept byte-at-a-time and free of NumPy on purpose: the reader under test
    undoes filters a whole anti-diagonal at a time, and the writer filters a
    whole image at once, so this is the independent statement of the rule that
    both are checked against.
    """
    if kind > 4:
        return 0  # not a filter: a test asked for a type the reader must refuse
    if kind == 4:
        estimate = left + above - corner
        distances = (abs(estimate - left), abs(estimate - above), abs(estimate - corner))
        # Ties go to left, then above: min() keeps the first of equal keys.
        return min(zip(distances, (0, 1, 2), (left, above, corner), strict=True))[2]
    return (0, left, above, (left + above) >> 1)[kind]


def png_filter_rows(rows: Sequence[bytes], kinds: Sequence[int], samples: int) -> bytes:
    """Filter pixel rows one byte at a time, each row led by its filter type."""
    out = bytearray()
    prior = bytes(len(rows[0]))
    for row, kind in zip(rows, kinds, strict=True):
        out.append(kind)
        for index, value in enumerate(row):
            left = row[index - samples] if index >= samples else 0
            corner = prior[index - samples] if index >= samples else 0
            out.append((value - png_predict(kind, left, prior[index], corner)) & 0xFF)
        prior = row
    return bytes(out)


def png_unfilter_rows(filtered: bytes, height: int, samples: int) -> list[bytes]:
    """Undo :func:`png_filter_rows`, again one byte at a time."""
    stride = len(filtered) // height
    rows: list[bytes] = []
    prior = bytes(stride - 1)
    for start in range(0, len(filtered), stride):
        kind, body = filtered[start], filtered[start + 1 : start + stride]
        row = bytearray()
        for index, value in enumerate(body):
            left = row[index - samples] if index >= samples else 0
            corner = prior[index - samples] if index >= samples else 0
            row.append((value + png_predict(kind, left, prior[index], corner)) & 0xFF)
        prior = bytes(row)
        rows.append(prior)
    return rows


def png_chunk(kind: bytes, data: bytes = b"", *, crc: int | None = None) -> bytes:
    """Frame one PNG chunk. ``crc`` overrides the checksum, to write a bad one."""
    checksum = zlib.crc32(kind + data) if crc is None else crc
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def build_png(
    width: int,
    height: int,
    pixels: list[Pixel],
    *,
    filters: int | Sequence[int] = 0,
    alpha: int | Sequence[int] | None = None,
    colour_type: int | None = None,
    bit_depth: int = 8,
    compression: int = 0,
    filter_method: int = 0,
    interlace: int = 0,
    before_idat: Sequence[bytes] = (),
    after_idat: Sequence[bytes] = (),
    idat_size: int | None = None,
    between_idat: bytes = b"",
    surplus: bytes = b"",
    end: bool = True,
) -> bytes:
    """Build a PNG, with hooks for the files this reader must refuse.

    ``filters`` is one filter type for every row or a sequence cycled over the
    rows. ``alpha`` makes the file RGBA, with one value throughout or one per
    pixel. The header fields can be overridden on their own, which makes a
    header that lies about a body built as eight-bit truecolour: enough for a
    reader that refuses at the header. ``before_idat`` and ``after_idat`` take
    ready-made chunks, ``idat_size`` splits the stream over several ``IDAT``
    chunks with ``between_idat`` after the first, ``surplus`` is appended to
    the filtered rows before compression, and ``end=False`` omits ``IEND``.
    """
    samples = 3 if alpha is None else 4
    if alpha is None:
        body = [bytes(channel for channel in pixel) for pixel in pixels]
    else:
        alphas = [alpha] * len(pixels) if isinstance(alpha, int) else list(alpha)
        body = [bytes((*pixel, a)) for pixel, a in zip(pixels, alphas, strict=True)]
    rows = [b"".join(body[y * width : (y + 1) * width]) for y in range(height)]
    cycle = [filters] if isinstance(filters, int) else list(filters)
    kinds = [cycle[y % len(cycle)] for y in range(height)]

    packed = zlib.compress(png_filter_rows(rows, kinds, samples) + surplus)
    size = idat_size or max(len(packed), 1)
    pieces = [packed[start : start + size] for start in range(0, len(packed), size)]

    header = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        (2 if alpha is None else 6) if colour_type is None else colour_type,
        compression,
        filter_method,
        interlace,
    )
    out = bytearray(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header))
    for chunk in before_idat:
        out += chunk
    for index, piece in enumerate(pieces):
        out += png_chunk(b"IDAT", piece)
        if index == 0:
            out += between_idat
    for chunk in after_idat:
        out += chunk
    if end:
        out += png_chunk(b"IEND")
    return bytes(out)


def write_png(path: Path, width: int, height: int, pixels: list[Pixel], **kwargs: Any) -> Path:
    """Write a PNG built by :func:`build_png`. See it for the keyword arguments."""
    path.write_bytes(build_png(width, height, pixels, **kwargs))
    return path


def gradient_pixels(width: int, height: int) -> list[Pixel]:
    """A smooth gradient, which the low-frequency codec reproduces well."""
    return [
        (x * 16 % 256, y * 16 % 256, (x + y) * 8 % 256) for y in range(height) for x in range(width)
    ]


@pytest.fixture
def gradient_bmp(tmp_path: Path) -> Path:
    """A 16x16 BMP with a smooth gradient."""
    width = height = 16
    return write_bmp(tmp_path / "gradient.bmp", width, height, gradient_pixels(width, height))


@pytest.fixture
def gradient_ppm(tmp_path: Path) -> Path:
    """The same gradient as `gradient_bmp`, as a binary P6 pixmap."""
    width = height = 16
    return write_ppm(tmp_path / "gradient.ppm", width, height, gradient_pixels(width, height))


@pytest.fixture
def gradient_tiff(tmp_path: Path) -> Path:
    """The same gradient as `gradient_bmp`, as an uncompressed TIFF."""
    width = height = 16
    return write_tiff(tmp_path / "gradient.tif", width, height, gradient_pixels(width, height))


@pytest.fixture
def gradient_pam(tmp_path: Path) -> Path:
    """The same gradient as `gradient_bmp`, as an RGB PAM."""
    width = height = 16
    return write_pam(tmp_path / "gradient.pam", width, height, gradient_pixels(width, height))


@pytest.fixture
def gradient_png(tmp_path: Path) -> Path:
    """The same gradient as `gradient_bmp`, as an RGB PNG with every filter in turn."""
    width = height = 16
    return write_png(
        tmp_path / "gradient.png", width, height, gradient_pixels(width, height), filters=range(5)
    )


@pytest.fixture
def gradient_npy(tmp_path: Path) -> Path:
    """The same gradient as `gradient_bmp`, as a bare NumPy array."""
    width = height = 16
    return write_npy(tmp_path / "gradient.npy", width, height, gradient_pixels(width, height))


@pytest.fixture(scope="session")
def root(pytestconfig: pytest.Config) -> Path:
    """The repository root, which is pytest's own rootdir.

    pytest anchors its rootdir at the ``pyproject.toml`` whose
    ``[tool.pytest.ini_options]`` table configured the run, from any working
    directory and for a test at any depth, so nothing here counts folders up
    from ``__file__``. The check turns a rootdir that landed somewhere else
    (a ``--rootdir`` or ``-c`` override, or that table going missing) into an
    error: without it every test that needs a sample would skip instead.
    """
    path = pytestconfig.rootpath
    assert (path / "pyproject.toml").is_file(), f"pytest's rootdir {path} is not the repository"
    return path


@pytest.fixture(scope="session")
def sample(root: Path) -> Sample:
    """Look up a file checked into ``data/`` by name, skipping if it is absent.

    The samples sit in a folder per file type, named after the suffix, so
    ``sample("earth.ppm")`` is ``data/ppm/earth.ppm`` and no test spells a
    folder. They are excluded from the sdist, so a test run against an unpacked
    distribution has to cope with them being missing.
    """

    def lookup(name: str) -> Path:
        path = root / "data" / Path(name).suffix.lstrip(".") / name
        if not path.exists():  # pragma: no cover - the sdist ships no samples
            pytest.skip(f"sample image missing: {path}")
        return path

    return lookup


@pytest.fixture
def sample_npy(sample: Sample) -> Path:
    """The 400x400 sample as a `.npy`, written by numpy from Pillow's array."""
    return sample("earth.npy")


@pytest.fixture
def sample_png(sample: Sample) -> Path:
    """The 400x400 sample as a PNG, written by Netpbm's `pnmtopng` (libpng)."""
    return sample("earth.png")


@pytest.fixture
def sample_pam(sample: Sample) -> Path:
    """The 400x400 sample pixmap checked in as a PAM."""
    return sample("earth.pam")


@pytest.fixture
def sample_tiff(sample: Sample) -> Path:
    """The 400x400 sample pixmap checked in as an uncompressed TIFF."""
    return sample("earth.tiff")


@pytest.fixture
def sample_bmp(sample: Sample) -> Path:
    """The 400x400 sample bitmap checked into the repository."""
    return sample("image.bmp")


@pytest.fixture
def sample_ppm(sample: Sample) -> Path:
    """The 400x400 sample pixmap checked into the repository."""
    return sample("earth.ppm")
