from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from walsh.image import BMP_HEADER_FORMAT, BMP_PIXEL_OFFSET, BMP_SIGNATURE, align

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Pixel = tuple[int, int, int]


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
) -> bytes:
    """Build an uncompressed TIFF, with hooks for the unsupported profiles.

    The keyword arguments exist so tests can produce files this reader must
    reject -- compressed, palette, 16-bit, planar, rotated -- which Pillow will
    not emit on request.
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


def write_tiff(path: Path, width: int, height: int, pixels: list[Pixel], **kwargs: Any) -> Path:
    """Write an uncompressed TIFF from RGB pixels given top row first."""
    path.write_bytes(build_tiff(width, height, pixels, **kwargs))
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
def sample_tiff() -> Path:
    """The 400x400 sample pixmap checked in as an uncompressed TIFF."""
    return _sample("earth.tiff")


@pytest.fixture
def sample_bmp() -> Path:
    """The 400x400 sample bitmap checked into the repository."""
    return _sample("image.bmp")


@pytest.fixture
def sample_ppm() -> Path:
    """The 400x400 sample pixmap checked into the repository."""
    return _sample("earth.ppm")


def _sample(name: str) -> Path:
    """Return a checked-in sample image, skipping if it is not present.

    The samples are excluded from the sdist, so a test run against an unpacked
    distribution has to cope with them being missing.
    """
    path = DATA_DIR / name
    if not path.exists():  # pragma: no cover
        pytest.skip(f"sample image missing: {path}")
    return path
