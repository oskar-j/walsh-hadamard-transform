from __future__ import annotations

import struct
from pathlib import Path

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
def sample_bmp() -> Path:
    """The 400x400 sample image checked into the repository."""
    path = DATA_DIR / "image.bmp"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"sample image missing: {path}")
    return path
