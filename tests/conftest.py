from __future__ import annotations

import struct
from pathlib import Path

import pytest

from walsh.image import BMP_HEADER_FORMAT, BMP_PIXEL_OFFSET, BMP_SIGNATURE, align

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def write_bmp(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> Path:
    """Write a minimal 24-bit uncompressed BMP for tests."""
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
        width * height * 3,
        2835,
        2835,
        0,
        0,
    )
    body = bytearray()
    for row in range(height):
        for pixel in pixels[row * width : (row + 1) * width]:
            body += struct.pack("<BBB", *pixel)
        body += padding
    path.write_bytes(header + bytes(body))
    return path


@pytest.fixture
def gradient_bmp(tmp_path: Path) -> Path:
    """A 16x16 BMP with a smooth gradient, so the low-frequency codec does well."""
    width = height = 16
    pixels = [
        (x * 16 % 256, y * 16 % 256, (x + y) * 8 % 256) for y in range(height) for x in range(width)
    ]
    return write_bmp(tmp_path / "gradient.bmp", width, height, pixels)


@pytest.fixture
def sample_bmp() -> Path:
    """The 400x400 sample image checked into the repository."""
    path = DATA_DIR / "image.bmp"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"sample image missing: {path}")
    return path
