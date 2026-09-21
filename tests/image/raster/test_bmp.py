"""The BMP reader and writer: blue-green-red, bottom-up on disk, RGB top-down in memory."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from conftest import gradient_pixels, write_bmp
from walsh.image import (
    BMP_HEADER_FORMAT,
    BMP_PIXEL_OFFSET,
    BMP_SIGNATURE,
    BMPImage,
    UnsupportedFileFormatError,
    align,
)


def test_bmp_roundtrip_is_byte_exact(tmp_path: Path) -> None:
    """Odd width forces row padding, which is the easy thing to get wrong."""
    width, height = 3, 2
    pixels = [(i, i + 1, i + 2) for i in range(width * height)]
    source = write_bmp(tmp_path / "in.bmp", width, height, pixels)

    image = BMPImage()
    image.load(str(source))
    assert image.get_dimensions() == (width, height)
    assert image.get_raw_data() == pixels

    destination = tmp_path / "out.bmp"
    image.save(str(destination))
    assert destination.read_bytes() == source.read_bytes()


def test_bmp_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.bmp"
    header = bytearray(
        struct.pack(
            "<2sIHHIIIIHHIIIIII",
            b"BM",
            100,
            0,
            0,
            54,
            40,
            2,
            2,
            1,
            8,
            0,
            12,
            2835,
            2835,
            0,
            0,
        )
    )
    path.write_bytes(bytes(header) + b"\x00" * 64)

    with pytest.raises(UnsupportedFileFormatError, match="24-bit"):
        BMPImage().load(str(path))


def test_bmp_rows_come_back_top_down(tmp_path: Path) -> None:
    """BMP stores rows bottom-up; the contract says top row first."""
    width, height = 2, 3
    pixels = [(row * 10, row * 10, row * 10) for row in range(height) for _ in range(width)]
    source = write_bmp(tmp_path / "rows.bmp", width, height, pixels)

    image = BMPImage()
    image.load(str(source))
    assert image.get_raw_data() == pixels
    # The first stored row is the bottom one, so the file ends with the top row.
    assert source.read_bytes()[-6:] == bytes([0, 0, 0, 0, 0, 0])


def test_bmp_negative_height_means_top_down(tmp_path: Path) -> None:
    """A negative header height marks a bitmap already stored top-down."""
    width, height = 2, 2
    pixels = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]

    bottom_up = write_bmp(tmp_path / "bu.bmp", width, height, pixels)
    raw = bytearray(bottom_up.read_bytes())

    # Flip the sign of the height field and reverse the stored rows to match.
    raw[22:26] = struct.pack("<i", -height)
    body = raw[54:]
    stride = align(width * 3, 4)
    rows = [bytes(body[i : i + stride]) for i in range(0, len(body), stride)]
    top_down = tmp_path / "td.bmp"
    top_down.write_bytes(bytes(raw[:54]) + b"".join(reversed(rows)))

    image = BMPImage()
    image.load(str(top_down))
    assert image.get_raw_data() == pixels


def test_bmp_rejects_a_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "short.bmp"
    path.write_bytes(b"BM\x00\x00")
    with pytest.raises(UnsupportedFileFormatError, match="truncated BMP header"):
        BMPImage().load(str(path))


def test_bmp_rejects_truncated_pixel_data(tmp_path: Path) -> None:
    source = write_bmp(tmp_path / "full.bmp", 4, 4, gradient_pixels(4, 4))
    truncated = tmp_path / "cut.bmp"
    truncated.write_bytes(source.read_bytes()[:60])
    with pytest.raises(UnsupportedFileFormatError, match="truncated BMP pixel data"):
        BMPImage().load(str(truncated))


@pytest.mark.parametrize(("width", "height"), [(0, 4), (4, 0), (0, 0), (-4, 4)])
def test_bmp_rejects_non_positive_dimensions(tmp_path: Path, width: int, height: int) -> None:
    """Zero is the dangerous one, not the negative one.

    A zero width makes the row stride zero, so the truncation guard in
    `_read_data` compares 0 < 0 and can never fire: the reader loops over the
    whole declared height against a file holding no pixel data, and the codec
    then writes a `.cim` it cannot read back. PPM and PAM already rejected a
    dimension of zero.
    """
    header = struct.pack(
        BMP_HEADER_FORMAT,
        BMP_SIGNATURE,
        54,
        0,
        0,
        BMP_PIXEL_OFFSET,
        40,
        width,
        height,
        1,
        24,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    path = tmp_path / "degenerate.bmp"
    path.write_bytes(header)

    with pytest.raises(UnsupportedFileFormatError, match="dimensions must be positive"):
        BMPImage().load(str(path))


@pytest.mark.parametrize("offset", [0, 20, 53])
def test_bmp_rejects_offset_inside_header(tmp_path: Path, offset: int) -> None:
    """Pixel data cannot legally begin before the file and info headers end."""
    header = struct.pack(
        BMP_HEADER_FORMAT,
        BMP_SIGNATURE,
        100,
        0,
        0,
        offset,
        40,
        2,
        2,
        1,
        24,
        0,
        12,
        2835,
        2835,
        0,
        0,
    )
    path = tmp_path / "bad_offset.bmp"
    path.write_bytes(header + b"\x00" * 64)

    with pytest.raises(UnsupportedFileFormatError, match="pixel offset must be at least 54"):
        BMPImage().load(str(path))
