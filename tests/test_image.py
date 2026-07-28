from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import write_bmp
from walsh.image import (
    BlockDescription,
    BMPImage,
    CustomizableImage,
    UnsupportedFileFormatError,
    align,
)


@pytest.mark.parametrize(
    ("value", "alignment", "expected"),
    [(1, 4, 4), (4, 4, 4), (5, 4, 8), (12, 4, 12), (13, 4, 16)],
)
def test_align(value: int, alignment: int, expected: int) -> None:
    assert align(value, alignment) == expected


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


def test_cim_roundtrip_keeps_only_the_packed_corner(tmp_path: Path) -> None:
    original, packed = 8, 4
    block = np.arange(original * original, dtype=np.float64).reshape(original, original)

    image = CustomizableImage()
    image.set_dimensions(8, 8)
    description = BlockDescription(original, packed, 1)
    image.set_descriptions(description, description, description)
    image.set_data([block], [block], [block])

    path = tmp_path / "out.cim"
    image.save(str(path))

    restored = CustomizableImage.load(str(path)).get_y_data()
    assert len(restored) == 1
    # The kept corner survives; everything outside it comes back as zero.
    np.testing.assert_allclose(restored[0][:packed, :packed], block[:packed, :packed])
    np.testing.assert_allclose(restored[0][packed:, :], 0)
    np.testing.assert_allclose(restored[0][:, packed:], 0)


def test_cim_coefficients_are_rounded_not_truncated(tmp_path: Path) -> None:
    description = BlockDescription(2, 2, 1)
    image = CustomizableImage()
    image.set_dimensions(2, 2)
    image.set_descriptions(description, description, description)
    block = np.array([[2.6, -2.6], [0.4, -0.4]])
    image.set_data([block], [block], [block])

    path = tmp_path / "rounded.cim"
    image.save(str(path))

    restored = CustomizableImage.load(str(path)).get_y_data()[0]
    np.testing.assert_array_equal(restored, np.array([[3.0, -3.0], [0.0, -0.0]]))


def test_cim_file_size_matches_the_declared_layout(tmp_path: Path) -> None:
    description = BlockDescription(8, 4, 2)
    image = CustomizableImage()
    image.set_dimensions(16, 8)
    image.set_descriptions(description, description, description)
    blocks = [np.zeros((8, 8)), np.ones((8, 8))]
    image.set_data(blocks, blocks, blocks)

    path = tmp_path / "sized.cim"
    image.save(str(path))

    header = struct.calcsize("<II") + 3 * struct.calcsize("<HHH")
    payload = 3 * 2 * 4 * 4 * 2  # channels * blocks * 4x4 coefficients * int16
    assert path.stat().st_size == header + payload


def test_set_data_before_descriptions_is_an_error() -> None:
    with pytest.raises(ValueError, match="set_descriptions"):
        CustomizableImage().set_data([], [], [])
