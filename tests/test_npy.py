from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import gradient_pixels, write_npy, write_ppm
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import NPYImage, PPMImage, reader_for


def test_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    pixels = gradient_pixels(4, 3)
    source = write_npy(tmp_path / "in.npy", 4, 3, pixels)

    image = NPYImage()
    image.load(str(source))
    assert image.get_dimensions() == (4, 3)
    assert image.get_raw_data() == pixels

    destination = tmp_path / "out.npy"
    image.save(str(destination))
    assert destination.read_bytes() == source.read_bytes()


def test_numpy_reads_back_exactly_what_was_written(tmp_path: Path) -> None:
    """numpy itself is the reference implementation: a file written here must
    load with np.load as the (height, width, 3) uint8 array it came from."""
    pixels = gradient_pixels(5, 2)
    image = NPYImage()
    image.set_dimensions(5, 2)
    image.set_raw_data(pixels)
    path = tmp_path / "ours.npy"
    image.save(str(path))

    array = np.load(path, allow_pickle=False)
    assert array.shape == (2, 5, 3)
    assert array.dtype == np.uint8
    assert array.flags.c_contiguous
    assert array.reshape(-1, 3).tolist() == [list(p) for p in pixels]


@pytest.mark.parametrize("channels", [None, 1], ids=["(h, w)", "(h, w, 1)"])
def test_greyscale_is_replicated_across_the_channels(tmp_path: Path, channels: int | None) -> None:
    pixels = [(10, 0, 0), (200, 0, 0), (0, 0, 0), (255, 0, 0)]
    path = write_npy(tmp_path / "grey.npy", 2, 2, pixels, channels=channels)

    image = NPYImage()
    image.load(str(path))
    assert image.get_dimensions() == (2, 2)
    assert image.get_raw_data() == [(10, 10, 10), (200, 200, 200), (0, 0, 0), (255, 255, 255)]


def test_fully_opaque_rgba_drops_the_alpha_channel(tmp_path: Path) -> None:
    pixels = gradient_pixels(3, 3)
    path = write_npy(tmp_path / "rgba.npy", 3, 3, pixels, channels=4, alpha=255)

    image = NPYImage()
    image.load(str(path))
    assert image.get_raw_data() == pixels


@pytest.mark.parametrize("alpha", [0, 128, 254])
def test_transparency_is_refused_rather_than_flattened(tmp_path: Path, alpha: int) -> None:
    """Dropping a non-opaque alpha means inventing a background, and a
    four-channel array might equally be CMYK; the message names both."""
    path = write_npy(tmp_path / "trans.npy", 3, 3, gradient_pixels(3, 3), channels=4, alpha=alpha)
    with pytest.raises(UnsupportedFileFormatError, match=r"not fully opaque.*CMYK"):
        NPYImage().load(str(path))


def test_fortran_ordered_arrays_decode_correctly(tmp_path: Path) -> None:
    """np.save records the memory order; a column-major body must be reshaped as such."""
    pixels = gradient_pixels(5, 3)
    path = write_npy(tmp_path / "f.npy", 5, 3, pixels, fortran_order=True)
    assert b"'fortran_order': True" in path.read_bytes()[:128]

    image = NPYImage()
    image.load(str(path))
    assert image.get_raw_data() == pixels


@pytest.mark.parametrize(
    ("dtype", "match"),
    [
        ("float32", "dtype float32"),
        ("float64", "dtype float64"),
        ("uint16", "dtype uint16"),
        ("int8", "dtype int8"),
        ("bool", "dtype bool"),
    ],
)
def test_other_dtypes_are_rejected_by_name(tmp_path: Path, dtype: str, match: str) -> None:
    """A float array could be scaled 0-1 or 0-255; guessing is how silent
    corruption starts, so only uint8 is accepted."""
    path = write_npy(tmp_path / "typed.npy", 2, 2, gradient_pixels(2, 2), dtype=dtype)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        NPYImage().load(str(path))


@pytest.mark.parametrize(("channels", "match"), [(2, "2 channels"), (5, "5 channels")])
def test_other_channel_counts_are_rejected_by_name(
    tmp_path: Path, channels: int, match: str
) -> None:
    path = write_npy(tmp_path / "chan.npy", 2, 2, gradient_pixels(2, 2), channels=channels)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        NPYImage().load(str(path))


def test_a_pickled_object_array_is_refused_from_its_header_and_never_loaded(
    tmp_path: Path,
) -> None:
    """np.load with pickling enabled executes arbitrary code. The reader must
    never enable it, and refuses the file from the header's dtype alone."""
    path = tmp_path / "pickled.npy"
    with path.open("wb") as file:
        np.save(file, np.array([{"a": 1}, None], dtype=object), allow_pickle=True)

    with pytest.raises(UnsupportedFileFormatError, match="dtype object"):
        NPYImage().load(str(path))


def test_a_header_declaring_a_huge_array_costs_nothing_to_refuse(tmp_path: Path) -> None:
    """A 144-byte file whose header declares 12.9 GB. The old habit of
    file.read(declared) would allocate that before reading a byte."""
    import io

    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "|u1", "fortran_order": False, "shape": (65535, 65535, 3)}
    )
    path = tmp_path / "bomb.npy"
    path.write_bytes(header.getvalue() + b"\x00" * 16)

    with pytest.raises(
        UnsupportedFileFormatError, match=r"truncated .npy data: expected 12884508675 bytes, got 16"
    ):
        NPYImage().load(str(path))


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("empty", b"", "not a .npy file"),
        ("wrong magic", b"P6\n2 2\n255\n" + bytes(12), "not a .npy file"),
        ("magic only", b"\x93NUMPY\x01\x00", "invalid .npy header"),
        (
            "format version 3.0",
            b"\x93NUMPY\x03\x00" + bytes(64),
            "unsupported .npy format version 3.0",
        ),
    ],
)
def test_malformed_npy_is_rejected(tmp_path: Path, name: str, content: bytes, match: str) -> None:
    path = tmp_path / "bad.npy"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        NPYImage().load(str(path))


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((12,), "shape \\(12,\\)"),
        ((2, 2, 3, 1), "shape \\(2, 2, 3, 1\\)"),
        ((0, 4, 3), "dimensions must be positive"),
        ((4, 0, 3), "dimensions must be positive"),
    ],
)
def test_image_shapes_only(tmp_path: Path, shape: tuple[int, ...], match: str) -> None:
    path = tmp_path / "shape.npy"
    with path.open("wb") as file:
        np.save(file, np.zeros(shape, dtype=np.uint8), allow_pickle=False)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        NPYImage().load(str(path))


def test_truncated_body_is_reported(tmp_path: Path) -> None:
    path = write_npy(tmp_path / "cut.npy", 4, 4, gradient_pixels(4, 4))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(
        UnsupportedFileFormatError, match=r"truncated \.npy data: expected 48 bytes, got 38"
    ):
        NPYImage().load(str(path))


@pytest.mark.parametrize("name", ["x.npy", "x.NPY"])
def test_reader_for_dispatches_npy(name: str) -> None:
    assert isinstance(reader_for(name), NPYImage)


def test_npy_and_ppm_agree_on_the_same_picture(tmp_path: Path) -> None:
    """The array is the raster contract with a header in front of it."""
    pixels = gradient_pixels(8, 5)
    npy = write_npy(tmp_path / "same.npy", 8, 5, pixels)
    ppm = write_ppm(tmp_path / "same.ppm", 8, 5, pixels)

    from_npy = NPYImage()
    from_npy.load(str(npy))
    from_ppm = PPMImage()
    from_ppm.load(str(ppm))
    assert from_npy.get_raw_data() == from_ppm.get_raw_data() == pixels
    assert from_npy.get_dimensions() == from_ppm.get_dimensions()


def test_the_checked_in_sample_was_written_by_numpy_not_by_this_package(sample_npy: Path) -> None:
    """data/earth.npy came from np.save of Pillow's array, so loading it is a
    foreign-writer check; and this package's writer must reproduce it exactly."""
    magic, major, minor = struct.unpack("<6sBB", sample_npy.read_bytes()[:8])
    assert (magic, major, minor) == (b"\x93NUMPY", 1, 0)

    image = NPYImage()
    image.load(str(sample_npy))
    assert image.get_dimensions() == (400, 400)
    assert sample_npy.stat().st_size == 128 + 400 * 400 * 3
