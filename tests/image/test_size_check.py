"""A reader puts a picture's size to its caller's check before reading a pixel (0.5.8, #81).

``RasterImage.set_size_check`` is how ``Codec`` refuses a picture too large
for the ``.cim`` container for the cost of its header, where it used to read
and decode the whole picture first. These tests hold every reader to the
three things that makes true: the check sees the size the file declares, it
runs before the pixels are read, and it runs after the header is known to be
valid, so a file outside a reader's profile is still refused as such.
"""

from __future__ import annotations

import pickle
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    build_png,
    build_tiff,
    gradient_pixels,
    write_bmp,
    write_npy,
    write_pam,
    write_ppm,
)
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import PickleImage, reader_for

#: The formats that state their size in a header ahead of the pixels.
HEADER_FORMATS = ["bmp", "png", "ppm", "pam", "tiff", "npy"]


class _Refused(Exception):
    """What the test's check raises, so it cannot be mistaken for a reader's error."""


def _refuse(width: int, height: int) -> None:
    raise _Refused(f"{width}x{height}")


def _recorder() -> tuple[list[tuple[int, int]], Callable[[int, int], None]]:
    seen: list[tuple[int, int]] = []
    return seen, lambda width, height: seen.append((width, height))


def _write(kind: str, path: Path, width: int, height: int) -> Path:
    pixels = gradient_pixels(width, height)
    if kind == "bmp":
        return write_bmp(path, width, height, pixels)
    if kind == "png":
        path.write_bytes(build_png(width, height, pixels))
        return path
    if kind == "ppm":
        return write_ppm(path, width, height, pixels)
    if kind == "pam":
        return write_pam(path, width, height, pixels)
    if kind == "tiff":
        path.write_bytes(build_tiff(width, height, pixels))
        return path
    return write_npy(path, width, height, pixels)


def _load(path: Path, check: Callable[[int, int], None] | None = None) -> object:
    image = reader_for(str(path))
    if check is not None:
        image.set_size_check(check)
    image.load(str(path))
    return image


@pytest.mark.parametrize("kind", HEADER_FORMATS)
def test_the_check_sees_the_size_the_header_declares(kind: str, tmp_path: Path) -> None:
    path = _write(kind, tmp_path / f"p.{kind}", 7, 5)
    seen, record = _recorder()
    image = reader_for(str(path))
    image.set_size_check(record)
    image.load(str(path))
    assert seen == [(7, 5)]
    assert image.get_dimensions() == (7, 5)


@pytest.mark.parametrize("kind", HEADER_FORMATS)
def test_the_check_runs_before_a_pixel_is_read(kind: str, tmp_path: Path) -> None:
    """The pixel data is cut short. Without a check the reader reports that;
    with one that refuses, the refusal is what comes back, so the reader had
    not reached the pixels when it asked."""
    whole = _write(kind, tmp_path / f"whole.{kind}", 16, 16).read_bytes()
    path = tmp_path / f"cut.{kind}"
    path.write_bytes(whole[:-100])

    with pytest.raises(UnsupportedFileFormatError):
        _load(path)
    with pytest.raises(_Refused, match=r"^16x16$"):
        _load(path, _refuse)


def _outside_the_profile(kind: str, path: Path) -> Path:
    """A file whose header states its size and then something the reader refuses."""
    pixels = gradient_pixels(4, 4)
    if kind == "bmp":
        data = bytearray(write_bmp(path, 4, 4, pixels).read_bytes())
        data[28:30] = struct.pack("<H", 32)  # bits per pixel, after the size
        path.write_bytes(bytes(data))
        return path
    if kind == "png":
        path.write_bytes(build_png(4, 4, pixels, bit_depth=16))
        return path
    if kind == "ppm":
        path.write_bytes(b"P6\n4 4\n65535\n" + bytes(96))
        return path
    if kind == "pam":
        return write_pam(path, 4, 4, pixels, depth=4, tupltype="RGB_ALPHA")
    if kind == "tiff":
        path.write_bytes(build_tiff(4, 4, pixels, compression=5))
        return path
    return write_npy(path, 4, 4, pixels, dtype="float64")


@pytest.mark.parametrize("kind", HEADER_FORMATS)
def test_a_header_the_reader_refuses_is_reported_before_the_check(
    kind: str, tmp_path: Path
) -> None:
    """The check is only ever put a size from a header the reader accepts."""
    path = _outside_the_profile(kind, tmp_path / f"bad.{kind}")
    seen, record = _recorder()
    with pytest.raises(UnsupportedFileFormatError):
        _load(path, record)
    assert seen == []


def test_no_check_is_the_default_and_none_removes_one(tmp_path: Path) -> None:
    path = _write("ppm", tmp_path / "p.ppm", 3, 2)
    reader_for(str(path)).load(str(path))  # nothing to call

    image = reader_for(str(path))
    image.set_size_check(_refuse)
    image.set_size_check(None)
    image.load(str(path))
    assert image.get_dimensions() == (3, 2)


# --- a pickle states its size in its lists, not a header -----------------------


def _dump(path: Path, value: object) -> Path:
    path.write_bytes(pickle.dumps(value, protocol=4))
    return path


def test_a_pickle_of_rows_is_checked_before_its_samples_are_walked(tmp_path: Path) -> None:
    """Float samples are refused by the walk; a check that refuses is heard first."""
    path = _dump(tmp_path / "rows.pkl", [[(0.5, 0.5, 0.5)] * 6 for _ in range(4)])
    with pytest.raises(UnsupportedFileFormatError, match="samples of type float"):
        PickleImage().load(str(path))
    image = PickleImage()
    image.set_size_check(_refuse)
    with pytest.raises(_Refused, match=r"^6x4$"):
        image.load(str(path))


def test_a_flat_list_is_checked_at_the_size_declared_for_it(tmp_path: Path) -> None:
    path = _dump(tmp_path / "flat.pkl", [(1, 2, 3)] * 20)
    seen, record = _recorder()
    image = PickleImage()
    image.declare_size(5, 4)
    image.set_size_check(record)
    image.load(str(path))
    assert seen == [(5, 4)]


def test_a_pickled_array_is_checked_before_it_is_converted(tmp_path: Path) -> None:
    """A translucent RGBA array is refused in the conversion to RGB, after the check."""
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    path = _dump(tmp_path / "rgba.pkl", rgba)
    with pytest.raises(UnsupportedFileFormatError, match="not fully opaque"):
        PickleImage().load(str(path))
    image = PickleImage()
    image.set_size_check(_refuse)
    with pytest.raises(_Refused, match=r"^5x4$"):
        image.load(str(path))


def test_an_npy_object_array_is_checked_like_the_pickle_it_holds(tmp_path: Path) -> None:
    rows = np.empty(4, dtype=object)
    rows[:] = [[(0.5, 0.5, 0.5)] * 6 for _ in range(4)]
    path = tmp_path / "rows.npy"
    np.save(path, rows, allow_pickle=True)
    with pytest.raises(UnsupportedFileFormatError, match="samples of type float"):
        reader_for(str(path)).load(str(path))
    with pytest.raises(_Refused, match=r"^6x4$"):
        _load(path, _refuse)
