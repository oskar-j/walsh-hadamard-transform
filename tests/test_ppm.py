from __future__ import annotations

from pathlib import Path

import pytest

from conftest import gradient_pixels, write_bmp, write_ppm
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import BMPImage, PPMImage, reader_for


def test_p6_roundtrip(tmp_path: Path) -> None:
    pixels = gradient_pixels(4, 3)
    source = write_ppm(tmp_path / "in.ppm", 4, 3, pixels)

    image = PPMImage()
    image.load(str(source))
    assert image.get_dimensions() == (4, 3)
    assert image.get_raw_data() == pixels

    destination = tmp_path / "out.ppm"
    image.save(str(destination))
    assert destination.read_bytes() == source.read_bytes()


def test_p3_ascii_is_readable(tmp_path: Path) -> None:
    pixels = gradient_pixels(4, 3)
    source = write_ppm(tmp_path / "in.ppm", 4, 3, pixels, ascii_form=True)

    image = PPMImage()
    image.load(str(source))
    assert image.get_raw_data() == pixels


def test_p3_is_written_back_as_p6(tmp_path: Path) -> None:
    """P3 and P6 carry the same information; P6 is far smaller."""
    source = write_ppm(tmp_path / "in.ppm", 4, 3, gradient_pixels(4, 3), ascii_form=True)
    image = PPMImage()
    image.load(str(source))

    destination = tmp_path / "out.ppm"
    image.save(str(destination))
    assert destination.read_bytes().startswith(b"P6\n")
    assert destination.stat().st_size < source.stat().st_size


def test_header_comments_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "commented.ppm"
    path.write_bytes(
        b"P6\n# written by a tool\n2 1\n# another comment\n255\n" + bytes([1, 2, 3, 4, 5, 6])
    )
    image = PPMImage()
    image.load(str(path))
    assert image.get_dimensions() == (2, 1)
    assert image.get_raw_data() == [(1, 2, 3), (4, 5, 6)]


@pytest.mark.parametrize("separator", [b" ", b"\t", b"\r\n", b"\n\n  \n"])
def test_any_whitespace_separates_header_fields(tmp_path: Path, separator: bytes) -> None:
    path = tmp_path / "spaced.ppm"
    path.write_bytes(
        b"P6" + separator + b"1" + separator + b"1" + separator + b"255\n" + bytes([9, 8, 7])
    )
    image = PPMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(9, 8, 7)]


def test_low_maxval_is_rescaled_to_full_range(tmp_path: Path) -> None:
    """A maxval of 1 means each sample is 0 or 1, which maps to 0 or 255."""
    path = tmp_path / "bilevel.ppm"
    path.write_bytes(b"P6\n2 1\n1\n" + bytes([0, 1, 0, 1, 1, 1]))
    image = PPMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(0, 255, 0), (255, 255, 255)]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"P5\n1 1\n255\n\x00", "P3 or P6"),
        (b"P6\n1 1\n65535\n", "16-bit"),
        (b"P6\n0 1\n255\n", "width"),
        (b"P6\n1 x\n255\n", "not a number"),
        (b"P6\n1 1\n255\n\x00", "truncated PPM data"),
        (b"P6\n2 2\n", "truncated PPM header"),
        (b"P3\n2 2\n255\n1 2 3", "truncated PPM data"),
        (b"P3\n1 1\n255\n1 2 zz", "not a number"),
    ],
)
def test_malformed_ppm_is_rejected(tmp_path: Path, content: bytes, match: str) -> None:
    path = tmp_path / "bad.ppm"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        PPMImage().load(str(path))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("x.bmp", BMPImage),
        ("x.BMP", BMPImage),
        ("x.ppm", PPMImage),
        ("x.PPM", PPMImage),
        ("x.pnm", PPMImage),
    ],
)
def test_reader_for_dispatches_on_suffix(name: str, expected: type) -> None:
    assert isinstance(reader_for(name), expected)


def test_reader_for_defaults_to_bmp_without_a_suffix() -> None:
    """Keeps stdin/stdout piping working, which carries no filename."""
    assert isinstance(reader_for(None), BMPImage)
    assert isinstance(reader_for("noextension"), BMPImage)


def test_reader_for_rejects_an_unknown_suffix() -> None:
    with pytest.raises(UnsupportedFileFormatError, match="unsupported image format"):
        reader_for("photo.jpg")


def test_bmp_and_ppm_agree_on_the_same_picture(tmp_path: Path) -> None:
    """The point of the shared RGB contract: identical pixels either way."""
    pixels = gradient_pixels(8, 5)
    bmp = write_bmp(tmp_path / "same.bmp", 8, 5, pixels)
    ppm = write_ppm(tmp_path / "same.ppm", 8, 5, pixels)

    from_bmp = BMPImage()
    from_bmp.load(str(bmp))
    from_ppm = PPMImage()
    from_ppm.load(str(ppm))

    assert from_bmp.get_raw_data() == from_ppm.get_raw_data() == pixels
    assert from_bmp.get_dimensions() == from_ppm.get_dimensions()
