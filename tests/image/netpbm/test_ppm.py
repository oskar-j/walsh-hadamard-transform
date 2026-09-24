from __future__ import annotations

import tracemalloc
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
        (
            b"P6\n2000000000 2000000000\n255\n\x01\x02\x03",
            "truncated PPM data: expected 12000000000000000000 bytes, got 3",
        ),
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


def test_a_header_cannot_make_the_reader_allocate_what_the_file_lacks(tmp_path: Path) -> None:
    """Nothing bounds the width and height, and file.read(n) allocates n
    bytes before reading any: this 20-byte file asked for 48 MiB, and one
    declaring 100000x10000 for 3 GB, before three bytes were found (#56)."""
    path = tmp_path / "liar.ppm"
    path.write_bytes(b"P6\n4096 4096\n255\n\x01\x02\x03")
    assert path.stat().st_size == 20

    tracemalloc.start()
    try:
        with pytest.raises(UnsupportedFileFormatError, match="expected 50331648 bytes, got 3"):
            PPMImage().load(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000


@pytest.mark.parametrize("content", [b"P6\n1 1\n15\n\x01\x02\x16", b"P3\n1 1\n15\n1 2 22"])
def test_samples_above_maxval_are_rejected(tmp_path: Path, content: bytes) -> None:
    """The format forbids it, and passing 22/15 of full scale through would not end well."""
    path = tmp_path / "over.ppm"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match="22 exceeds maxval 15"):
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


def test_ascii_samples_may_be_separated_by_any_run_of_whitespace(tmp_path: Path) -> None:
    """Netpbm allows any amount of whitespace between numbers, and the token
    reader consumes exactly one character after each, so runs of two or more,
    blank lines included, are the loop's own business."""
    path = tmp_path / "spaced.ppm"
    path.write_bytes(b"P3\n2 2\n255\n\n  1 2 3\t\t4 5 6\n\n\n7 8 9   10 11 12\n\n")
    image = PPMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]


def test_an_ascii_ppm_that_ends_before_its_samples_do_is_refused(tmp_path: Path) -> None:
    """P3 promises width * height * 3 numbers; six of twelve, then end of file."""
    path = tmp_path / "short.ppm"
    path.write_bytes(b"P3\n2 2\n255\n1 2 3 4 5 6\n")
    with pytest.raises(UnsupportedFileFormatError, match="truncated PPM data"):
        PPMImage().load(str(path))
