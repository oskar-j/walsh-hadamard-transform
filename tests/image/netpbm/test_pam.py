from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from conftest import gradient_pixels, write_pam, write_ppm
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import PAMImage, PPMImage, reader_for


def test_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    pixels = gradient_pixels(4, 3)
    source = write_pam(tmp_path / "in.pam", 4, 3, pixels)

    image = PAMImage()
    image.load(str(source))
    assert image.get_dimensions() == (4, 3)
    assert image.get_raw_data() == pixels

    destination = tmp_path / "out.pam"
    image.save(str(destination))
    assert destination.read_bytes() == source.read_bytes()


def test_header_is_the_netpbm_layout(tmp_path: Path) -> None:
    """Field order and spelling match pamtopam, so the two agree byte for byte."""
    image = PAMImage()
    image.set_dimensions(2, 1)
    image.set_raw_data([(1, 2, 3), (4, 5, 6)])
    path = tmp_path / "out.pam"
    image.save(str(path))
    assert path.read_bytes() == (
        b"P7\nWIDTH 2\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB\nENDHDR\n" + bytes(range(1, 7))
    )


def test_header_fields_may_come_in_any_order(tmp_path: Path) -> None:
    path = write_pam(
        tmp_path / "shuffled.pam",
        2,
        1,
        [(1, 2, 3), (4, 5, 6)],
        header_lines=["TUPLTYPE RGB", "MAXVAL 255", "DEPTH 3", "HEIGHT 1", "WIDTH 2"],
    )
    image = PAMImage()
    image.load(str(path))
    assert image.get_dimensions() == (2, 1)
    assert image.get_raw_data() == [(1, 2, 3), (4, 5, 6)]


def test_comments_blank_lines_and_stray_whitespace_are_tolerated(tmp_path: Path) -> None:
    path = write_pam(
        tmp_path / "noisy.pam",
        2,
        1,
        [(1, 2, 3), (4, 5, 6)],
        header_lines=[
            "# written by a tool",
            "",
            "WIDTH\t2  ",
            "   ",
            "HEIGHT 1",
            "# DEPTH 4",
            "DEPTH 3",
            "MAXVAL 255",
            "TUPLTYPE RGB",
        ],
    )
    image = PAMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(1, 2, 3), (4, 5, 6)]


def test_crlf_line_endings_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "crlf.pam"
    path.write_bytes(
        b"P7\r\nWIDTH 1\r\nHEIGHT 1\r\nDEPTH 3\r\nMAXVAL 255\r\nENDHDR\r\n" + bytes([9, 8, 7])
    )
    image = PAMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(9, 8, 7)]


def test_tupltype_is_optional(tmp_path: Path) -> None:
    """DEPTH 3 with MAXVAL 255 and no TUPLTYPE means RGB by convention."""
    path = write_pam(tmp_path / "untyped.pam", 1, 1, [(9, 8, 7)], tupltype=None)
    image = PAMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(9, 8, 7)]


def test_several_tupltype_lines_are_one_value(tmp_path: Path) -> None:
    """The specification joins repeated TUPLTYPE lines with a space."""
    accepted = write_pam(
        tmp_path / "split-ok.pam",
        1,
        1,
        [(9, 8, 7)],
        header_lines=["WIDTH 1", "HEIGHT 1", "DEPTH 3", "MAXVAL 255", "TUPLTYPE RGB", "TUPLTYPE"],
    )
    image = PAMImage()
    image.load(str(accepted))
    assert image.get_raw_data() == [(9, 8, 7)]

    rejected = write_pam(
        tmp_path / "split-bad.pam",
        1,
        1,
        [(9, 8, 7)],
        header_lines=["WIDTH 1", "HEIGHT 1", "DEPTH 3", "MAXVAL 255", "TUPLTYPE RGB", "TUPLTYPE X"],
    )
    with pytest.raises(UnsupportedFileFormatError, match="tuple type 'RGB X'"):
        PAMImage().load(str(rejected))


def test_low_maxval_is_rescaled_to_full_range(tmp_path: Path) -> None:
    """With maxval 15 a sample of 7 is 7/15 of full scale, which rounds to 119."""
    path = write_pam(tmp_path / "nibbles.pam", 1, 1, [(0, 15, 7)], maxval=15)
    image = PAMImage()
    image.load(str(path))
    assert image.get_raw_data() == [(0, 255, 119)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tupltype": "GRAYSCALE", "depth": 1}, "tuple type 'GRAYSCALE'"),
        ({"tupltype": "GRAYSCALE_ALPHA", "depth": 2}, "tuple type 'GRAYSCALE_ALPHA'"),
        ({"tupltype": "RGB_ALPHA", "depth": 4}, "tuple type 'RGB_ALPHA'"),
        ({"tupltype": "BLACKANDWHITE", "depth": 1}, "tuple type 'BLACKANDWHITE'"),
        ({"tupltype": None, "depth": 1}, "depth 1 is not supported"),
        ({"tupltype": "RGB", "depth": 4}, "depth 4 is not supported"),
        ({"maxval": 65535}, "16-bit"),
    ],
)
def test_other_profiles_are_rejected_by_name(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    """Like the TIFF reader: say what the file is, rather than guess at it."""
    path = write_pam(tmp_path / "other.pam", 1, 1, [(1, 2, 3)], **kwargs)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        PAMImage().load(str(path))


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"", "starting P7"),
        (b"P6\n1 1\n255\n\x00\x00\x00", "starting P7"),
        (b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\n", "no ENDHDR"),
        (b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR", "no ENDHDR"),
        (b"P7\nWIDTH 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n", "missing HEIGHT"),
        (b"P7\nWIDTH 1\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n", "WIDTH given twice"),
        (b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nGAMMA 2.2\nENDHDR\n", "unknown PAM header"),
        (b"P7\nWIDTH 0\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n", "invalid PAM WIDTH: 0"),
        (b"P7\nWIDTH x\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n", "not a number"),
        (b"P7\nWIDTH\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n", "not a number"),
        (
            b"P7\nWIDTH 2\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nENDHDR\n\x01\x02\x03",
            "truncated PAM data",
        ),
        (
            b"P7\nWIDTH 2000000000\nHEIGHT 2000000000\nDEPTH 3\nMAXVAL 255\nENDHDR\n\x01\x02\x03",
            "truncated PAM data: expected 12000000000000000000 bytes, got 3",
        ),
        (
            b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 15\nENDHDR\n\x01\x02\x16",
            "22 exceeds maxval 15",
        ),
    ],
)
def test_malformed_pam_is_rejected(tmp_path: Path, content: bytes, match: str) -> None:
    path = tmp_path / "bad.pam"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        PAMImage().load(str(path))


def test_a_header_cannot_make_the_reader_allocate_what_the_file_lacks(tmp_path: Path) -> None:
    """The PPM reader's bound, through the raster reader the two share (#56)."""
    path = write_pam(tmp_path / "liar.pam", 4096, 4096, [(1, 2, 3)])

    tracemalloc.start()
    try:
        with pytest.raises(UnsupportedFileFormatError, match="expected 50331648 bytes, got 3"):
            PAMImage().load(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000


@pytest.mark.parametrize("name", ["x.pam", "x.PAM"])
def test_reader_for_dispatches_pam(name: str) -> None:
    assert isinstance(reader_for(name), PAMImage)


def test_pam_and_ppm_agree_on_the_same_picture(tmp_path: Path) -> None:
    """Same raster behind a different header: identical pixels either way."""
    pixels = gradient_pixels(8, 5)
    pam = write_pam(tmp_path / "same.pam", 8, 5, pixels)
    ppm = write_ppm(tmp_path / "same.ppm", 8, 5, pixels)

    from_pam = PAMImage()
    from_pam.load(str(pam))
    from_ppm = PPMImage()
    from_ppm.load(str(ppm))

    assert from_pam.get_raw_data() == from_ppm.get_raw_data() == pixels
    assert from_pam.get_dimensions() == from_ppm.get_dimensions()
