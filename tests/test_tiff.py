from __future__ import annotations

import struct
from pathlib import Path

import pytest

from conftest import build_tiff, gradient_pixels, write_bmp, write_ppm, write_tiff
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import BMPImage, PPMImage, TIFFImage, reader_for


def test_roundtrip(tmp_path: Path) -> None:
    pixels = gradient_pixels(5, 4)
    source = write_tiff(tmp_path / "in.tif", 5, 4, pixels)

    image = TIFFImage()
    image.load(str(source))
    assert image.get_dimensions() == (5, 4)
    assert image.get_raw_data() == pixels

    # Not compared byte for byte against the source: the fixture writes an
    # Orientation tag so it can also produce rotated files, and the writer
    # omits it, which is a different but equally valid encoding.
    destination = tmp_path / "out.tif"
    image.save(str(destination))

    reread = TIFFImage()
    reread.load(str(destination))
    assert reread.get_raw_data() == pixels
    assert reread.get_dimensions() == (5, 4)


def test_our_own_output_is_byte_stable(tmp_path: Path) -> None:
    """Reading and rewriting our own file must be a fixed point."""
    pixels = gradient_pixels(5, 4)
    first, second = tmp_path / "1.tif", tmp_path / "2.tif"

    original = TIFFImage()
    original.set_dimensions(5, 4)
    original.set_raw_data(pixels)
    original.save(str(first))

    reread = TIFFImage()
    reread.load(str(first))
    reread.save(str(second))
    assert second.read_bytes() == first.read_bytes()


def test_big_endian_is_read(tmp_path: Path) -> None:
    """The header declares its own byte order, so both must work."""
    pixels = gradient_pixels(3, 2)
    little = tmp_path / "le.tif"
    big = tmp_path / "be.tif"
    little.write_bytes(build_tiff(3, 2, pixels, order=b"II"))
    big.write_bytes(build_tiff(3, 2, pixels, order=b"MM"))

    from_little, from_big = TIFFImage(), TIFFImage()
    from_little.load(str(little))
    from_big.load(str(big))
    assert from_little.get_raw_data() == from_big.get_raw_data() == pixels


def test_multiple_strips_are_concatenated(tmp_path: Path) -> None:
    pixels = gradient_pixels(4, 8)
    path = tmp_path / "strips.tif"
    path.write_bytes(build_tiff(4, 8, pixels, rows_per_strip=2))

    image = TIFFImage()
    image.load(str(path))
    assert image.get_raw_data() == pixels


def test_always_written_little_endian(tmp_path: Path) -> None:
    """Whatever was read, output is II; the two carry identical information."""
    source = tmp_path / "be.tif"
    source.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), order=b"MM"))

    image = TIFFImage()
    image.load(str(source))
    destination = tmp_path / "out.tif"
    image.save(str(destination))
    assert destination.read_bytes()[:2] == b"II"

    reread = TIFFImage()
    reread.load(str(destination))
    assert reread.get_raw_data() == image.get_raw_data()


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        ("LZW", {"compression": 5}, "compression 5 .LZW."),
        ("PackBits", {"compression": 32773}, "compression 32773 .PackBits."),
        ("palette", {"photometric": 3}, "palette colour"),
        ("CMYK", {"photometric": 5}, "CMYK"),
        ("greyscale", {"samples": 1}, "samples per pixel"),
        ("16-bit", {"bits": (16, 16, 16)}, "8 bits per sample"),
        ("planar", {"planar": 2}, "chunky"),
        ("rotated", {"orientation": 6}, "top-left orientation"),
    ],
)
def test_unsupported_profiles_are_rejected(
    tmp_path: Path, name: str, kwargs: dict, match: str
) -> None:
    """TIFF is a container; anything outside the supported profile must be named."""
    path = tmp_path / "x.tif"
    path.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), **kwargs))
    with pytest.raises(UnsupportedFileFormatError, match=match):
        TIFFImage().load(str(path))


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("empty", b"", "truncated TIFF header"),
        ("short header", b"II*\x00", "truncated TIFF header"),
        ("bad byte order", b"XX\x2a\x00\x08\x00\x00\x00", "byte order marker"),
        ("bad magic", b"II\x63\x00\x08\x00\x00\x00", "TIFF magic"),
        ("directory past EOF", b"II\x2a\x00\xff\x00\x00\x00", "truncated TIFF directory"),
    ],
)
def test_malformed_tiff_is_rejected(tmp_path: Path, name: str, content: bytes, match: str) -> None:
    path = tmp_path / "bad.tif"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        TIFFImage().load(str(path))


def test_truncated_pixel_data_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cut.tif"
    full = build_tiff(4, 4, gradient_pixels(4, 4))
    path.write_bytes(full[:-20])
    with pytest.raises(UnsupportedFileFormatError, match="truncated TIFF strip"):
        TIFFImage().load(str(path))


def test_missing_required_tag_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nowidth.tif"
    path.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), omit={256}))
    with pytest.raises(UnsupportedFileFormatError, match="missing required tag 256"):
        TIFFImage().load(str(path))


def test_mismatched_strip_tags_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.tif"
    path.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), omit={279}))
    with pytest.raises(UnsupportedFileFormatError, match="strip offsets but 0 byte counts"):
        TIFFImage().load(str(path))


@pytest.mark.parametrize("name", ["x.tif", "x.TIF", "x.tiff", "x.TIFF"])
def test_suffix_dispatch(name: str) -> None:
    assert isinstance(reader_for(name), TIFFImage)


def test_all_three_formats_agree_on_the_same_picture(tmp_path: Path) -> None:
    """The shared RGB contract, now across three formats."""
    pixels = gradient_pixels(8, 5)
    readers = [
        (BMPImage(), write_bmp(tmp_path / "a.bmp", 8, 5, pixels)),
        (PPMImage(), write_ppm(tmp_path / "a.ppm", 8, 5, pixels)),
        (TIFFImage(), write_tiff(tmp_path / "a.tif", 8, 5, pixels)),
    ]
    for image, path in readers:
        image.load(str(path))
        assert image.get_raw_data() == pixels, f"{type(image).__name__} disagrees"
        assert image.get_dimensions() == (8, 5)


def test_written_header_layout_is_minimal(tmp_path: Path) -> None:
    """8-byte header, 10-entry directory, BitsPerSample, then pixels."""
    path = tmp_path / "layout.tif"
    image = TIFFImage()
    image.set_dimensions(4, 4)
    image.set_raw_data(gradient_pixels(4, 4))
    image.save(str(path))
    raw = path.read_bytes()

    order, magic, ifd_offset = struct.unpack("<2sHI", raw[:8])
    assert (order, magic, ifd_offset) == (b"II", 42, 8)
    (count,) = struct.unpack("<H", raw[8:10])
    assert count == 10
    tags = [struct.unpack("<H", raw[10 + i * 12 : 12 + i * 12])[0] for i in range(count)]
    assert tags == sorted(tags), "TIFF requires directory entries in tag order"
    assert len(raw) == 8 + 2 + count * 12 + 4 + 6 + 4 * 4 * 3
