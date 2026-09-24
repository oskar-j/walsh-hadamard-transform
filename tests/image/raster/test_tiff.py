from __future__ import annotations

import struct
import tracemalloc
from pathlib import Path

import pytest

from conftest import (
    build_tiff,
    build_tiff_with_strip_table,
    gradient_pixels,
    write_bmp,
    write_ppm,
    write_tiff,
)
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


def test_a_final_strip_padded_to_a_whole_row_still_loads(tmp_path: Path) -> None:
    """RowsPerStrip does not have to divide the height, so the last strip may
    carry padding. Its byte count then pushes the declared total past the
    pixels the image holds, which is ordinary and must not be rejected."""
    width, height = 4, 3
    row = width * 3
    payload = bytes(range(2 * row)) + bytes(range(100, 100 + 2 * row))
    path = tmp_path / "padded.tif"
    path.write_bytes(
        build_tiff_with_strip_table(
            width, height, offsets=[0, 2 * row], counts=[2 * row, 2 * row], payload=payload
        )
    )

    image = TIFFImage()
    image.load(str(path))
    assert image.get_dimensions() == (width, height)
    data = image.get_raw_data()
    assert len(data) == width * height
    assert data[0] == (0, 1, 2)
    assert data[-1] == (109, 110, 111)


def test_strips_beyond_the_declared_pixels_are_rejected(tmp_path: Path) -> None:
    """Several entries pointing at one region made the reader accumulate
    gigabytes for a tiny image. Reads are now clamped to what is outstanding,
    and a strip with nothing left to contribute is a disagreement worth
    naming."""
    width = height = 8
    repeats = 200
    chunk = 4096
    path = tmp_path / "repeated.tif"
    path.write_bytes(
        build_tiff_with_strip_table(
            width,
            height,
            offsets=[0] * repeats,
            counts=[chunk] * repeats,
            payload=bytes(chunk),
        )
    )

    with pytest.raises(UnsupportedFileFormatError, match="lies beyond the 192 bytes"):
        TIFFImage().load(str(path))


def test_strips_that_reread_one_region_cannot_outgrow_the_file(tmp_path: Path) -> None:
    """The clamp above bounds the pixels by the dimensions the header
    declares, which the file chooses. Declare enough of them and it stops
    biting: a 1 MB file whose 512 strips all pointed at one region loaded as
    a 16384x10922 picture, that megabyte repeated, with no error (#57). An
    uncompressed image cannot hold more pixel bytes than its file; this is
    the same shape at a size a test can afford, 12 strips over 4 KB."""
    region = bytes(range(256)) * 16
    path = tmp_path / "repeated.tif"
    path.write_bytes(
        build_tiff_with_strip_table(
            128, 128, offsets=[0] * 12, counts=[len(region)] * 12, payload=region
        )
    )
    size = path.stat().st_size
    assert 12 * len(region) == 128 * 128 * 3 > size

    with pytest.raises(
        UnsupportedFileFormatError,
        match=f"a 128x128 image, 49152 bytes of pixels, but the whole file is {size} bytes",
    ):
        TIFFImage().load(str(path))


def _ifd_of(*entries: bytes) -> bytes:
    """A little-endian header and one directory holding ``entries`` verbatim."""
    return (
        struct.pack("<2sHI", b"II", 42, 8)
        + struct.pack("<H", len(entries))
        + b"".join(entries)
        + struct.pack("<I", 0)
    )


def test_an_out_of_line_value_is_checked_against_the_file_before_it_is_read(
    tmp_path: Path,
) -> None:
    """``count`` is a 32-bit field and ``file.read(n)`` allocates n bytes
    before reading any, so a file this size could ask for 4 GiB, or 34 GB in
    DOUBLEs, and only then report that the field ran past the end (#57)."""
    path = tmp_path / "long.tif"
    path.write_bytes(_ifd_of(_entry("<", 273, 4, 2**24) + struct.pack("<I", 0)))
    assert path.stat().st_size == 26

    tracemalloc.start()
    try:
        with pytest.raises(
            UnsupportedFileFormatError, match="TIFF field at offset 0 runs past the end"
        ):
            TIFFImage().load(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


@pytest.mark.parametrize(
    ("entry", "name"),
    [
        ((50000, 4, 2**24, 8), "a count running 64 MiB past the end of the file"),
        ((50000, 99, 1, 0), "a field type no TIFF defines"),
        ((50000, 3, 4, 2**31), "an offset beyond the end of the file"),
    ],
)
def test_a_tag_this_profile_never_reads_is_never_decoded(
    entry: tuple[int, int, int, int], name: str, tmp_path: Path
) -> None:
    """Each of these used to refuse a picture that did not need the field,
    and the first asked for its 64 MiB before doing so. The specification
    asks a reader to skip a field it does not know, and now nothing about
    one is even looked at beyond its tag (#57)."""
    pixels = gradient_pixels(3, 2)
    path = tmp_path / "junk.tif"
    path.write_bytes(build_tiff(3, 2, pixels, extra_entries=[entry]))

    image = TIFFImage()
    image.load(str(path))
    assert image.get_raw_data() == pixels, name


@pytest.mark.parametrize(
    ("tag", "written", "field_type", "name"),
    [
        (259, 3, 8, "Compression as a signed SHORT, which read as absent meant none"),
        (256, 4, 5, "ImageWidth as a RATIONAL, which read as absent was 'missing'"),
    ],
)
def test_a_tag_this_profile_reads_must_be_an_unsigned_integer(
    tag: int, written: int, field_type: int, name: str, tmp_path: Path
) -> None:
    """Every tag the profile reads is a SHORT or a LONG. Another type used to
    decode to nothing, so the tag counted as absent and its default applied,
    and a compressed file could have been read as raw pixels."""
    data = build_tiff(2, 2, gradient_pixels(2, 2))
    entry = _entry("<", tag, written, 1)
    assert data.count(entry) == 1
    path = tmp_path / "typed.tif"
    path.write_bytes(data.replace(entry, _entry("<", tag, field_type, 1)))
    with pytest.raises(
        UnsupportedFileFormatError, match=f"TIFF tag {tag} has field type {field_type}"
    ):
        TIFFImage().load(str(path))


def test_many_entries_over_one_region_cost_nothing(tmp_path: Path) -> None:
    """Every entry's value used to be decoded and kept, before the profile
    was even checked: 200 junk entries over one region of a 264 KB file cost
    100 MiB. 200 entries of 12 KB each, over the same 12 KB, here (#57)."""
    pixels = gradient_pixels(64, 64)
    junk = [(40000 + index, 4, 3000, 8) for index in range(200)]
    path = tmp_path / "junk.tif"
    path.write_bytes(build_tiff(64, 64, pixels, extra_entries=junk))
    assert path.stat().st_size > 8 + 3000 * 4

    tracemalloc.start()
    try:
        image = TIFFImage()
        image.load(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert image.get_raw_data() == pixels
    assert peak < 1_000_000


def test_a_tag_this_profile_reads_may_appear_only_once(tmp_path: Path) -> None:
    """Two strip tables describe no single image, and the last used to win
    without a word. Decoding every copy also let 16,384 of them, each over
    the same region of a 200 KB file, cost ten seconds (#57)."""
    path = tmp_path / "twice.tif"
    path.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), extra_entries=[(273, 4, 1, 8)]))
    with pytest.raises(
        UnsupportedFileFormatError, match="TIFF directory holds tag 273 more than once"
    ):
        TIFFImage().load(str(path))


# --- every guard only a file we did not write can reach (#26) -------------------


def _entry(prefix: str, tag: int, field_type: int, count: int) -> bytes:
    return struct.pack(f"{prefix}HHI", tag, field_type, count)


def test_a_file_without_strip_offsets_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "no-offsets.tif"
    path.write_bytes(build_tiff(2, 2, gradient_pixels(2, 2), omit={273}))
    with pytest.raises(UnsupportedFileFormatError, match="TIFF has no strip offsets"):
        TIFFImage().load(str(path))


def test_an_unknown_field_type_is_refused_by_number(tmp_path: Path) -> None:
    data = build_tiff(2, 2, gradient_pixels(2, 2))
    compression = _entry("<", 259, 3, 1)
    assert data.count(compression) == 1
    path = tmp_path / "type99.tif"
    path.write_bytes(data.replace(compression, _entry("<", 259, 99, 1)))
    with pytest.raises(UnsupportedFileFormatError, match="unknown TIFF field type 99"):
        TIFFImage().load(str(path))


def test_a_value_offset_past_the_end_of_the_file_is_refused(tmp_path: Path) -> None:
    """BitsPerSample holds three SHORTs, six bytes, so it lives out of line."""
    data = build_tiff(2, 2, gradient_pixels(2, 2))
    bits = _entry("<", 258, 3, 3)
    at = data.index(bits) + len(bits)
    beyond = len(data) * 4
    path = tmp_path / "beyond.tif"
    path.write_bytes(data[:at] + struct.pack("<I", beyond) + data[at + 4 :])
    with pytest.raises(
        UnsupportedFileFormatError, match=f"TIFF field at offset {beyond} runs past the end"
    ):
        TIFFImage().load(str(path))


def test_a_directory_that_ends_inside_an_entry_is_refused(tmp_path: Path) -> None:
    """Header, a count of two, one whole entry, five bytes of the second."""
    data = (
        struct.pack("<2sHI", b"II", 42, 8)
        + struct.pack("<H", 2)
        + _entry("<", 256, 4, 1)
        + struct.pack("<I", 2)
        + b"\x01\x01\x03\x00\x01"
    )
    assert len(data) == 27
    path = tmp_path / "cut.tif"
    path.write_bytes(data)
    with pytest.raises(UnsupportedFileFormatError, match="truncated TIFF directory entry 1 of 2"):
        TIFFImage().load(str(path))


def test_a_strip_byte_count_short_of_the_pixels_is_refused(tmp_path: Path) -> None:
    """One strip of a 2x2 image is 12 bytes, and its count sits inline."""
    data = build_tiff(2, 2, gradient_pixels(2, 2))
    counts = _entry("<", 279, 4, 1) + struct.pack("<I", 12)
    assert data.count(counts) == 1
    path = tmp_path / "short.tif"
    path.write_bytes(data.replace(counts, _entry("<", 279, 4, 1) + struct.pack("<I", 6)))
    with pytest.raises(
        UnsupportedFileFormatError, match="truncated TIFF pixel data: expected 12 bytes, got 6"
    ):
        TIFFImage().load(str(path))


@pytest.mark.parametrize(
    ("tag", "field_type", "count", "name"),
    [
        (282, 5, 1, "XResolution, RATIONAL"),
        (306, 2, 20, "DateTime, ASCII"),
        (339, 3, 3, "SampleFormat, three SHORTs"),
        (34665, 4, 1, "an Exif IFD pointer, LONG"),
    ],
)
def test_tags_this_profile_never_needs_are_skipped_not_refused(
    tag: int, field_type: int, count: int, name: str, tmp_path: Path
) -> None:
    """Ordinary third-party files carry resolution, dates and more, in types
    the reader never decodes. They must be read past, whether inline or out
    of line, and the picture must come back unchanged."""
    pixels = gradient_pixels(3, 2)
    path = tmp_path / "tagged.tif"
    # An out-of-line value may point anywhere: the bytes are discarded.
    path.write_bytes(build_tiff(3, 2, pixels, extra_entries=[(tag, field_type, count, 8)]))
    image = TIFFImage()
    image.load(str(path))
    assert image.get_dimensions() == (3, 2), name
    assert image.get_raw_data() == pixels, name


def test_pillow_tiffs_load_and_pillow_reads_ours(tmp_path: Path) -> None:
    """The interoperability claim, pinned: every other TIFF the suite parses,
    the suite wrote. Pillow's default file and one with a resolution, which
    adds RATIONAL tags this reader keeps opaque, must both load, and Pillow
    must read what this writer produces."""
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    pixels = gradient_pixels(5, 4)
    array = np.array(pixels, dtype=np.uint8).reshape(4, 5, 3)
    picture = Image.fromarray(array)
    for name, options in (("plain", {}), ("with-dpi", {"dpi": (72, 72)})):
        path = tmp_path / f"{name}.tiff"
        picture.save(path, format="TIFF", **options)
        image = TIFFImage()
        image.load(str(path))
        assert image.get_dimensions() == (5, 4), name
        assert image.get_raw_data() == pixels, name

    ours = tmp_path / "ours.tiff"
    image.save(str(ours))
    with Image.open(ours) as read_back:
        assert read_back.size == (5, 4)
        assert np.array_equal(np.asarray(read_back.convert("RGB")), array)
