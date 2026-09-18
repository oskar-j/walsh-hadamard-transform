"""PNG: 8-bit RGB, or RGBA that is opaque throughout, inflated by stdlib zlib (#17).

The files read here are built by `build_png` in conftest, which filters one
byte at a time straight from the specification and shares no code with the
reader. That matters because the reader does not work that way: it undoes the
filters a whole anti-diagonal at a time, and the byte loop is what it has to
agree with.
"""

from __future__ import annotations

import struct
import tracemalloc
import zlib
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    Pixel,
    build_png,
    gradient_pixels,
    png_chunk,
    png_unfilter_rows,
    write_png,
    write_ppm,
)
from walsh.image import PNG_SIGNATURE, PNGImage, PPMImage, UnsupportedFileFormatError, reader_for
from walsh.image.raster import png as png_module


def _random_pixels(width: int, height: int, seed: int = 17) -> list[Pixel]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(width * height, 3))
    return [(int(r), int(g), int(b)) for r, g, b in values]


def _load(path: Path) -> PNGImage:
    image = PNGImage()
    image.load(str(path))
    return image


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG into ``(type, payload)``, checking each CRC on the way."""
    assert data[:8] == PNG_SIGNATURE
    found, position = [], 8
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position : position + 8])
        payload = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        assert crc == zlib.crc32(kind + payload), kind
        found.append((kind, payload))
        position += 12 + length
    return found


# --- round trip and the written layout ---------------------------------------


def test_roundtrip(tmp_path: Path) -> None:
    pixels = gradient_pixels(7, 5)
    image = PNGImage()
    image.set_dimensions(7, 5)
    image.set_raw_data(pixels)
    path = tmp_path / "out.png"
    image.save(str(path))

    restored = _load(path)
    assert restored.get_dimensions() == (7, 5)
    assert restored.get_raw_data() == pixels


def test_written_layout_is_minimal(tmp_path: Path) -> None:
    """Signature, IHDR, one IDAT, IEND: eight-bit RGB, not interlaced."""
    image = PNGImage()
    image.set_dimensions(7, 5)
    image.set_raw_data(gradient_pixels(7, 5))
    path = tmp_path / "out.png"
    image.save(str(path))

    chunks = _chunks(path.read_bytes())
    assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[0][1]) == (7, 5, 8, 2, 0, 0, 0)
    assert len(zlib.decompress(chunks[1][1])) == 5 * (1 + 7 * 3)
    assert chunks[2][1] == b""


def test_the_written_rows_decode_under_the_byte_loop(tmp_path: Path) -> None:
    """The writer filters the whole image in one pass per filter type. What it
    writes is checked here by the spec's own byte-at-a-time rule rather than
    by this package's reader, so the two cannot be wrong in the same way."""
    width, height = 13, 9
    pixels = _random_pixels(width, height) + gradient_pixels(width, height)
    image = PNGImage()
    image.set_dimensions(width, height * 2)
    image.set_raw_data(pixels)
    path = tmp_path / "out.png"
    image.save(str(path))

    filtered = zlib.decompress(_chunks(path.read_bytes())[1][1])
    rows = png_unfilter_rows(filtered, height * 2, 3)
    assert b"".join(rows) == bytes(channel for pixel in pixels for channel in pixel)


def test_the_writer_gives_each_row_the_filter_with_the_smallest_sum() -> None:
    """Black ties at zero under every filter, and a tie goes to the lowest
    type. Flat grey costs a whole row unfiltered, one pixel under Sub, and
    nothing at all under Up once there is a row above to predict from."""
    black = np.zeros((4, 6, 3), dtype=np.uint8)
    assert png_module._filter_rows(black)[:, 0].tolist() == [0, 0, 0, 0]

    grey = np.full((4, 6, 3), 90, dtype=np.uint8)
    filtered = png_module._filter_rows(grey)
    assert filtered[:, 0].tolist() == [1, 2, 2, 2]
    assert filtered[0, 1:].tolist() == [90, 90, 90] + [0] * 15
    assert not filtered[1:, 1:].any()


def test_the_writer_reads_a_filtered_byte_as_signed_when_it_adds_them_up() -> None:
    """A ramp falling by one leaves 255 in every byte under Sub, which is -1:
    a sum of ones. Counted as 255 each, Sub would lose to no filter at all,
    and the row would be written a great deal larger than it need be."""
    ramp = np.arange(200, 188, -1, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
    filtered = png_module._filter_rows(ramp)
    assert filtered[0].tolist() == [1, 200, 200, 200] + [255] * 33


def test_the_writer_filters_the_sample_exactly_as_libpng_did(sample_png: Path) -> None:
    """data/png/earth.png was written by libpng, which chooses a filter per
    row by the same rule. Filtering its pixels again here gives libpng's
    filtered rows back, every filter type and every byte: the two encoders
    differ only in what DEFLATE then makes of them."""
    packed = b"".join(data for kind, data in _chunks(sample_png.read_bytes()) if kind == b"IDAT")
    theirs = np.frombuffer(zlib.decompress(packed), dtype=np.uint8).reshape(400, -1)

    ours = png_module._filter_rows(_load(sample_png).get_array())
    assert np.bincount(ours[:, 0], minlength=5).tolist() == [0, 45, 0, 354, 1]
    assert np.array_equal(ours, theirs)


def test_filtering_in_bands_changes_no_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    """The writer filters a band of rows at a time to bound its memory. A
    row's filter depends only on the row above, which the band before owns."""
    rng = np.random.default_rng(29)
    smooth = np.add.outer(np.arange(23) * 9, np.arange(17) * 5)[:, :, None] + np.arange(3) * 30
    pixels = ((smooth + rng.integers(0, 40, size=smooth.shape)) % 256).astype(np.uint8)

    whole = png_module._filter_rows(pixels)
    assert len(set(whole[:, 0].tolist())) >= 3, "a picture that exercises several filters"
    monkeypatch.setattr(png_module, "_FILTER_BAND_ROWS", 4)
    assert np.array_equal(png_module._filter_rows(pixels), whole)

    rows = png_unfilter_rows(whole.tobytes(), 23, 3)
    assert b"".join(rows) == pixels.tobytes()


def test_a_large_stream_is_split_over_several_idat_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk's length is 31 bits, so the writer must be able to split. The
    limit is lowered here rather than two gigabytes written."""
    monkeypatch.setattr(png_module, "_IDAT_LIMIT", 40)
    pixels = _random_pixels(9, 9)
    image = PNGImage()
    image.set_dimensions(9, 9)
    image.set_raw_data(pixels)
    path = tmp_path / "split.png"
    image.save(str(path))

    kinds = [kind for kind, _ in _chunks(path.read_bytes())]
    assert kinds.count(b"IDAT") > 3
    assert all(len(payload) <= 40 for kind, payload in _chunks(path.read_bytes()))
    assert _load(path).get_raw_data() == pixels


def test_an_empty_image_cannot_be_written(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    with pytest.raises(ValueError, match="cannot hold an empty image"):
        PNGImage().save(str(path))
    assert not path.exists()


@pytest.mark.parametrize("name", ["x.png", "x.PNG", "dir.with.dots/x.Png"])
def test_suffix_dispatch(name: str) -> None:
    assert isinstance(reader_for(name), PNGImage)


# --- the five filters ---------------------------------------------------------


@pytest.mark.parametrize("alpha", [None, 255], ids=["rgb", "rgba"])
@pytest.mark.parametrize("kind", range(5), ids=["none", "sub", "up", "average", "paeth"])
def test_every_filter_type_is_undone(tmp_path: Path, kind: int, alpha: int | None) -> None:
    pixels = _random_pixels(11, 7)
    path = write_png(tmp_path / "f.png", 11, 7, pixels, filters=kind, alpha=alpha)
    assert _load(path).get_raw_data() == pixels


@pytest.mark.parametrize(
    ("width", "height"),
    [(1, 1), (9, 1), (1, 9), (2, 2), (64, 5), (3, 70), (33, 33)],
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("kinds", ["all five", "none, sub and up"])
def test_rows_may_each_choose_their_own_filter(
    tmp_path: Path, width: int, height: int, kinds: str
) -> None:
    """Encoders choose per row. With Average or Paeth anywhere the band goes
    through the wavefront, and without them through the row-wise path, so both
    are run over every shape, including the single row and single column
    where a diagonal is one pixel long."""
    rng = np.random.default_rng(width * 1000 + height)
    filters = rng.integers(0, 5 if kinds == "all five" else 3, size=height).tolist()
    pixels = _random_pixels(width, height, seed=height)
    path = write_png(tmp_path / "mixed.png", width, height, pixels, filters=filters)

    image = _load(path)
    assert image.get_dimensions() == (width, height)
    assert image.get_raw_data() == pixels


def test_a_band_takes_its_first_row_from_the_band_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tall images are unfiltered in bands so the wavefront's grid stays
    bounded. The floor is lowered to four rows here, so 23 rows make six
    bands. Each starts on a row that predicts from the row above, which
    belongs to the band before, and the bands alternate between the wavefront
    and the row-wise path, so the hand-over is crossed in both directions."""
    monkeypatch.setattr(png_module, "_MIN_BAND_ROWS", 4)
    bands = [(4, 3, 1, 2), (2, 1, 0, 2), (3, 0, 4, 1), (2, 2, 2, 2), (4, 4, 4, 4), (2, 0, 1)]
    rows = [kind for band in bands for kind in band]
    assert all(band[0] in (2, 3, 4) for band in bands)
    assert [any(kind >= 3 for kind in band) for band in bands] == [True, False] * 3

    width, height = 3, len(rows)
    pixels = _random_pixels(width, height)
    path = write_png(tmp_path / "bands.png", width, height, pixels, filters=rows)
    assert _load(path).get_raw_data() == pixels


def test_a_row_with_a_sixth_filter_type_is_refused(tmp_path: Path) -> None:
    path = write_png(tmp_path / "f.png", 4, 3, gradient_pixels(4, 3), filters=[0, 5, 1])
    with pytest.raises(UnsupportedFileFormatError, match="row 1 uses filter type 5"):
        _load(path)


# --- chunks -------------------------------------------------------------------


def test_the_stream_may_be_spread_over_many_idat_chunks(tmp_path: Path) -> None:
    pixels = _random_pixels(12, 12)
    path = write_png(tmp_path / "many.png", 12, 12, pixels, filters=range(5), idat_size=7)
    assert [kind for kind, _ in _chunks(path.read_bytes())].count(b"IDAT") > 10
    assert _load(path).get_raw_data() == pixels


def test_chunks_that_change_no_pixel_are_skipped(tmp_path: Path) -> None:
    """Ancillary chunks, and PLTE, which in a truecolour file only suggests a
    palette to quantise to. `acTL` is what makes a file an animated PNG: its
    still image is what is read."""
    pixels = gradient_pixels(5, 4)
    path = write_png(
        tmp_path / "extras.png",
        5,
        4,
        pixels,
        before_idat=[
            png_chunk(b"gAMA", struct.pack(">I", 45455)),
            png_chunk(b"sRGB", b"\x00"),
            png_chunk(b"PLTE", bytes(range(12))),
            png_chunk(b"acTL", struct.pack(">II", 1, 0)),
        ],
        after_idat=[png_chunk(b"tEXt", b"Comment\x00made by hand"), png_chunk(b"tIME", bytes(7))],
    )
    assert _load(path).get_raw_data() == pixels


def test_bytes_after_iend_are_ignored(tmp_path: Path) -> None:
    pixels = gradient_pixels(3, 3)
    path = tmp_path / "trailing.png"
    path.write_bytes(build_png(3, 3, pixels) + b"appended by some uploader")
    assert _load(path).get_raw_data() == pixels


def test_a_critical_chunk_this_reader_does_not_know_is_refused(tmp_path: Path) -> None:
    """An upper-case first letter means a decoder that skips it shows the
    wrong picture, so the format forbids skipping."""
    path = write_png(
        tmp_path / "critical.png", 3, 3, gradient_pixels(3, 3), before_idat=[png_chunk(b"NEWc")]
    )
    with pytest.raises(UnsupportedFileFormatError, match=r"critical chunk.*NEWc"):
        _load(path)


MALFORMED = [
    ("empty", b"", "expected the PNG signature"),
    ("not-a-png", b"P6\n2 2\n255\n" + bytes(12), "expected the PNG signature"),
    ("signature-only", PNG_SIGNATURE, "ends without an IEND"),
    ("half-a-chunk-header", PNG_SIGNATURE + b"\x00\x00\x00", "ends inside a chunk header"),
    ("chunk-cut-short", build_png(4, 4, gradient_pixels(4, 4))[:20], "chunk IHDR is cut short"),
    (
        "bad-crc",
        PNG_SIGNATURE + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0), crc=1),
        "chunk IHDR fails its CRC",
    ),
    ("type-not-letters", PNG_SIGNATURE + png_chunk(b"ID4T"), "is not four letters"),
    (
        "length-above-the-limit",
        PNG_SIGNATURE + struct.pack(">I4s", 2**31, b"IDAT"),
        r"declares 2147483648 bytes, above the format's limit",
    ),
    (
        "length-the-file-cannot-hold",
        PNG_SIGNATURE + struct.pack(">I4s", 2**31 - 1, b"IHDR") + bytes(64),
        "chunk IHDR is cut short",
    ),
    ("no-ihdr-first", PNG_SIGNATURE + png_chunk(b"IEND"), "the first chunk is IEND, not IHDR"),
    ("ends-after-ihdr", build_png(2, 2, gradient_pixels(2, 2))[:33], "ends without an IEND"),
]


@pytest.mark.parametrize(("name", "content", "match"), MALFORMED, ids=[c[0] for c in MALFORMED])
def test_malformed_png_is_rejected(tmp_path: Path, name: str, content: bytes, match: str) -> None:
    path = tmp_path / f"{name}.png"
    path.write_bytes(content)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        _load(path)


def test_a_file_with_no_idat_chunk_is_refused(tmp_path: Path) -> None:
    header = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    path = tmp_path / "hollow.png"
    path.write_bytes(PNG_SIGNATURE + header + png_chunk(b"IEND"))
    with pytest.raises(UnsupportedFileFormatError, match="no IDAT chunk"):
        _load(path)


def test_a_second_header_is_refused(tmp_path: Path) -> None:
    again = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 9, 9, 8, 2, 0, 0, 0))
    path = write_png(tmp_path / "twice.png", 3, 3, gradient_pixels(3, 3), before_idat=[again])
    with pytest.raises(UnsupportedFileFormatError, match="second IHDR"):
        _load(path)


def test_idat_chunks_must_be_consecutive(tmp_path: Path) -> None:
    path = write_png(
        tmp_path / "split.png",
        6,
        6,
        _random_pixels(6, 6),
        idat_size=20,
        between_idat=png_chunk(b"tEXt", b"k\x00v"),
    )
    with pytest.raises(UnsupportedFileFormatError, match="IDAT chunks are not consecutive"):
        _load(path)


def test_a_file_that_ends_before_iend_is_truncated(tmp_path: Path) -> None:
    """Even with every pixel present: a cut-off download reads the same from
    wherever it was cut."""
    path = write_png(tmp_path / "cut.png", 3, 3, gradient_pixels(3, 3), end=False)
    with pytest.raises(UnsupportedFileFormatError, match="ends without an IEND"):
        _load(path)


# --- the header ---------------------------------------------------------------


UNSUPPORTED = [
    ("palette", {"colour_type": 3}, r"colour type 3 \(palette\)"),
    ("greyscale", {"colour_type": 0}, r"colour type 0 \(greyscale\)"),
    ("grey-alpha", {"colour_type": 4}, r"colour type 4 \(greyscale with alpha\)"),
    ("16-bit-rgb", {"bit_depth": 16}, "only 8 bits per sample is supported, got 16"),
    ("16-bit-rgba", {"bit_depth": 16, "colour_type": 6}, "got 16"),
    ("interlaced", {"interlace": 1}, r"interlaced \(Adam7\) PNG is not supported"),
]


@pytest.mark.parametrize(("name", "kwargs", "match"), UNSUPPORTED, ids=[c[0] for c in UNSUPPORTED])
def test_unsupported_profiles_are_rejected_by_name(
    tmp_path: Path, name: str, kwargs: dict[str, int], match: str
) -> None:
    path = write_png(tmp_path / f"{name}.png", 4, 4, gradient_pixels(4, 4), **kwargs)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        _load(path)


INVALID = [
    ("colour-type-5", {"colour_type": 5}, "unknown colour type 5"),
    ("4-bit-rgb", {"bit_depth": 4}, r"4 bits per sample is not allowed for colour type 2 \(RGB\)"),
    ("16-bit-palette", {"bit_depth": 16, "colour_type": 3}, "16 bits per sample is not allowed"),
    ("compression-1", {"compression": 1}, "compression method 1, filter method 0"),
    ("filter-method-1", {"filter_method": 1}, "filter method 1 and interlace method 0"),
    ("interlace-2", {"interlace": 2}, "interlace method 2"),
]


@pytest.mark.parametrize(("name", "kwargs", "match"), INVALID, ids=[c[0] for c in INVALID])
def test_a_header_the_format_forbids_is_invalid_rather_than_unsupported(
    tmp_path: Path, name: str, kwargs: dict[str, int], match: str
) -> None:
    path = write_png(tmp_path / f"{name}.png", 4, 4, gradient_pixels(4, 4), **kwargs)
    with pytest.raises(UnsupportedFileFormatError, match=f"invalid PNG header.*{match}"):
        _load(path)


@pytest.mark.parametrize(("width", "height"), [(0, 4), (4, 0), (2**31, 4)])
def test_impossible_dimensions_are_rejected(tmp_path: Path, width: int, height: int) -> None:
    header = png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    path = tmp_path / "size.png"
    path.write_bytes(PNG_SIGNATURE + header + png_chunk(b"IEND"))
    with pytest.raises(UnsupportedFileFormatError, match=f"invalid PNG dimensions {width}x"):
        _load(path)


def test_a_header_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "short.png"
    path.write_bytes(PNG_SIGNATURE + png_chunk(b"IHDR", bytes(12)) + png_chunk(b"IEND"))
    with pytest.raises(UnsupportedFileFormatError, match="IHDR holds 12 bytes, not 13"):
        _load(path)


# --- pixel data ---------------------------------------------------------------


def test_pixel_data_that_is_not_a_zlib_stream_is_refused(tmp_path: Path) -> None:
    """The chunk's CRC is right, so only the inflater can tell."""
    header = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    path = tmp_path / "garbage.png"
    path.write_bytes(
        PNG_SIGNATURE + header + png_chunk(b"IDAT", b"not deflate") + png_chunk(b"IEND")
    )
    with pytest.raises(UnsupportedFileFormatError, match="corrupt PNG pixel data"):
        _load(path)


def test_inflation_carries_on_from_where_the_last_step_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inflater is asked for a bounded number of bytes at a time, and what
    it has not consumed is fed back. Five bytes a step makes that every step."""
    monkeypatch.setattr(png_module, "_INFLATE_STEP", 5)
    pixels = _random_pixels(10, 10)
    path = write_png(tmp_path / "steps.png", 10, 10, pixels, filters=range(5), idat_size=33)
    assert _load(path).get_raw_data() == pixels


def test_pixel_data_short_of_the_declared_size_is_truncated(tmp_path: Path) -> None:
    """The header says 4x6 and the stream holds 4x4."""
    data = bytearray(build_png(4, 4, gradient_pixels(4, 4)))
    data[8:33] = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 6, 8, 2, 0, 0, 0))
    path = tmp_path / "short.png"
    path.write_bytes(bytes(data))
    with pytest.raises(UnsupportedFileFormatError, match="expected 78 bytes, got 52"):
        _load(path)


def test_a_header_that_declares_gigabytes_costs_nothing(tmp_path: Path) -> None:
    """Nothing is allocated from the header, only from what the stream
    delivers, so a lie about the size is reported for a few kilobytes."""
    data = bytearray(build_png(4, 4, gradient_pixels(4, 4)))
    data[8:33] = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2**31 - 1, 2**31 - 1, 8, 6, 0, 0, 0))
    path = tmp_path / "liar.png"
    path.write_bytes(bytes(data))

    tracemalloc.start()
    try:
        with pytest.raises(UnsupportedFileFormatError, match="truncated PNG pixel data"):
            _load(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_inflation_stops_at_the_size_the_header_declares(tmp_path: Path) -> None:
    """DEFLATE expands a thousandfold, so 60 KB of compressed zeros after an
    8x8 image would otherwise cost 64 MB. The surplus is never inflated, and
    the picture still loads: libpng ignores such a surplus too."""
    pixels = gradient_pixels(8, 8)
    path = write_png(tmp_path / "bomb.png", 8, 8, pixels, surplus=bytes(64 * 2**20))
    assert path.stat().st_size < 100_000

    tracemalloc.start()
    try:
        image = _load(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert image.get_raw_data() == pixels
    assert peak < 2_000_000


# --- transparency -------------------------------------------------------------


def test_rgba_that_is_opaque_throughout_is_read_as_rgb(tmp_path: Path) -> None:
    pixels = _random_pixels(6, 5)
    path = write_png(tmp_path / "opaque.png", 6, 5, pixels, alpha=255, filters=range(5))
    image = _load(path)
    assert image.get_array().shape == (5, 6, 3)
    assert image.get_raw_data() == pixels


def test_real_transparency_is_refused_with_a_count(tmp_path: Path) -> None:
    """Dropping alpha means choosing a background, and the file names none."""
    alpha = [255] * 30
    alpha[7] = 254
    alpha[22] = 0
    path = write_png(tmp_path / "glass.png", 6, 5, _random_pixels(6, 5), alpha=alpha)
    with pytest.raises(UnsupportedFileFormatError, match="alpha is below 255 in 2 of 30 pixels"):
        _load(path)


def test_a_trns_colour_that_some_pixel_has_is_transparency_too(tmp_path: Path) -> None:
    """An RGB file can key one colour out through tRNS. It is an ancillary
    chunk, so a reader that skipped it would flatten the picture silently."""
    pixels = gradient_pixels(4, 4)
    key = png_chunk(b"tRNS", struct.pack(">3H", *pixels[5]))
    path = write_png(tmp_path / "keyed.png", 4, 4, pixels, before_idat=[key])
    with pytest.raises(UnsupportedFileFormatError, match=r"tRNS.*1 of 16 pixels have it"):
        _load(path)


@pytest.mark.parametrize(
    "key", [(1, 2, 3), (300, 0, 0)], ids=["a colour no pixel has", "a sample above eight bits"]
)
def test_a_trns_colour_that_no_pixel_has_changes_nothing(
    tmp_path: Path, key: tuple[int, int, int]
) -> None:
    pixels = gradient_pixels(4, 4)
    assert key not in pixels
    chunk = png_chunk(b"tRNS", struct.pack(">3H", *key))
    path = write_png(tmp_path / "unkeyed.png", 4, 4, pixels, before_idat=[chunk])
    assert _load(path).get_raw_data() == pixels


def test_a_trns_chunk_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    path = write_png(
        tmp_path / "trns.png", 4, 4, gradient_pixels(4, 4), before_idat=[png_chunk(b"tRNS", b"\0")]
    )
    with pytest.raises(UnsupportedFileFormatError, match="tRNS holds 1 bytes"):
        _load(path)


def test_a_trns_chunk_in_an_rgba_file_is_ignored(tmp_path: Path) -> None:
    """The format forbids it there, and the alpha channel is what decides."""
    pixels = gradient_pixels(4, 4)
    chunk = png_chunk(b"tRNS", struct.pack(">3H", *pixels[0]))
    path = write_png(tmp_path / "both.png", 4, 4, pixels, alpha=255, before_idat=[chunk])
    assert _load(path).get_raw_data() == pixels


# --- other implementations ----------------------------------------------------


def test_the_sample_written_by_libpng_is_the_picture_in_the_ppm(
    sample_png: Path, sample_ppm: Path
) -> None:
    """data/png/earth.png came from Netpbm's pnmtopng, so it is a foreign
    writer's file: 37 IDAT chunks and Sub, Average and Paeth rows."""
    expected = PPMImage()
    expected.load(str(sample_ppm))
    assert np.array_equal(_load(sample_png).get_array(), expected.get_array())


def test_png_and_ppm_of_one_picture_agree(tmp_path: Path) -> None:
    pixels = gradient_pixels(8, 8)
    from_ppm = PPMImage()
    from_ppm.load(str(write_ppm(tmp_path / "p.ppm", 8, 8, pixels)))
    from_png = _load(write_png(tmp_path / "p.png", 8, 8, pixels, filters=range(5)))
    assert np.array_equal(from_png.get_array(), from_ppm.get_array())


def test_pillow_pngs_load_and_pillow_reads_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The independent check. Pillow is in the `demo` extra, which CI's test
    jobs do not install, so this skips there and runs locally."""
    pil_image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(3)
    smooth = np.add.outer(np.arange(40) * 5, np.arange(50) * 3)[:, :, None] + np.arange(3) * 40
    array = ((smooth + rng.integers(0, 6, size=smooth.shape)) % 256).astype(np.uint8)

    for label, mode, options in [
        ("plain", "RGB", {}),
        ("optimised", "RGB", {"optimize": True}),
        ("opaque-rgba", "RGBA", {}),
    ]:
        path = tmp_path / f"{label}.png"
        pil_image.fromarray(array).convert(mode).save(path, **options)
        assert np.array_equal(_load(path).get_array(), array), label

    translucent = pil_image.fromarray(array).convert("RGBA")
    translucent.putpixel((3, 3), (0, 0, 0, 10))
    translucent.save(tmp_path / "translucent.png")
    with pytest.raises(UnsupportedFileFormatError, match="1 of 2000 pixels"):
        _load(tmp_path / "translucent.png")

    pil_image.fromarray(array).convert("P").save(tmp_path / "palette.png")
    with pytest.raises(UnsupportedFileFormatError, match="palette"):
        _load(tmp_path / "palette.png")

    ours = PNGImage()
    ours.set_array(array.copy())
    for label, limit in [("whole", 2**31 - 1), ("split", 64)]:
        monkeypatch.setattr(png_module, "_IDAT_LIMIT", limit)
        path = tmp_path / f"ours-{label}.png"
        ours.save(str(path))
        with pil_image.open(path) as opened:
            assert opened.mode == "RGB"
            assert np.array_equal(np.asarray(opened), array), label
