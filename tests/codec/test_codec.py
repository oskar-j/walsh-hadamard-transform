from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from walsh.codec import Action, Codec
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import BMPImage, PPMImage, reader_for


def _pixels(path: Path) -> np.ndarray:
    image = reader_for(path)
    image.load(str(path))
    return np.asarray(image.get_raw_data(), dtype=float)


def test_compress_then_extract_restores_shape_and_approximate_content(
    gradient_bmp: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "out.cim"
    restored = tmp_path / "back.bmp"

    Codec().compress(input=str(gradient_bmp), output=str(compressed)).run()
    assert compressed.exists()

    Codec().extract(input=str(compressed), output=str(restored)).run()

    before, after = _pixels(gradient_bmp), _pixels(restored)
    assert before.shape == after.shape
    # Lossy, but a smooth gradient must survive with a small mean error.
    assert np.abs(before - after).mean() < 24


def test_compression_actually_shrinks_the_data(sample_bmp: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "sample.cim"
    Codec().compress(input=str(sample_bmp), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_bmp.stat().st_size


def test_smaller_packed_block_size_produces_a_smaller_file(
    gradient_bmp: Path, tmp_path: Path
) -> None:
    sizes = {}
    for packed in (2, 4):
        output = tmp_path / f"packed{packed}.cim"
        Codec(packed_block_size=packed).compress(input=str(gradient_bmp), output=str(output)).run()
        sizes[packed] = output.stat().st_size
    assert sizes[2] < sizes[4]


def test_non_multiple_dimensions_are_padded_and_cropped_back(
    tmp_path: Path,
) -> None:
    """An 18x18 image is a multiple of neither block size nor of the packed 4.

    20x20 was used here until 0.4.3 and hid the padding bug entirely: the
    reconstruction is constant on 4-wide tiles, so a pad boundary that lands on
    a multiple of 4 is exactly representable whatever the padding contains.
    """
    from conftest import write_bmp

    width = height = 18
    pixels = [(x * 12 % 256, y * 12 % 256, 128) for y in range(height) for x in range(width)]
    source = write_bmp(tmp_path / "odd.bmp", width, height, pixels)

    compressed = tmp_path / "odd.cim"
    restored = tmp_path / "odd-back.bmp"
    Codec().compress(input=str(source), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()

    image = BMPImage()
    image.load(str(restored))
    assert image.get_dimensions() == (width, height)
    assert len(image.get_raw_data()) == width * height

    before = np.asarray(pixels, dtype=float).reshape(height, width, 3)
    after = np.asarray(image.get_raw_data(), dtype=float).reshape(height, width, 3)
    # The edge columns and rows must be no worse than the interior. Padding
    # with zeros put them an order of magnitude out.
    interior = np.abs(before[:-1, :-1] - after[:-1, :-1]).mean()
    assert np.abs(before[:, -1] - after[:, -1]).mean() < interior + 5
    assert np.abs(before[-1, :] - after[-1, :]).mean() < interior + 5


def test_padding_size_helper() -> None:
    assert Codec._get_padding_size(16, 8) == 0
    assert Codec._get_padding_size(17, 8) == 7
    assert Codec._get_padding_size(20, 16) == 12


def test_merge_crops_padding() -> None:
    blocks = [np.full((4, 4), i, dtype=float) for i in range(4)]
    merged = Codec._merge(blocks, width=6, height=6)
    assert merged.shape == (6, 6)


def test_run_without_compress_or_extract_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"nothing to run; call compress\(\) or extract\(\)"):
        Codec().run()


def test_compress_and_extract_record_what_run_will_do() -> None:
    """Both only plan: nothing is read or written until run()."""
    codec = Codec().compress(input="missing.ppm", output="never.cim")
    assert (codec._action, codec._input, codec._output) == (
        Action.COMPRESS,
        "missing.ppm",
        "never.cim",
    )
    codec.extract("missing.cim", "never.ppm")
    assert (codec._action, codec._input, codec._output) == (
        Action.EXTRACT,
        "missing.cim",
        "never.ppm",
    )
    assert not Path("never.cim").exists() and not Path("never.ppm").exists()


def test_the_names_compress_and_extract_no_longer_run_anything() -> None:
    """Up to 0.5.0 `compress()` and `extract()` took no arguments and ran at
    once. A caller from then gets told what is missing, not a silent no-op."""
    for name in ("compress", "extract"):
        with pytest.raises(TypeError, match="'input' and 'output'"):
            getattr(Codec(), name)()


def test_builder_methods_return_self() -> None:
    codec = Codec()
    assert codec.compress(input="a.ppm", output="b.cim") is codec
    assert codec.extract(input="b.cim", output="c.ppm") is codec
    assert codec.with_coeff_removal(0.1) is codec
    assert codec.with_input_size(4, 4) is codec


def test_bmp_and_ppm_sources_compress_identically(tmp_path: Path) -> None:
    """The shared RGB contract means the source format cannot change the result."""
    from conftest import gradient_pixels, write_bmp, write_ppm

    pixels = gradient_pixels(16, 16)
    bmp = write_bmp(tmp_path / "same.bmp", 16, 16, pixels)
    ppm = write_ppm(tmp_path / "same.ppm", 16, 16, pixels)

    from_bmp = tmp_path / "from_bmp.cim"
    from_ppm = tmp_path / "from_ppm.cim"
    Codec().compress(input=str(bmp), output=str(from_bmp)).run()
    Codec().compress(input=str(ppm), output=str(from_ppm)).run()

    assert from_bmp.read_bytes() == from_ppm.read_bytes()


def test_ppm_roundtrips_through_the_codec(gradient_ppm: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "p.cim"
    restored = tmp_path / "back.ppm"
    Codec().compress(input=str(gradient_ppm), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()

    before, after = _pixels(gradient_ppm), _pixels(restored)
    assert before.shape == after.shape
    assert np.abs(before - after).mean() < 24


def test_cross_format_conversion_preserves_the_picture(gradient_bmp: Path, tmp_path: Path) -> None:
    """Compress from BMP, extract to PPM: same picture, not flipped or swapped."""
    compressed = tmp_path / "x.cim"
    as_ppm = tmp_path / "out.ppm"
    as_bmp = tmp_path / "out.bmp"

    Codec().compress(input=str(gradient_bmp), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(as_ppm)).run()
    Codec().extract(input=str(compressed), output=str(as_bmp)).run()

    from walsh.image import BMPImage, PPMImage

    ppm_image, bmp_image = PPMImage(), BMPImage()
    ppm_image.load(str(as_ppm))
    bmp_image.load(str(as_bmp))
    assert ppm_image.get_raw_data() == bmp_image.get_raw_data()


def test_unknown_output_format_is_rejected(gradient_bmp: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "u.cim"
    Codec().compress(input=str(gradient_bmp), output=str(compressed)).run()

    codec = Codec().extract(input=str(compressed), output=str(tmp_path / "out.jpg"))
    with pytest.raises(UnsupportedFileFormatError, match="unsupported image format"):
        codec.run()


def test_sample_ppm_compresses_and_survives_the_round_trip(
    sample_ppm: Path, tmp_path: Path
) -> None:
    """The checked-in Blue Marble photo, end to end through the real codec."""
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.ppm"
    Codec().compress(input=str(sample_ppm), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_ppm.stat().st_size

    Codec().extract(input=str(compressed), output=str(restored)).run()

    before, after = _pixels(sample_ppm), _pixels(restored)
    assert before.shape == after.shape
    # A photograph, so less forgiving than the synthetic gradient, but the
    # low-frequency corner still carries the picture.
    assert np.abs(before - after).mean() < 20


def test_tiff_roundtrips_through_the_codec(gradient_tiff: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "t.cim"
    restored = tmp_path / "back.tif"
    Codec().compress(input=str(gradient_tiff), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()

    before, after = _pixels(gradient_tiff), _pixels(restored)
    assert before.shape == after.shape
    assert np.abs(before - after).mean() < 24


def test_every_source_format_compresses_identically(tmp_path: Path) -> None:
    """BMP, PNG, PPM, PAM, TIFF and a bare array of one picture must produce the same .cim."""
    from conftest import (
        gradient_pixels,
        write_bmp,
        write_npy,
        write_pam,
        write_png,
        write_ppm,
        write_tiff,
    )

    pixels = gradient_pixels(16, 16)
    sources = {
        "bmp": write_bmp(tmp_path / "s.bmp", 16, 16, pixels),
        "ppm": write_ppm(tmp_path / "s.ppm", 16, 16, pixels),
        "pam": write_pam(tmp_path / "s.pam", 16, 16, pixels),
        "tif": write_tiff(tmp_path / "s.tif", 16, 16, pixels),
        "npy": write_npy(tmp_path / "s.npy", 16, 16, pixels),
        "png": write_png(tmp_path / "s.png", 16, 16, pixels, filters=range(5)),
    }
    digests = {}
    for name, path in sources.items():
        output = tmp_path / f"{name}.cim"
        Codec().compress(input=str(path), output=str(output)).run()
        digests[name] = output.read_bytes()

    assert len(set(digests.values())) == 1, sorted(digests)


@pytest.mark.parametrize("suffix", [".bmp", ".ppm", ".pam", ".tif", ".npy", ".png"])
def test_extract_to_any_format_gives_the_same_picture(
    gradient_bmp: Path, tmp_path: Path, suffix: str
) -> None:
    compressed = tmp_path / "x.cim"
    Codec().compress(input=str(gradient_bmp), output=str(compressed)).run()

    reference = tmp_path / "ref.bmp"
    target = tmp_path / f"out{suffix}"
    Codec().extract(input=str(compressed), output=str(reference)).run()
    Codec().extract(input=str(compressed), output=str(target)).run()

    np.testing.assert_array_equal(_pixels(reference), _pixels(target))


def test_checked_in_samples_agree_across_containers(
    sample_ppm: Path,
    sample_tiff: Path,
    sample_pam: Path,
    sample_npy: Path,
    sample_png: Path,
    tmp_path: Path,
) -> None:
    """The `earth` samples under data/ are one picture in several containers."""
    digests = []
    for source in (sample_ppm, sample_tiff, sample_pam, sample_npy, sample_png):
        output = tmp_path / f"{source.suffix.lstrip('.')}.cim"
        Codec().compress(input=str(source), output=str(output)).run()
        digests.append(output.read_bytes())

    for other in (sample_tiff, sample_pam, sample_npy, sample_png):
        np.testing.assert_array_equal(_pixels(sample_ppm), _pixels(other))
    assert len(set(digests)) == 1


def test_sample_png_survives_the_round_trip(sample_png: Path, tmp_path: Path) -> None:
    """The one compressed source: the .cim is still smaller than the PNG."""
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.png"
    Codec().compress(input=str(sample_png), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_png.stat().st_size

    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert np.abs(_pixels(sample_png) - _pixels(restored)).mean() < 20


def test_sample_npy_survives_the_round_trip(sample_npy: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.npy"
    Codec().compress(input=str(sample_npy), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_npy.stat().st_size

    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert np.abs(_pixels(sample_npy) - _pixels(restored)).mean() < 20
    assert np.load(restored, allow_pickle=False).shape == (400, 400, 3)


def test_sample_pam_survives_the_round_trip(sample_pam: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.pam"
    Codec().compress(input=str(sample_pam), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_pam.stat().st_size

    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert np.abs(_pixels(sample_pam) - _pixels(restored)).mean() < 20


def test_sample_tiff_survives_the_round_trip(sample_tiff: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.tiff"
    Codec().compress(input=str(sample_tiff), output=str(compressed)).run()
    assert compressed.stat().st_size < sample_tiff.stat().st_size

    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert np.abs(_pixels(sample_tiff) - _pixels(restored)).mean() < 20


def test_slice_cuts_blocks_row_major_and_merge_inverts_it() -> None:
    """A 20x13 plane in 8-blocks: two rows of three blocks, padded on both axes."""
    width, height, block = 20, 13, 8
    plane = np.arange(width * height, dtype=float).reshape(height, width)

    blocks = Codec()._slice(plane.reshape(-1), width, height, block)
    assert len(blocks) == 2 * 3
    assert all(b.shape == (block, block) for b in blocks)
    # Block 1 is the top row, second column: columns 8-15 of rows 0-7.
    np.testing.assert_array_equal(blocks[1], plane[0:8, 8:16])
    # Block 5 is the last one: rows 8-12 and columns 16-19 are picture, and
    # the rest replicates the edge rather than being zeroed.
    np.testing.assert_array_equal(blocks[5][:5, :4], plane[8:13, 16:20])
    for row in blocks[5][5:]:
        np.testing.assert_array_equal(row, blocks[5][4])
    for column in blocks[5][:, 4:].T:
        np.testing.assert_array_equal(column, blocks[5][:, 3])

    np.testing.assert_array_equal(Codec._merge(blocks, width, height), plane)


def test_get_array_is_the_live_contiguous_store() -> None:
    """The primary accessor: a (height, width, 3) uint8 view of the image's own
    pixels, so the codec never builds a Python object per pixel."""
    from conftest import gradient_pixels
    from walsh.image import PPMImage

    image = PPMImage()
    image.set_dimensions(7, 5)
    image.set_raw_data(gradient_pixels(7, 5))

    array = image.get_array()
    assert array.shape == (5, 7, 3)
    assert array.dtype == np.uint8
    assert array.flags.c_contiguous
    assert array.flags.writeable

    array[0, 0] = (9, 8, 7)  # a view, so this is the image
    assert image.get_raw_data()[0] == (9, 8, 7)


def test_set_array_takes_dimensions_from_the_shape() -> None:
    from walsh.image import PPMImage

    pixels = (np.arange(4 * 6 * 3) % 256).astype(np.uint8).reshape(4, 6, 3)
    image = PPMImage()
    image.set_array(pixels)
    assert image.get_dimensions() == (6, 4)
    np.testing.assert_array_equal(image.get_array(), pixels)
    assert np.shares_memory(image.get_array(), pixels), "contiguous uint8 is kept, not copied"


def test_set_array_copies_what_it_cannot_keep() -> None:
    """A non-contiguous view is made contiguous; a read-only buffer, as
    np.frombuffer yields, is copied so get_array stays writable."""
    from walsh.image import PPMImage

    base = (np.arange(4 * 8 * 3) % 256).astype(np.uint8).reshape(4, 8, 3)
    flipped = base[::-1, ::-1]  # non-contiguous
    image = PPMImage()
    image.set_array(flipped)
    np.testing.assert_array_equal(image.get_array(), flipped)
    assert image.get_array().flags.c_contiguous

    frozen = np.frombuffer(base.tobytes(), dtype=np.uint8).reshape(4, 8, 3)
    assert not frozen.flags.writeable
    image.set_array(frozen)
    assert image.get_array().flags.writeable
    np.testing.assert_array_equal(image.get_array(), base)


@pytest.mark.parametrize(
    ("array", "match"),
    [
        (np.zeros((2, 2, 3), dtype=np.float64), "must be uint8"),
        (np.zeros((2, 2, 3), dtype=np.int32), "must be uint8"),
        (np.zeros((2, 2), dtype=np.uint8), r"shaped \(height, width, 3\)"),
        (np.zeros((2, 2, 4), dtype=np.uint8), r"shaped \(height, width, 3\)"),
    ],
)
def test_set_array_rejects_the_wrong_dtype_or_shape(array: np.ndarray, match: str) -> None:
    """Checked, not cast: a silent cast is how a float or a 300 would become
    a wrong pixel with no error anywhere."""
    from walsh.image import PPMImage

    with pytest.raises(ValueError, match=match):
        PPMImage().set_array(array)


def test_raw_data_converters_round_trip_and_do_not_alias() -> None:
    """get_raw_data builds a fresh list each call: the pre-0.4.10 promise that
    mutating it mutated the image is gone, deliberately, and documented."""
    from conftest import gradient_pixels
    from walsh.image import PPMImage

    pixels = gradient_pixels(7, 5)  # odd width, so no accidental alignment
    image = PPMImage()
    image.set_dimensions(7, 5)
    image.set_raw_data(pixels)

    back = image.get_raw_data()
    assert back == pixels
    assert all(type(channel) is int for channel in back[0])

    back[0] = (0, 0, 0)
    assert image.get_raw_data()[0] == pixels[0], "the list is a copy"
    assert image.get_raw_data() is not image.get_raw_data()


def test_the_two_call_build_is_still_transiently_inconsistent() -> None:
    """set_dimensions then set_raw_data must keep working, and save() is still
    where a mismatch is caught -- get_array cannot shape a mismatch either."""
    from walsh.image import PPMImage

    image = PPMImage()
    image.set_dimensions(4, 4)
    image.set_raw_data([(1, 2, 3)] * 3)
    with pytest.raises(ValueError, match="require 16"):
        image.get_array()


def test_merge_takes_a_stack_without_copying_it() -> None:
    """Sub-item 3 of #27: the stack that _slice produced flows through the
    transform and back into _merge as one array, never split and re-stacked."""
    width, height, block = 20, 13, 8
    plane = np.arange(width * height, dtype=float).reshape(height, width)
    stack = Codec()._slice(plane.reshape(-1), width, height, block)
    assert isinstance(stack, np.ndarray) and stack.shape == (6, 8, 8)
    assert stack.flags.c_contiguous

    merged = Codec._merge(stack, width, height)
    np.testing.assert_array_equal(merged, plane)
    # and a plain list of blocks still works, for direct callers
    np.testing.assert_array_equal(Codec._merge(list(stack), width, height), plane)


def test_extract_fills_channels_without_blocks_with_neutral_values(tmp_path: Path) -> None:
    """A .cim declaring no blocks decodes to zero luma and centred chroma: black."""
    import struct

    source = tmp_path / "empty.cim"
    source.write_bytes(struct.pack("<II", 5, 3) + struct.pack("<HHH", 8, 4, 0) * 3)
    restored = tmp_path / "empty.ppm"

    Codec().extract(input=str(source), output=str(restored)).run()

    pixels = _pixels(restored)
    assert pixels.shape == (15, 3)
    assert not pixels.any()


@pytest.mark.parametrize(("width", "height"), [(17, 16), (18, 18), (16, 19), (23, 21), (1, 1)])
def test_a_flat_colour_survives_at_any_size(width: int, height: int, tmp_path: Path) -> None:
    """The padding regression, at its most visible.

    A single flat colour has no detail to lose, so the codec should return it
    almost exactly whatever the dimensions. Before 0.4.3 the padding was zeros,
    which the low-pass reconstruction smeared back over the last columns and
    rows: solid orange came back with a pure green edge, off by 200 of 255.
    1x1 is the extreme case, smaller than every block, so it was entirely
    padding.
    """
    from conftest import write_ppm

    colour = (200, 30, 60)
    source = write_ppm(tmp_path / "flat.ppm", width, height, [colour] * (width * height))
    compressed = tmp_path / "flat.cim"
    restored = tmp_path / "flat-back.ppm"

    Codec().compress(input=str(source), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()

    image = PPMImage()
    image.load(str(restored))
    got = np.asarray(image.get_raw_data(), dtype=int)
    assert np.abs(got - np.asarray(colour)).max() <= 2, f"worst pixel {got.max()}"


# -- the .cim container's 16-bit block count -------------------------------


@pytest.mark.parametrize(
    ("width", "height", "block"),
    [(13, 7, 8), (16, 16, 8), (2040, 2040, 8), (2048, 2048, 8), (4000, 2000, 8), (4032, 3024, 16)],
)
def test_block_count_is_predictable_without_slicing(width: int, height: int, block: int) -> None:
    """The check runs before any work, so its arithmetic must match _slice exactly."""
    plane = np.zeros(width * height)
    assert Codec._count_blocks(width, height, block) == len(
        Codec()._slice(plane, width, height, block)
    )


def test_an_image_at_the_limit_is_accepted() -> None:
    """2040x2040 is 255x255 = 65025 luma blocks, just inside the 16-bit field."""
    Codec()._check_fits_the_container(2040, 2040)


@pytest.mark.parametrize(
    ("width", "height", "channel", "blocks"),
    [
        (2048, 2048, "luma", 65536),  # 256x256, one block over
        (4032, 3024, "luma", 190512),  # a stock 12 MP phone photo
        (4000, 2000, "luma", 125000),  # non-square, so the check cannot assume squares
    ],
)
def test_an_image_past_the_limit_is_rejected_with_a_useful_message(
    width: int, height: int, channel: str, blocks: int
) -> None:
    """It used to surface as struct.error naming neither channel nor limit."""
    with pytest.raises(UnsupportedFileFormatError) as caught:
        Codec()._check_fits_the_container(width, height)

    message = str(caught.value)
    assert f"{width}x{height}" in message
    assert f"{blocks} {channel} blocks" in message
    assert "65535" in message
    assert "--y-block-size" in message


def test_the_block_size_the_message_suggests_actually_works() -> None:
    """An unactionable error is barely better than the struct one it replaced."""
    import re

    with pytest.raises(UnsupportedFileFormatError) as caught:
        Codec()._check_fits_the_container(4032, 3024)

    suggested = re.search(r"--y-block-size (\d+)", str(caught.value))
    assert suggested, str(caught.value)
    Codec(y_block_size=int(suggested.group(1)))._check_fits_the_container(4032, 3024)


def test_the_chroma_channel_is_checked_too_and_names_its_own_option() -> None:
    """Luma binds at the defaults, so chroma needs a configuration to surface."""
    codec = Codec(y_block_size=64, cb_block_size=8, cr_block_size=8)
    with pytest.raises(UnsupportedFileFormatError, match="Cb blocks"):
        codec._check_fits_the_container(2048, 2048)


def test_compress_rejects_an_oversized_image_before_touching_the_output(
    tmp_path: Path,
) -> None:
    """End to end, using a 1-pixel block so the boundary costs 65536 pixels, not 4 million."""
    from conftest import gradient_pixels, write_ppm

    source = write_ppm(tmp_path / "big.ppm", 256, 256, gradient_pixels(256, 256))
    output = tmp_path / "existing.cim"
    output.write_bytes(b"A PREVIOUS ENCODE")

    codec = Codec(y_block_size=1, packed_block_size=1)
    with pytest.raises(UnsupportedFileFormatError, match=r"too large for the \.cim container"):
        codec.compress(input=str(source), output=str(output)).run()

    assert output.read_bytes() == b"A PREVIOUS ENCODE"


def test_the_same_image_compresses_with_a_larger_block(tmp_path: Path) -> None:
    """The documented escape hatch has to stay real."""
    from conftest import gradient_pixels, write_ppm

    source = write_ppm(tmp_path / "big.ppm", 256, 256, gradient_pixels(256, 256))
    output = tmp_path / "out.cim"

    Codec(y_block_size=2, packed_block_size=1).compress(input=str(source), output=str(output)).run()

    assert output.stat().st_size > 0


def test_merge_rejects_a_block_count_that_cannot_tile_the_plane() -> None:
    """Before 0.4.7 a surplus was dropped silently and a shortfall raised a
    numpy reshape error naming no dimension. The reader now guarantees the
    count for any file it accepts; this covers direct callers."""
    blocks = [np.full((4, 4), i, dtype=float) for i in range(3)]  # 6x6 needs 4
    with pytest.raises(ValueError, match="3 blocks cannot tile a 6x6 plane"):
        Codec._merge(blocks, width=6, height=6)


def test_merge_takes_its_row_count_from_the_height_not_the_block_count() -> None:
    """A 6x6 plane in 4-blocks is 2x2 blocks; all four must be placed."""
    blocks = [np.full((4, 4), i, dtype=float) for i in range(4)]
    merged = Codec._merge(blocks, width=6, height=6)
    assert merged.shape == (6, 6)
    assert merged[0, 0] == 0 and merged[0, 5] == 1 and merged[5, 0] == 2 and merged[5, 5] == 3


# -- block geometry is validated once, in Codec.__init__, before any file is touched --


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"y_block_size": 0}, "y_block_size must be a positive power of two, got 0"),
        ({"cb_block_size": 6}, "cb_block_size must be a positive power of two"),
        ({"cr_block_size": -8}, "cr_block_size must be a positive power of two"),
        ({"y_block_size": 256, "cb_block_size": 256, "cr_block_size": 256}, "at most 128"),
        ({"packed_block_size": 0}, "packed_block_size must be between 1 and"),
        ({"packed_block_size": 16}, r"smallest block edge \(8\), got 16"),
        ({"y_block_size": 4, "packed_block_size": 6}, r"\(4\), got 6"),
    ],
)
def test_invalid_block_geometry_is_rejected_at_construction(
    kwargs: dict[str, int], match: str
) -> None:
    """Each of these used to fail late: a bare ZeroDivisionError, a file the
    reader refuses at exit 0, or a white picture returned mid-grey at exit 0."""
    with pytest.raises(ValueError, match=match):
        Codec(**kwargs)


@pytest.mark.parametrize("packed", [3, 6])
def test_packed_size_need_not_be_a_power_of_two(
    packed: int, gradient_ppm: Path, tmp_path: Path
) -> None:
    """3 and 6 compress and extract cleanly today; requiring a power of two
    here would be a regression, not a fix."""
    compressed, restored = tmp_path / "p.cim", tmp_path / "p.ppm"
    codec = Codec(packed_block_size=packed)
    codec.compress(input=str(gradient_ppm), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert _pixels(restored).shape == _pixels(gradient_ppm).shape


def test_packed_may_equal_the_smallest_edge(gradient_ppm: Path, tmp_path: Path) -> None:
    """--y-block-size 16 --chroma-block-size 16 --packed-block-size 16 is lossless
    cropping and must stay legal: the bound is the smallest edge, not a constant."""
    compressed, restored = tmp_path / "p.cim", tmp_path / "p.ppm"
    codec = Codec(y_block_size=16, cb_block_size=16, cr_block_size=16, packed_block_size=16)
    codec.compress(input=str(gradient_ppm), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert np.abs(_pixels(restored) - _pixels(gradient_ppm)).max() <= 1


def test_the_largest_legal_edge_keeps_white_white(tmp_path: Path) -> None:
    """Edge 128 is the last size whose DC coefficient fits int16 (255 * 128 <
    32767); a white image survives it. Edge 256 is refused rather than
    returned mid-grey."""
    from conftest import write_ppm
    from walsh.image import MAX_BLOCK_SIZE

    assert MAX_BLOCK_SIZE == 128
    source = write_ppm(tmp_path / "white.ppm", 256, 256, [(255, 255, 255)] * (256 * 256))
    compressed, restored = tmp_path / "w.cim", tmp_path / "w.ppm"
    codec = Codec(y_block_size=128, cb_block_size=128, cr_block_size=128)
    codec.compress(input=str(source), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()
    assert _pixels(restored).min() >= 254

    with pytest.raises(ValueError, match="at most 128"):
        Codec(y_block_size=256, cb_block_size=256, cr_block_size=256)


# --- coefficient removal is plumbed through, and bites (#26) --------------------


def _stored_coefficients(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes()[26:], dtype="<i2")


def test_coeff_removal_leaves_fewer_coefficients_the_higher_it_is(
    gradient_bmp: Path, tmp_path: Path
) -> None:
    """The one assertion that catches the threshold being dropped on the way
    from the builder to the transform. The gradient's 96 stored coefficients
    include 30 non-zero ones; a threshold below all of them changes nothing,
    which is why a byte comparison at 0.5 would assert nothing, and then the
    ladder is 30, 14, 5."""
    outputs: dict[float | None, Path] = {}
    for coeff in (None, 0.5, 100.0, 500.0):
        output = tmp_path / f"{coeff}.cim"
        Codec().with_coeff_removal(coeff).compress(
            input=str(gradient_bmp), output=str(output)
        ).run()
        outputs[coeff] = output

    assert outputs[0.5].read_bytes() == outputs[None].read_bytes()
    assert outputs[100.0].read_bytes() != outputs[None].read_bytes()
    assert outputs[500.0].read_bytes() != outputs[100.0].read_bytes()
    assert {coeff: len(path.read_bytes()) for coeff, path in outputs.items()} == dict.fromkeys(
        outputs, 218
    ), "the threshold zeroes coefficients; it never changes the layout"

    non_zero = {
        coeff: int((_stored_coefficients(path) != 0).sum()) for coeff, path in outputs.items()
    }
    assert non_zero == {None: 30, 0.5: 30, 100.0: 14, 500.0: 5}
    survivors = _stored_coefficients(outputs[500.0])
    survivors = survivors[survivors != 0]
    assert np.all(np.abs(survivors) >= 500), "every survivor is at or above the threshold"
