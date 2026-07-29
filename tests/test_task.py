from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import BMPImage, reader_for
from walsh.task import Action, Task


def _pixels(path: Path) -> np.ndarray:
    image = reader_for(path)
    image.load(str(path))
    return np.asarray(image.get_raw_data(), dtype=float)


def test_compress_then_extract_restores_shape_and_approximate_content(
    gradient_bmp: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "out.cim"
    restored = tmp_path / "back.bmp"

    Task().with_action("compress").with_input(str(gradient_bmp)).with_output(str(compressed)).run()
    assert compressed.exists()

    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()

    before, after = _pixels(gradient_bmp), _pixels(restored)
    assert before.shape == after.shape
    # Lossy, but a smooth gradient must survive with a small mean error.
    assert np.abs(before - after).mean() < 24


def test_compression_actually_shrinks_the_data(sample_bmp: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "sample.cim"
    Task().with_action("compress").with_input(str(sample_bmp)).with_output(str(compressed)).run()
    assert compressed.stat().st_size < sample_bmp.stat().st_size


def test_smaller_packed_block_size_produces_a_smaller_file(
    gradient_bmp: Path, tmp_path: Path
) -> None:
    sizes = {}
    for packed in (2, 4):
        output = tmp_path / f"packed{packed}.cim"
        Task(packed_block_size=packed).with_action("compress").with_input(
            str(gradient_bmp)
        ).with_output(str(output)).run()
        sizes[packed] = output.stat().st_size
    assert sizes[2] < sizes[4]


def test_non_multiple_dimensions_are_padded_and_cropped_back(
    tmp_path: Path,
) -> None:
    """A 20x20 image is not a multiple of the 16-wide chroma block."""
    from conftest import write_bmp

    width = height = 20
    pixels = [(x * 12 % 256, y * 12 % 256, 128) for y in range(height) for x in range(width)]
    source = write_bmp(tmp_path / "odd.bmp", width, height, pixels)

    compressed = tmp_path / "odd.cim"
    restored = tmp_path / "odd-back.bmp"
    Task().with_action("compress").with_input(str(source)).with_output(str(compressed)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()

    image = BMPImage()
    image.load(str(restored))
    assert image.get_dimensions() == (width, height)
    assert len(image.get_raw_data()) == width * height


def test_padding_size_helper() -> None:
    assert Task._get_padding_size(16, 8) == 0
    assert Task._get_padding_size(17, 8) == 7
    assert Task._get_padding_size(20, 16) == 12


def test_merge_crops_padding() -> None:
    blocks = [np.full((4, 4), i, dtype=float) for i in range(4)]
    merged = Task._merge(blocks, width=6, height=6)
    assert merged.shape == (6, 6)


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        Task().with_action("sharpen")


def test_run_without_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="no action selected"):
        Task().run()


def test_action_accepts_enum_and_string() -> None:
    assert Task().with_action(Action.COMPRESS)._action is Action.COMPRESS
    assert Task().with_action("extract")._action is Action.EXTRACT


def test_builder_methods_return_self() -> None:
    task = Task()
    assert task.with_input("a") is task
    assert task.with_output("b") is task
    assert task.with_coeff_removal(0.1) is task
    assert task.with_action("compress") is task


def test_bmp_and_ppm_sources_compress_identically(tmp_path: Path) -> None:
    """The shared RGB contract means the source format cannot change the result."""
    from conftest import gradient_pixels, write_bmp, write_ppm

    pixels = gradient_pixels(16, 16)
    bmp = write_bmp(tmp_path / "same.bmp", 16, 16, pixels)
    ppm = write_ppm(tmp_path / "same.ppm", 16, 16, pixels)

    from_bmp = tmp_path / "from_bmp.cim"
    from_ppm = tmp_path / "from_ppm.cim"
    Task().with_action("compress").with_input(str(bmp)).with_output(str(from_bmp)).run()
    Task().with_action("compress").with_input(str(ppm)).with_output(str(from_ppm)).run()

    assert from_bmp.read_bytes() == from_ppm.read_bytes()


def test_ppm_roundtrips_through_the_codec(gradient_ppm: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "p.cim"
    restored = tmp_path / "back.ppm"
    Task().with_action("compress").with_input(str(gradient_ppm)).with_output(str(compressed)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()

    before, after = _pixels(gradient_ppm), _pixels(restored)
    assert before.shape == after.shape
    assert np.abs(before - after).mean() < 24


def test_cross_format_conversion_preserves_the_picture(gradient_bmp: Path, tmp_path: Path) -> None:
    """Compress from BMP, extract to PPM: same picture, not flipped or swapped."""
    compressed = tmp_path / "x.cim"
    as_ppm = tmp_path / "out.ppm"
    as_bmp = tmp_path / "out.bmp"

    Task().with_action("compress").with_input(str(gradient_bmp)).with_output(str(compressed)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(as_ppm)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(as_bmp)).run()

    from walsh.image import BMPImage, PPMImage

    ppm_image, bmp_image = PPMImage(), BMPImage()
    ppm_image.load(str(as_ppm))
    bmp_image.load(str(as_bmp))
    assert ppm_image.get_raw_data() == bmp_image.get_raw_data()


def test_unknown_output_format_is_rejected(gradient_bmp: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "u.cim"
    Task().with_action("compress").with_input(str(gradient_bmp)).with_output(str(compressed)).run()

    task = Task().with_action("extract").with_input(str(compressed))
    with pytest.raises(UnsupportedFileFormatError, match="unsupported image format"):
        task.with_output(str(tmp_path / "out.jpg")).run()


def test_sample_ppm_compresses_and_survives_the_round_trip(
    sample_ppm: Path, tmp_path: Path
) -> None:
    """The checked-in Blue Marble photo, end to end through the real codec."""
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.ppm"
    Task().with_action("compress").with_input(str(sample_ppm)).with_output(str(compressed)).run()
    assert compressed.stat().st_size < sample_ppm.stat().st_size

    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()

    before, after = _pixels(sample_ppm), _pixels(restored)
    assert before.shape == after.shape
    # A photograph, so less forgiving than the synthetic gradient, but the
    # low-frequency corner still carries the picture.
    assert np.abs(before - after).mean() < 20


def test_tiff_roundtrips_through_the_codec(gradient_tiff: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "t.cim"
    restored = tmp_path / "back.tif"
    Task().with_action("compress").with_input(str(gradient_tiff)).with_output(str(compressed)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()

    before, after = _pixels(gradient_tiff), _pixels(restored)
    assert before.shape == after.shape
    assert np.abs(before - after).mean() < 24


def test_every_source_format_compresses_identically(tmp_path: Path) -> None:
    """BMP, PPM and TIFF of one picture must produce the same .cim."""
    from conftest import gradient_pixels, write_bmp, write_ppm, write_tiff

    pixels = gradient_pixels(16, 16)
    sources = {
        "bmp": write_bmp(tmp_path / "s.bmp", 16, 16, pixels),
        "ppm": write_ppm(tmp_path / "s.ppm", 16, 16, pixels),
        "tif": write_tiff(tmp_path / "s.tif", 16, 16, pixels),
    }
    digests = {}
    for name, path in sources.items():
        output = tmp_path / f"{name}.cim"
        Task().with_action("compress").with_input(str(path)).with_output(str(output)).run()
        digests[name] = output.read_bytes()

    assert digests["bmp"] == digests["ppm"] == digests["tif"]


@pytest.mark.parametrize("suffix", [".bmp", ".ppm", ".tif"])
def test_extract_to_any_format_gives_the_same_picture(
    gradient_bmp: Path, tmp_path: Path, suffix: str
) -> None:
    compressed = tmp_path / "x.cim"
    Task().with_action("compress").with_input(str(gradient_bmp)).with_output(str(compressed)).run()

    reference = tmp_path / "ref.bmp"
    target = tmp_path / f"out{suffix}"
    Task().with_action("extract").with_input(str(compressed)).with_output(str(reference)).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(target)).run()

    np.testing.assert_array_equal(_pixels(reference), _pixels(target))


def test_checked_in_samples_agree_across_containers(
    sample_ppm: Path, sample_tiff: Path, tmp_path: Path
) -> None:
    """data/earth.ppm and data/earth.tiff are one picture in two containers."""
    digests = []
    for source in (sample_ppm, sample_tiff):
        output = tmp_path / f"{source.suffix.lstrip('.')}.cim"
        Task().with_action("compress").with_input(str(source)).with_output(str(output)).run()
        digests.append(output.read_bytes())

    np.testing.assert_array_equal(_pixels(sample_ppm), _pixels(sample_tiff))
    assert digests[0] == digests[1]


def test_sample_tiff_survives_the_round_trip(sample_tiff: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "earth.cim"
    restored = tmp_path / "earth.tiff"
    Task().with_action("compress").with_input(str(sample_tiff)).with_output(str(compressed)).run()
    assert compressed.stat().st_size < sample_tiff.stat().st_size

    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()
    assert np.abs(_pixels(sample_tiff) - _pixels(restored)).mean() < 20
