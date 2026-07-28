from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from walsh.image import BMPImage
from walsh.task import Action, Task


def _pixels(path: Path) -> np.ndarray:
    image = BMPImage()
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
