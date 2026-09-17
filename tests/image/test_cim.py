"""The ``.cim`` container: header, block descriptions, int16 coefficients."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from walsh.image import (
    BlockDescription,
    CustomizableImage,
    UnsupportedFileFormatError,
)


def test_cim_roundtrip_keeps_only_the_packed_corner(tmp_path: Path) -> None:
    original, packed = 8, 4
    block = np.arange(original * original, dtype=np.float64).reshape(original, original)

    image = CustomizableImage()
    image.set_dimensions(8, 8)
    description = BlockDescription(original, packed, 1)
    image.set_descriptions(description, description, description)
    image.set_data([block], [block], [block])

    path = tmp_path / "out.cim"
    image.save(str(path))

    restored = CustomizableImage.load(str(path)).get_y_data()
    assert len(restored) == 1
    # The kept corner survives; everything outside it comes back as zero.
    np.testing.assert_allclose(restored[0][:packed, :packed], block[:packed, :packed])
    np.testing.assert_allclose(restored[0][packed:, :], 0)
    np.testing.assert_allclose(restored[0][:, packed:], 0)


def test_cim_coefficients_are_rounded_not_truncated(tmp_path: Path) -> None:
    description = BlockDescription(2, 2, 1)
    image = CustomizableImage()
    image.set_dimensions(2, 2)
    image.set_descriptions(description, description, description)
    block = np.array([[2.6, -2.6], [0.4, -0.4]])
    image.set_data([block], [block], [block])

    path = tmp_path / "rounded.cim"
    image.save(str(path))

    restored = CustomizableImage.load(str(path)).get_y_data()[0]
    np.testing.assert_array_equal(restored, np.array([[3.0, -3.0], [0.0, -0.0]]))


def test_cim_file_size_matches_the_declared_layout(tmp_path: Path) -> None:
    description = BlockDescription(8, 4, 2)
    image = CustomizableImage()
    image.set_dimensions(16, 8)
    image.set_descriptions(description, description, description)
    blocks = [np.zeros((8, 8)), np.ones((8, 8))]
    image.set_data(blocks, blocks, blocks)

    path = tmp_path / "sized.cim"
    image.save(str(path))

    header = struct.calcsize("<II") + 3 * struct.calcsize("<HHH")
    payload = 3 * 2 * 4 * 4 * 2  # channels * blocks * 4x4 coefficients * int16
    assert path.stat().st_size == header + payload


def test_set_data_before_descriptions_is_an_error() -> None:
    with pytest.raises(ValueError, match="set_descriptions"):
        CustomizableImage().set_data([], [], [])


@pytest.mark.parametrize(
    ("name", "data", "match"),
    [
        ("three bytes", b"\x01\x02\x03", "truncated .cim header"),
        ("header only", b"\x00" * 8, "truncated .cim y block description"),
        (
            # 40x8 in 8-blocks is 5 blocks, so the count is right and only the
            # data is missing; 16x16 would now be caught as a geometry mismatch.
            "declares blocks it does not have",
            struct.pack("<II", 40, 8) + struct.pack("<HHH", 8, 4, 5) * 3,
            "truncated .cim block 0",
        ),
        (
            "block cut in half",
            struct.pack("<II", 8, 8) + struct.pack("<HHH", 8, 4, 1) * 3 + b"\x00" * 16,
            "truncated .cim block 0",
        ),
        (
            "packed size exceeds block size",
            struct.pack("<II", 8, 8) + struct.pack("<HHH", 4, 8, 1) * 3 + b"\x00" * 128,
            "packed size 8 exceeds block size 4",
        ),
        # -- header geometry, 0.4.7 (#22): rejected before anything allocates --
        (
            "zero width",
            struct.pack("<II", 0, 16) + struct.pack("<HHH", 8, 4, 0) * 3,
            "dimensions must be positive, got 0x16",
        ),
        (
            "zero height",
            struct.pack("<II", 16, 0) + struct.pack("<HHH", 8, 4, 0) * 3,
            "dimensions must be positive, got 16x0",
        ),
        (
            "block size 0",
            struct.pack("<II", 8, 8) + struct.pack("<HHH", 0, 0, 0) * 3,
            "y block description: block size 0 is not a positive power of two",
        ),
        (
            "block size not a power of two",
            struct.pack("<II", 8, 8) + struct.pack("<HHH", 6, 4, 4) * 3,
            "block size 6 is not a positive power of two",
        ),
        (
            "packed size 0",
            struct.pack("<II", 16, 16) + struct.pack("<HHH", 8, 0, 4) * 3,
            "packed size must be at least 1, got 0",
        ),
        (
            "too few blocks for the dimensions",
            struct.pack("<II", 16, 16) + struct.pack("<HHH", 8, 4, 1) * 3 + b"\x00" * 96,
            "y block description: 1 blocks declared, but a 16x16 image in 8-pixel blocks has 4",
        ),
        (
            "too many blocks for the dimensions",
            struct.pack("<II", 16, 16) + struct.pack("<HHH", 8, 4, 5) * 3 + b"\x00" * 480,
            "5 blocks declared, but a 16x16 image in 8-pixel blocks has 4",
        ),
        (
            # The bomb from #22: 65535 blocks of 65535x65535 would be 2 PiB.
            "a 26-byte allocation bomb",
            struct.pack("<II", 8, 8)
            + struct.pack("<HHH", 65535, 0, 65535)
            + struct.pack("<HHH", 16, 4, 0) * 2,
            "block size 65535 is not a positive power of two",
        ),
    ],
)
def test_malformed_cim_raises_a_walsh_error(
    tmp_path: Path, name: str, data: bytes, match: str
) -> None:
    """struct.error is not a WalshError, so it must not escape the reader."""
    path = tmp_path / "bad.cim"
    path.write_bytes(data)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        CustomizableImage.load(str(path))


def test_cim_declaring_no_blocks_is_valid(tmp_path: Path) -> None:
    """Zero blocks per channel is legitimate; extract fills them with neutrals."""
    path = tmp_path / "empty.cim"
    path.write_bytes(struct.pack("<II", 8, 8) + struct.pack("<HHH", 8, 4, 0) * 3)

    image = CustomizableImage.load(str(path))
    assert image.get_dimensions() == (8, 8)
    assert image.get_y_data() == []


def test_truncation_names_the_first_incomplete_block(tmp_path: Path) -> None:
    """Three 2x2 blocks declared, one and a half present: block 1 is the short one.

    6x2 in 2-blocks is exactly three blocks, so the header is consistent and
    the failure is the missing data, not the geometry.
    """
    description = struct.pack("<HHH", 2, 2, 3)
    path = tmp_path / "short.cim"
    path.write_bytes(struct.pack("<II", 6, 2) + description * 3 + b"\x00" * 12)

    with pytest.raises(UnsupportedFileFormatError, match=r"block 1: expected 8 bytes, got 4"):
        CustomizableImage.load(str(path))


def test_cim_multi_block_channels_keep_order_and_values(tmp_path: Path) -> None:
    description = BlockDescription(4, 2, 3)
    image = CustomizableImage()
    image.set_dimensions(12, 4)
    image.set_descriptions(description, description, description)
    channels = {
        "y": [np.full((4, 4), value, dtype=float) for value in (1, -2, 3)],
        "cb": [np.full((4, 4), value, dtype=float) for value in (10, 20, 30)],
        "cr": [np.full((4, 4), value, dtype=float) for value in (-7, 0, 7)],
    }
    image.set_data(channels["y"], channels["cb"], channels["cr"])
    path = tmp_path / "three.cim"
    image.save(str(path))

    restored = CustomizableImage.load(str(path))
    for expected, got in (
        (channels["y"], restored.get_y_data()),
        (channels["cb"], restored.get_cb_data()),
        (channels["cr"], restored.get_cr_data()),
    ):
        assert len(got) == 3
        for before, after in zip(expected, got, strict=True):
            np.testing.assert_array_equal(after[:2, :2], before[:2, :2])
            assert not after[2:, :].any()
            assert not after[:, 2:].any()


def test_cim_saves_a_channel_with_no_blocks_as_nothing(tmp_path: Path) -> None:
    """Zero blocks writes zero bytes for that channel, which is what load accepts."""
    description = BlockDescription(8, 4, 0)
    image = CustomizableImage()
    image.set_dimensions(8, 8)
    image.set_descriptions(description, description, description)
    image.set_data([], [], [])

    path = tmp_path / "empty.cim"
    image.save(str(path))

    assert path.stat().st_size == struct.calcsize("<II") + 3 * struct.calcsize("<HHH")
    assert CustomizableImage.load(str(path)).get_y_data() == []


def test_set_descriptions_rejects_a_block_count_the_field_cannot_hold() -> None:
    """The backstop for callers building a container directly rather than via Task.

    Without it the overflow surfaced from struct.pack during save(), naming
    neither the channel nor the limit.
    """
    from walsh.image import MAX_BLOCKS_PER_CHANNEL

    image = CustomizableImage()
    ok = BlockDescription(8, 4, MAX_BLOCKS_PER_CHANNEL)
    too_many = BlockDescription(8, 4, MAX_BLOCKS_PER_CHANNEL + 1)

    image.set_descriptions(ok, ok, ok)  # exactly at the limit is fine

    for position in range(3):
        descriptions = [ok, ok, ok]
        descriptions[position] = too_many
        with pytest.raises(ValueError, match="16-bit field"):
            CustomizableImage().set_descriptions(*descriptions)


def test_a_raster_image_fed_to_the_cim_reader_is_refused_cheaply(sample_bmp: Path) -> None:
    """The repository's own BMP parses as a 1396067650x7 image with a 0-pixel
    block size. Before 0.4.7 that meant a 78 GB allocation and a minute of
    work before failing; now the header alone is enough to say no."""
    with pytest.raises(UnsupportedFileFormatError, match="not a positive power of two"):
        CustomizableImage.load(str(sample_bmp))


def test_the_cim_reader_never_asks_for_the_declared_size_at_once(tmp_path: Path) -> None:
    """A consistent header can still declare far more data than the file holds.

    file.read(n) allocates n bytes before reading, so the old reader asked the
    stream for the whole declared total. This header is consistent (4 blocks of
    8 for 16x16) with packed 8, so it declares 512 bytes and the file has 4;
    the point is that the shortfall is reported per block, from a bounded read.
    """
    path = tmp_path / "short.cim"
    path.write_bytes(struct.pack("<II", 16, 16) + struct.pack("<HHH", 8, 8, 4) * 3 + b"\x00" * 4)
    with pytest.raises(UnsupportedFileFormatError, match=r"block 0: expected 128 bytes, got 4"):
        CustomizableImage.load(str(path))


def test_every_shipped_encoder_configuration_still_loads(tmp_path: Path) -> None:
    """The consistency rules are exact, so they must accept everything the
    encoder can produce: padded dimensions, every packed size, and odd block
    combinations, not just the defaults."""
    from conftest import gradient_pixels, write_ppm
    from walsh.task import Task

    width, height = 13, 7  # a multiple of neither block size
    source = write_ppm(tmp_path / "odd.ppm", width, height, gradient_pixels(width, height))
    configurations = [
        {},
        {"packed_block_size": 2},
        {"packed_block_size": 1},
        {"y_block_size": 16, "cb_block_size": 4, "cr_block_size": 4, "packed_block_size": 3},
        {"y_block_size": 32, "cb_block_size": 32, "packed_block_size": 4},
    ]
    for index, kwargs in enumerate(configurations):
        output = tmp_path / f"{index}.cim"
        Task(**kwargs).with_action("compress").with_input(str(source)).with_output(
            str(output)
        ).run()
        image = CustomizableImage.load(str(output))
        assert image.get_dimensions() == (width, height), kwargs
        assert len(image.get_y_data()) > 0, kwargs


def test_get_stack_is_the_array_form_of_the_block_lists(tmp_path: Path) -> None:
    """get_stack is what the codec consumes; get_*_data are views into it."""
    description = BlockDescription(4, 2, 3)
    image = CustomizableImage()
    image.set_dimensions(12, 4)
    image.set_descriptions(description, description, description)
    blocks = [np.full((4, 4), value, dtype=float) for value in (1, -2, 3)]
    image.set_data(blocks, blocks, blocks)
    path = tmp_path / "s.cim"
    image.save(str(path))

    loaded = CustomizableImage.load(str(path))
    stack = loaded.get_stack("y")
    assert stack.shape == (3, 4, 4)
    assert stack.flags.c_contiguous
    for view, block in zip(loaded.get_y_data(), stack, strict=True):
        assert np.shares_memory(view, stack)
        np.testing.assert_array_equal(view, block)
    assert loaded.get_stack("cb")[:, :2, :2].tolist() == [[[v] * 2] * 2 for v in (1, -2, 3)]
    with pytest.raises(KeyError):
        loaded.get_stack("alpha")


def test_get_stack_of_an_empty_channel_has_zero_length() -> None:
    image = CustomizableImage()
    assert len(image.get_stack("y")) == 0
    assert image.get_y_data() == []


def test_set_data_accepts_a_stack_as_well_as_a_list(tmp_path: Path) -> None:
    """Task hands over one (count, edge, edge) array; direct callers a list."""
    description = BlockDescription(4, 2, 2)
    stack = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    as_array, as_list = CustomizableImage(), CustomizableImage()
    for image, data in ((as_array, stack), (as_list, list(stack))):
        image.set_dimensions(8, 4)
        image.set_descriptions(description, description, description)
        image.set_data(data, data, data)
        image.save(str(tmp_path / f"{id(image)}.cim"))
    outputs = sorted(tmp_path.glob("*.cim"))
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_set_data_rejects_a_block_smaller_than_its_packed_size() -> None:
    """The write-side mirror of the reader's packed > original guard. numpy
    would clamp the crop to the whole block while the header said 16, and
    every reader in this package would then refuse the file."""
    image = CustomizableImage()
    image.set_dimensions(8, 8)
    description = BlockDescription(8, 16, 1)
    image.set_descriptions(description, description, description)
    block = [np.ones((8, 8))]
    with pytest.raises(
        ValueError, match=r"y block of shape \(8, 8\) is smaller than the packed size 16"
    ):
        image.set_data(block, block, block)
