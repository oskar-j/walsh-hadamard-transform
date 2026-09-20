"""The checked-in sample outputs are the codec's contract across releases.

Every file in `data/` that the codec produced is regenerated in the same commit
as any deliberate change to the codec, and the CHANGELOG says what changed and
by how much. Since 0.4.12 (#39) the transform's arithmetic is exact, so both
directions are byte-exact on every platform and BLAS library, and these tests
are plain equality again: 0.4.11 had to allow a one-step rounding difference
between Apple's Accelerate and scipy-openblas, and the wheel smoke in CI runs
the same comparison with `cmp`. `data/cim/transformed_earth.cim` was written by
0.4.12 on macOS; CI proves the Linux runner produces the same bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import Sample
from walsh.image import PNGImage, PPMImage
from walsh.task import Task

#: One picture in every container the codec reads. The PNG was written by
#: Netpbm's `pnmtopng`, so it is compressed and filtered by libpng, and it must
#: still reach the transform as exactly the pixels of the others.
SOURCES = ("earth.ppm", "earth.tiff", "earth.pam", "earth.npy", "earth.pkl", "earth.png")


@pytest.mark.parametrize("source", [*SOURCES])
def test_every_source_container_compresses_to_the_checked_in_cim(
    source: str, sample: Sample, tmp_path: Path
) -> None:
    output = tmp_path / "earth.cim"
    Task().compress(input=str(sample(source)), output=str(output)).run()
    assert output.read_bytes() == sample("transformed_earth.cim").read_bytes()


def test_every_source_container_agrees_with_every_other_exactly(
    sample: Sample, tmp_path: Path
) -> None:
    """Redundant with the test above while that one holds, and the separate
    signal when it does not: the source format cannot reach the arithmetic."""
    digests = set()
    for source in SOURCES:
        output = tmp_path / f"{source}.cim"
        Task().compress(input=str(sample(source)), output=str(output)).run()
        digests.add(output.read_bytes())
    assert len(digests) == 1


@pytest.mark.parametrize(
    "target",
    ["recreated.ppm", "recreated.tiff", "recreated.pam", "recreated.npy", "recreated.pkl"],
)
def test_the_checked_in_cim_extracts_to_every_checked_in_reconstruction(
    target: str, sample: Sample, tmp_path: Path
) -> None:
    output = tmp_path / target
    Task().extract(input=str(sample("transformed_earth.cim")), output=str(output)).run()
    assert output.read_bytes() == sample(target).read_bytes()


def test_the_bmp_sample_round_trips_to_its_checked_in_reconstruction(
    sample: Sample, tmp_path: Path
) -> None:
    compressed = tmp_path / "image.cim"
    restored = tmp_path / "recreated.bmp"
    Task().compress(input=str(sample("image.bmp")), output=str(compressed)).run()
    Task().extract(input=str(compressed), output=str(restored)).run()
    assert restored.read_bytes() == sample("recreated.bmp").read_bytes()


def test_the_checked_in_png_is_pinned_by_its_pixels_not_its_bytes(
    sample: Sample, tmp_path: Path
) -> None:
    """A PNG is a zlib stream, and DEFLATE output may differ from one zlib
    build to the next (zlib-ng, which some distributions ship as zlib, already
    does). So `recreated.png` is the one checked-in output compared by what it
    decodes to: exactly the pixels of `recreated.ppm`, from the file in the
    repository and from a PNG written just now."""
    expected = PPMImage()
    expected.load(str(sample("recreated.ppm")))

    fresh = tmp_path / "recreated.png"
    Task().extract(input=str(sample("transformed_earth.cim")), output=str(fresh)).run()

    for path in (sample("recreated.png"), fresh):
        decoded = PNGImage()
        decoded.load(str(path))
        assert np.array_equal(decoded.get_array(), expected.get_array()), path


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("earth.ppm", "recreated.ppm"),
        ("earth.tiff", "recreated.tiff"),
        ("earth.pam", "recreated.pam"),
        ("earth.npy", "recreated.npy"),
        ("earth.pkl", "recreated.pkl"),
        ("image.bmp", "recreated.bmp"),
    ],
)
def test_compressing_straight_to_a_picture_writes_the_checked_in_reconstruction(
    source: str, target: str, sample: Sample, tmp_path: Path
) -> None:
    """`compress(input=picture, output=picture)` skips the .cim file (0.5.1),
    and must write what the two steps write, which is what is checked in."""
    output = tmp_path / target
    Task().compress(input=str(sample(source)), output=str(output)).run()
    assert output.read_bytes() == sample(target).read_bytes()


def test_compressing_straight_to_a_png_writes_the_checked_in_pixels(
    sample: Sample, tmp_path: Path
) -> None:
    """The PNG again by its pixels, since its bytes follow the zlib build."""
    expected = PPMImage()
    expected.load(str(sample("recreated.ppm")))

    output = tmp_path / "direct.png"
    Task().compress(input=str(sample("earth.png")), output=str(output)).run()
    decoded = PNGImage()
    decoded.load(str(output))
    assert np.array_equal(decoded.get_array(), expected.get_array())
