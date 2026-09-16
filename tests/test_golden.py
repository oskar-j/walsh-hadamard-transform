"""The checked-in sample outputs are the codec's contract across releases.

Every file in `data/` that the codec produced is regenerated in the same commit
as any deliberate change to the codec, and the CHANGELOG says what changed and
by how much. Since 0.4.12 (#39) the transform's arithmetic is exact, so both
directions are byte-exact on every platform and BLAS library, and these tests
are plain equality again: 0.4.11 had to allow a one-step rounding difference
between Apple's Accelerate and scipy-openblas, and the wheel smoke in CI runs
the same comparison with `cmp`. `data/transformed_earth.cim` was written by
0.4.12 on macOS; CI proves the Linux runner produces the same bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import DATA_DIR
from walsh.task import Task


def _sample(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():  # pragma: no cover - the sdist ships no samples
        pytest.skip(f"sample image missing: {path}")
    return path


@pytest.mark.parametrize("source", ["earth.ppm", "earth.tiff", "earth.pam", "earth.npy"])
def test_every_source_container_compresses_to_the_checked_in_cim(
    source: str, tmp_path: Path
) -> None:
    output = tmp_path / "earth.cim"
    Task().with_action("compress").with_input(str(_sample(source))).with_output(str(output)).run()
    assert output.read_bytes() == _sample("transformed_earth.cim").read_bytes()


def test_every_source_container_agrees_with_every_other_exactly(tmp_path: Path) -> None:
    """Redundant with the test above while that one holds, and the separate
    signal when it does not: the source format cannot reach the arithmetic."""
    digests = set()
    for source in ("earth.ppm", "earth.tiff", "earth.pam", "earth.npy"):
        output = tmp_path / f"{source}.cim"
        Task().with_action("compress").with_input(str(_sample(source))).with_output(
            str(output)
        ).run()
        digests.add(output.read_bytes())
    assert len(digests) == 1


@pytest.mark.parametrize(
    "target", ["recreated.ppm", "recreated.tiff", "recreated.pam", "recreated.npy"]
)
def test_the_checked_in_cim_extracts_to_every_checked_in_reconstruction(
    target: str, tmp_path: Path
) -> None:
    output = tmp_path / target
    Task().with_action("extract").with_input(str(_sample("transformed_earth.cim"))).with_output(
        str(output)
    ).run()
    assert output.read_bytes() == _sample(target).read_bytes()


def test_the_bmp_sample_round_trips_to_its_checked_in_reconstruction(tmp_path: Path) -> None:
    compressed = tmp_path / "image.cim"
    restored = tmp_path / "recreated.bmp"
    Task().with_action("compress").with_input(str(_sample("image.bmp"))).with_output(
        str(compressed)
    ).run()
    Task().with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()
    assert restored.read_bytes() == _sample("recreated.bmp").read_bytes()
