"""The checked-in sample outputs are the codec's contract across releases.

`data/transformed_earth.cim` was written by 0.3.1 on macOS and nothing since
has been allowed to change it. The decode side is byte-exact on every platform
tried. The encode side is byte-exact on any one platform but not across BLAS
implementations: the transform's matrix products round differently in the last
bit under Apple's Accelerate and scipy-openblas, and where a coefficient lands
on an exact half, `np.rint` goes the other way. Measured in 0.4.11, that is
two of 60,000 coefficients, each off by one, and the decoded picture differs in
44 pixels by one level. So the encode check asserts that, not equality: the
header byte for byte, every coefficient within one, and almost none differing.
A deliberate change to the codec must regenerate the samples in the same commit
and say so in the CHANGELOG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import DATA_DIR
from walsh.task import Task

#: Bytes of `.cim` header before the coefficients: `<II` plus three `<HHH`.
HEADER = 26

#: The most coefficients allowed to differ across platforms, as a fraction. The
#: measured figure is 2 in 60,000; 0.1% leaves room without hiding a real change.
CROSS_PLATFORM_FRACTION = 0.001


def _sample(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():  # pragma: no cover - the sdist ships no samples
        pytest.skip(f"sample image missing: {path}")
    return path


def assert_same_encoding(produced: bytes, reference: bytes) -> None:
    """Equal up to the last-bit rounding of the platform's BLAS."""
    assert produced[:HEADER] == reference[:HEADER], "the header must be byte-identical"
    assert len(produced) == len(reference)
    ours = np.frombuffer(produced[HEADER:], dtype="<i2").astype(int)
    theirs = np.frombuffer(reference[HEADER:], dtype="<i2").astype(int)
    delta = np.abs(ours - theirs)
    assert int(delta.max()) <= 1, f"a coefficient differs by {int(delta.max())}, not a rounding"
    differing = int((delta > 0).sum())
    assert differing <= CROSS_PLATFORM_FRACTION * len(theirs), (
        f"{differing} of {len(theirs)} coefficients differ; a BLAS rounding difference "
        f"touches a handful, so this is a real change to the codec"
    )


@pytest.mark.parametrize("source", ["earth.ppm", "earth.tiff", "earth.pam", "earth.npy"])
def test_every_source_container_compresses_to_the_checked_in_cim(
    source: str, tmp_path: Path
) -> None:
    output = tmp_path / "earth.cim"
    Task().with_action("compress").with_input(str(_sample(source))).with_output(str(output)).run()
    assert_same_encoding(output.read_bytes(), _sample("transformed_earth.cim").read_bytes())


def test_every_source_container_agrees_with_every_other_exactly(tmp_path: Path) -> None:
    """Across containers on one platform the bytes must be identical: the
    source format cannot reach the arithmetic."""
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
    """Decoding is byte-exact everywhere: verified on macOS and Linux."""
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
