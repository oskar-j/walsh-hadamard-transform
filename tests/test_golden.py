"""The checked-in sample outputs are the codec's contract across releases.

Every earlier release claimed byte-identical output and verified it by hand;
this pins it. `data/transformed_earth.cim` was written by 0.3.1, and nothing
since has been allowed to change it. If a change here is deliberate, the
samples must be regenerated in the same commit and the CHANGELOG must say so.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import DATA_DIR
from walsh.task import Task

TRANSFORMED_SHA256 = "29942c8437509dcb8f631309be2cfb692a85246bf9f1969f1704b924a2582980"


def _sample(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():  # pragma: no cover - the sdist ships no samples
        pytest.skip(f"sample missing: {path}")
    return path


@pytest.mark.parametrize("source", ["earth.ppm", "earth.tiff", "earth.pam", "earth.npy"])
def test_every_source_container_compresses_to_the_checked_in_cim(
    source: str, tmp_path: Path
) -> None:
    output = tmp_path / "earth.cim"
    Task().with_action("compress").with_input(str(_sample(source))).with_output(str(output)).run()
    produced = output.read_bytes()
    assert produced == _sample("transformed_earth.cim").read_bytes()
    assert hashlib.sha256(produced).hexdigest() == TRANSFORMED_SHA256


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
