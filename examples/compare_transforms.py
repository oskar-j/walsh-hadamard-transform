"""Compare block transforms at the same geometry, through the whole pipeline.

``Task(transform=...)`` swaps the transform and nothing else: the colour
conversion, the padding, the crop to the low-frequency corner and the ``.cim``
container stay as they are. The file size depends only on the geometry, so
every row of the table below is the same number of bytes and the dB columns
are a like-for-like comparison of how much picture each transform packs into
its first few coefficients.

Three transforms ship with the package and are selected by name. The fourth
column is written here, to show what a transform of your own takes: subclass
``MatrixTransform`` and return an orthonormal matrix.

Needs numpy only. Run from the repository root::

    python examples/compare_transforms.py [image]

A file written with anything but the default transform is a ``.cim`` in name
only: the container does not record the transform, so it must be extracted by
a ``Task`` given the same one. This script keeps its files in a temporary
directory for that reason.
"""

from __future__ import annotations

import sys
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from walsh import MatrixTransform, Task, Transform, reader_for

Block = npt.NDArray[np.float64]

DATA = Path(__file__).resolve().parent.parent / "data"

#: Coefficients kept per axis. 3 and 6 are there on purpose: at a power of two
#: the first Walsh functions and the first Haar functions span the same
#: piecewise-constant subspace, so those two transforms tie exactly.
KEPT_PER_AXIS = (2, 3, 4, 6, 8)


class HartleyTransform(MatrixTransform):
    """The discrete Hartley transform: a real-valued cousin of the Fourier
    transform whose matrix, ``cas(2 * pi * i * k / n) / sqrt(n)`` with
    ``cas = cos + sin``, is symmetric and its own inverse."""

    @staticmethod
    @cache
    def _matrix(size: int) -> Block:
        angle = 2 * np.pi * np.outer(np.arange(size), np.arange(size)) / size
        return (np.cos(angle) + np.sin(angle)) / np.sqrt(size)

    def matrix(self, size: int) -> Block:
        return self._matrix(size)


#: Column heading -> what to hand ``Task(transform=...)``: a name or an instance.
TRANSFORMS: dict[str, Transform | str] = {
    "Walsh-Hadamard": "walsh",
    "DCT-II": "dct",
    "Haar": "haar",
    "Hartley (custom)": HartleyTransform(),
}


def pixels(path: Path) -> npt.NDArray[np.float64]:
    image = reader_for(str(path))
    image.load(str(path))
    return image.get_array().astype(np.float64)


def psnr(original: npt.NDArray[np.float64], restored: npt.NDArray[np.float64]) -> float:
    mse = float(np.mean((original - restored) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def round_trip(
    source: Path, transform: Transform | str, packed: int, workdir: Path
) -> tuple[float, int]:
    compressed = workdir / "out.cim"
    restored = workdir / f"back{source.suffix}"

    def task() -> Task:
        return Task(packed_block_size=packed, transform=transform)

    task().compress(input=str(source), output=str(compressed)).run()
    task().extract(input=str(compressed), output=str(restored)).run()
    return psnr(pixels(source), pixels(restored)), compressed.stat().st_size


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "ppm" / "earth.ppm"
    print(f"{source.name}, luma blocks of 8, chroma blocks of 16\n")
    print(f"{'kept per axis':>13}  {'bytes':>9}  " + "  ".join(f"{n:>16}" for n in TRANSFORMS))
    with tempfile.TemporaryDirectory() as tmp:
        for packed in KEPT_PER_AXIS:
            results = [round_trip(source, t, packed, Path(tmp)) for t in TRANSFORMS.values()]
            sizes = {size for _, size in results}
            assert len(sizes) == 1, "the size depends on the geometry alone"
            print(
                f"{packed:>13}  {sizes.pop():>9,}  "
                + "  ".join(f"{quality:>13.2f} dB" for quality, _ in results)
            )


if __name__ == "__main__":
    main()
