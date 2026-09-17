"""Compare block transforms at the same geometry, through the whole pipeline.

``Task(transform=...)`` swaps the transform and nothing else: the colour
conversion, the padding, the crop to the low-frequency corner and the ``.cim``
container stay as they are. The file size depends only on the geometry, so
every row of the table below is the same number of bytes and the PSNR column
is a like-for-like comparison of how much picture each transform packs into
its first few coefficients.

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

from walsh import Task, Transform, WalshHadamardTransform, reader_for

Block = npt.NDArray[np.float64]

DATA = Path(__file__).resolve().parent.parent / "data"

#: Coefficients kept per axis. 3 and 6 are there on purpose: at a power of two
#: the first Walsh functions and the first Haar functions span the same
#: piecewise-constant subspace, so those two transforms tie exactly.
KEPT_PER_AXIS = (2, 3, 4, 6, 8)


class MatrixTransform(Transform):
    """A separable orthonormal transform defined by one ``edge x edge`` matrix.

    ``transform`` is ``m @ block @ m.T`` and the inverse is its transpose. Both
    stack methods are overridden, because ``@`` broadcasts over a stack and the
    per-block default would cost a Python call per block.
    """

    def matrix(self, edge: int) -> Block:
        """Return the orthonormal matrix for a block of this edge."""
        raise NotImplementedError

    def transform(self, src: Block) -> Block:
        m = self.matrix(src.shape[-1])
        return m @ src @ m.T

    def inverse_transform(self, src: Block) -> Block:
        m = self.matrix(src.shape[-1])
        return m.T @ src @ m

    def transform_stack(self, stack: Block) -> Block:
        return self.transform(stack)

    def inverse_transform_stack(self, stack: Block) -> Block:
        return self.inverse_transform(stack)


class DiscreteCosineTransform(MatrixTransform):
    """The orthonormal DCT-II, the transform inside JPEG."""

    @staticmethod
    @cache
    def _matrix(edge: int) -> Block:
        k = np.arange(edge)[:, None]
        i = np.arange(edge)[None, :]
        m = np.cos(np.pi * (2 * i + 1) * k / (2 * edge)) * np.sqrt(2 / edge)
        m[0] /= np.sqrt(2)
        return m

    def matrix(self, edge: int) -> Block:
        return self._matrix(edge)


class HaarTransform(MatrixTransform):
    """The orthonormal Haar wavelet transform, rows from coarse to fine."""

    @staticmethod
    @cache
    def _matrix(edge: int) -> Block:
        m = np.ones((1, 1))
        while m.shape[0] < edge:
            coarse = np.kron(m, [1.0, 1.0])
            fine = np.kron(np.eye(m.shape[0]), [1.0, -1.0])
            m = np.vstack([coarse, fine])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def matrix(self, edge: int) -> Block:
        return self._matrix(edge)


def pixels(path: Path) -> npt.NDArray[np.float64]:
    image = reader_for(str(path))
    image.load(str(path))
    return image.get_array().astype(np.float64)


def psnr(original: npt.NDArray[np.float64], restored: npt.NDArray[np.float64]) -> float:
    mse = float(np.mean((original - restored) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def round_trip(source: Path, transform: Transform, packed: int, workdir: Path) -> tuple[float, int]:
    compressed = workdir / "out.cim"
    restored = workdir / f"back{source.suffix}"
    task = {"packed_block_size": packed, "transform": transform}
    Task(**task).with_action("compress").with_input(str(source)).with_output(str(compressed)).run()
    Task(**task).with_action("extract").with_input(str(compressed)).with_output(str(restored)).run()
    return psnr(pixels(source), pixels(restored)), compressed.stat().st_size


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "earth.ppm"
    transforms: dict[str, Transform] = {
        "Walsh-Hadamard": WalshHadamardTransform(),
        "DCT-II": DiscreteCosineTransform(),
        "Haar": HaarTransform(),
    }
    print(f"{source.name}, luma blocks of 8, chroma blocks of 16\n")
    print(f"{'kept per axis':>13}  {'bytes':>9}  " + "  ".join(f"{n:>14}" for n in transforms))
    with tempfile.TemporaryDirectory() as tmp:
        for packed in KEPT_PER_AXIS:
            results = [round_trip(source, t, packed, Path(tmp)) for t in transforms.values()]
            sizes = {size for _, size in results}
            assert len(sizes) == 1, "the size depends on the geometry alone"
            print(
                f"{packed:>13}  {sizes.pop():>9,}  "
                + "  ".join(f"{quality:>11.2f} dB" for quality, _ in results)
            )


if __name__ == "__main__":
    main()
