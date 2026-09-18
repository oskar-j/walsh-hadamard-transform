"""Draw how the Walsh-Hadamard transform turns a block of pixels into coefficients.

Three rows, top to bottom:

1. the eight Walsh functions of an 8-sample block, square waves whose only
   values are ``+1/sqrt(8)`` and ``-1/sqrt(8)``, ordered by how often they
   change sign (their *sequency*);
2. the 64 basis images an 8x8 block is decomposed into, each the outer
   product of two Walsh functions, with the 16 the codec keeps outlined;
3. one real block from the Blue Marble sample going through the codec: the
   pixels, their spectrum, the top-left corner that survives, and what comes
   back from it.

Needs the ``demo`` extra (matplotlib). Run from the repository root::

    python examples/plot_transform.py [output.png]

The default output is ``doc/how_it_works.png``, which the README shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt

from walsh import WalshHadamardTransform, reader_for
from walsh.colors import rgb_to_ycbcr
from walsh.transforms import hadamard_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "earth.ppm"
OUTPUT = ROOT / "doc" / "how_it_works.png"

EDGE = 8
KEPT = 4

Array = npt.NDArray[np.float64]


def luma_plane(path: Path) -> Array:
    """The picture's luma channel, as a 2-D array."""
    image = reader_for(str(path))
    image.load(str(path))
    array = image.get_array()
    height, width = array.shape[:2]
    return rgb_to_ycbcr(array.reshape(-1, 3).astype(np.float64))[:, 0].reshape(height, width)


#: The block shown is the one with the most contrast among those whose kept
#: corner holds at least this share of the energy: a real edge that sixteen
#: numbers capture. The busiest blocks, cloud texture and the limb of the
#: Earth, keep less and come back blurred, which is where the loss goes.
ENERGY_SHARE_AT_LEAST = 0.99


def pick_block(plane: Array) -> tuple[int, int]:
    """Choose the block to show; see :data:`ENERGY_SHARE_AT_LEAST`.

    Deterministic, so the figure is reproducible from the sample.
    """
    height, width = (n // EDGE * EDGE for n in plane.shape)
    rows, columns = height // EDGE, width // EDGE
    stack = plane[:height, :width].reshape(rows, EDGE, columns, EDGE).swapaxes(1, 2)
    stack = stack.reshape(rows * columns, EDGE, EDGE)
    spectra = WalshHadamardTransform().transform(stack)
    energy = (spectra**2).sum(axis=(1, 2))
    share = (spectra[:, :KEPT, :KEPT] ** 2).sum(axis=(1, 2)) / np.where(energy == 0, 1, energy)
    spread = stack.std(axis=(1, 2))
    spread[share < ENERGY_SHARE_AT_LEAST] = -1
    index = int(spread.argmax())
    return index // columns * EDGE, index % columns * EDGE


def draw_walsh_functions(fig: plt.Figure) -> None:
    h = hadamard_matrix(EDGE)
    axes = fig.subplots(1, EDGE)
    for sequency, ax in enumerate(axes):
        ax.step(np.arange(EDGE + 1), np.append(h[sequency], h[sequency, -1]), where="post", lw=2)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(0, EDGE)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([-1 / np.sqrt(EDGE), 1 / np.sqrt(EDGE)] if sequency == 0 else [])
        ax.set_yticklabels(["-1/√8", "+1/√8"] if sequency == 0 else [])
        ax.set_title(f"sequency {sequency}", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def draw_basis_images(ax: plt.Axes) -> None:
    h = hadamard_matrix(EDGE)
    tile = EDGE + 1  # one pixel of gap between basis images
    canvas = np.full((EDGE * tile - 1, EDGE * tile - 1), np.nan)
    for i in range(EDGE):
        for j in range(EDGE):
            canvas[i * tile : i * tile + EDGE, j * tile : j * tile + EDGE] = np.outer(h[i], h[j])
    ax.imshow(canvas, cmap="gray", vmin=-1 / EDGE, vmax=1 / EDGE, interpolation="nearest")
    ax.add_patch(
        Rectangle((-0.5, -0.5), KEPT * tile - 1, KEPT * tile - 1, fill=False, ec="#d62728", lw=2.5)
    )
    ax.set_xticks([i * tile + EDGE / 2 - 0.5 for i in range(EDGE)], [str(i) for i in range(EDGE)])
    ax.set_yticks([i * tile + EDGE / 2 - 0.5 for i in range(EDGE)], [str(i) for i in range(EDGE)])
    ax.set_xlabel("horizontal sequency")
    ax.set_ylabel("vertical sequency")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def draw_block_pipeline(fig: plt.Figure, plane: Array) -> None:
    top, left = pick_block(plane)
    block = plane[top : top + EDGE, left : left + EDGE]
    transform = WalshHadamardTransform()
    spectrum = transform.transform(block)
    kept = np.zeros_like(spectrum)
    kept[:KEPT, :KEPT] = np.rint(spectrum[:KEPT, :KEPT])
    restored = transform.inverse_transform(kept)
    limit = float(np.abs(spectrum).max())
    # Symmetric log scale: the DC term is hundreds, most of the rest single digits.
    norm = SymLogNorm(linthresh=8.0, vmin=-limit, vmax=limit)

    panels = [
        (block, "gray", None, f"8x8 block of luma\n(row {top}, column {left} of the sample)"),
        (spectrum, "RdBu_r", norm, "its 64 coefficients\n(sequency 0 top-left)"),
        (kept, "RdBu_r", norm, "the 16 the file keeps,\nrounded to integers"),
        (restored, "gray", None, "what comes back\nfrom those 16"),
    ]
    axes = fig.subplots(1, len(panels))
    for ax, (data, cmap, colour_norm, title) in zip(axes, panels, strict=True):
        if colour_norm is None:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(data, cmap=cmap, norm=colour_norm, interpolation="nearest")
            ax.add_patch(Rectangle((-0.5, -0.5), KEPT, KEPT, fill=False, ec="#d62728", lw=2))
            for (i, j), value in np.ndenumerate(data):
                shade = "white" if abs(value) > 0.4 * limit else "black"
                ax.text(
                    j, i, f"{round(value):d}", ha="center", va="center", fontsize=6.5, color=shade
                )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    error = float(np.abs(restored - block).mean())
    # The transform is orthonormal, so energy is the same on both sides.
    share = float((spectrum[:KEPT, :KEPT] ** 2).sum() / (spectrum**2).sum())
    fig.supxlabel(
        f"The 16 kept coefficients hold {share:.1%} of the block's energy; the reconstruction "
        f"is off by {error:.1f} of 255 on average. The rest was the fine detail.",
        fontsize=10,
        color="0.25",
    )


def main(output: Path = OUTPUT, source: Path = SOURCE) -> Path:
    plane = luma_plane(source)
    fig = plt.figure(figsize=(12, 13.5), dpi=150, facecolor="white", layout="constrained")
    fig.suptitle("How the Walsh-Hadamard transform sees a block of pixels", fontsize=16)
    rows = fig.subfigures(3, 1, height_ratios=[1.3, 4.6, 2.9], hspace=0.06)

    rows[0].suptitle(
        "1. The eight Walsh functions: square waves, no multiplications, ordered by sign changes",
        fontsize=12,
    )
    draw_walsh_functions(rows[0])

    rows[1].suptitle(
        "2. Two of them at a time make the 64 patterns any 8x8 block is a mix of; "
        "the codec keeps the 16 in the corner",
        fontsize=12,
    )
    draw_basis_images(rows[1].subplots(1, 1))

    rows[2].suptitle(
        "3. One real block: pixels, their coefficients, the kept corner, pixels again",
        fontsize=12,
    )
    draw_block_pipeline(rows[2], plane)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    target = main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
