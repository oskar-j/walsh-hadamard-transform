"""Compress an image and restore it, showing both plus their histograms.

Run from the repository root after ``uv sync --all-extras`` (or
``pip install -e .[demo]``)::

    python examples/roundtrip.py

The ``.cim`` and the restored picture are written to a temporary directory,
removed when the window is closed. Nothing is written to ``data/``: its
``recreated.*`` files are the references ``tests/codec/test_golden.py``
compares the codec against, and overwriting one from a changed codec makes
that test compare the codec with its own output.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from walsh import Codec

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCE = DATA / "bmp" / "image.bmp"


def round_trip(source: Path, workdir: Path) -> tuple[Path, Path]:
    """Compress ``source`` into ``workdir`` and restore it next to the ``.cim``."""
    compressed = workdir / "transformed.cim"  # raw post-transform data, not a standard format
    restored = workdir / f"recreated{source.suffix}"
    Codec().compress(input=str(source), output=str(compressed)).run()
    Codec().extract(input=str(compressed), output=str(restored)).run()
    return compressed, restored


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        compressed, restored = round_trip(SOURCE, Path(tmp))

        original_bytes = SOURCE.stat().st_size
        compressed_bytes = compressed.stat().st_size
        print(
            f"{original_bytes:,} B -> {compressed_bytes:,} B "
            f"({compressed_bytes / original_bytes:.1%} of the original)"
        )

        # The image after transformation shows a visible loss of quality, and its
        # colour profile (the per-pixel-value histogram) has shifted as well.
        before = Image.open(SOURCE)
        after = Image.open(restored)

        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for column, (title, image) in enumerate((("original", before), ("restored", after))):
            axes[0][column].imshow(np.asarray(image))
            axes[0][column].set_title(title)
            axes[0][column].axis("off")
            axes[1][column].hist(image.histogram(), bins=40)
            axes[1][column].set_title(f"{title} histogram")

        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
