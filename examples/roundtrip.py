"""Compress an image and restore it, showing both plus their histograms.

Run from the repository root after ``pip install -e .[demo]``::

    python examples/roundtrip.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from walsh import Task

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCE = DATA / "image.bmp"
COMPRESSED = DATA / "transformed.cim"  # raw post-transform data, not a standard format
RESTORED = DATA / "recreated.bmp"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    Task().with_action("compress").with_input(str(SOURCE)).with_output(str(COMPRESSED)).run()

    Task().with_action("extract").with_input(str(COMPRESSED)).with_output(str(RESTORED)).run()

    original_bytes = SOURCE.stat().st_size
    compressed_bytes = COMPRESSED.stat().st_size
    print(
        f"{original_bytes:,} B -> {compressed_bytes:,} B "
        f"({compressed_bytes / original_bytes:.1%} of the original)"
    )

    # The image after transformation shows a visible loss of quality, and its
    # colour profile (the per-pixel-value histogram) has shifted as well.
    before = Image.open(SOURCE)
    after = Image.open(RESTORED)

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
