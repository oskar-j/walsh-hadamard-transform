"""Image compression with the Walsh-Hadamard transform.

Typical use::

    from walsh import Codec

    Codec().compress(input="image.bmp", output="out.cim").run()
    Codec().extract(input="out.cim", output="back.bmp").run()

    # Or skip the .cim and write the lossy reconstruction directly:
    Codec().compress(input="image.bmp", output="image_compressed.bmp").run()
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from walsh.codec import Action, Codec
from walsh.colors import ColorModel, RgbColorModel, YCbCrColorModel
from walsh.exceptions import UnsupportedFileFormatError, WalshError
from walsh.image import (
    BlockDescription,
    BMPImage,
    CustomizableImage,
    NPYImage,
    PAMImage,
    PickleImage,
    PNGImage,
    PPMImage,
    RasterImage,
    TIFFImage,
    reader_for,
)
from walsh.transforms import (
    DiscreteCosineTransform,
    HaarTransform,
    MatrixTransform,
    Transform,
    WalshHadamardTransform,
    transform_for,
)
from walsh.vectorizer import CompressionStats, Vectorizer

try:
    __version__ = version("walsh")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "Action",
    "BMPImage",
    "BlockDescription",
    "Codec",
    "ColorModel",
    "CompressionStats",
    "CustomizableImage",
    "DiscreteCosineTransform",
    "HaarTransform",
    "MatrixTransform",
    "NPYImage",
    "PAMImage",
    "PNGImage",
    "PPMImage",
    "PickleImage",
    "RasterImage",
    "RgbColorModel",
    "TIFFImage",
    "Transform",
    "UnsupportedFileFormatError",
    "Vectorizer",
    "WalshError",
    "WalshHadamardTransform",
    "YCbCrColorModel",
    "__version__",
    "reader_for",
    "transform_for",
]
