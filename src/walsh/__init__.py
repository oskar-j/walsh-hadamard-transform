"""Image compression with the Walsh-Hadamard transform.

Typical use::

    from walsh import Task

    Task().with_action("compress").with_input("image.bmp").with_output("out.cim").run()
    Task().with_action("extract").with_input("out.cim").with_output("back.bmp").run()
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from walsh.colors import ColorModel, RgbColorModel, YCbCrColorModel
from walsh.exceptions import UnsupportedFileFormatError, WalshError
from walsh.image import (
    BlockDescription,
    BMPImage,
    CustomizableImage,
    PPMImage,
    RasterImage,
    TIFFImage,
    reader_for,
)
from walsh.task import Action, Task
from walsh.transforms import Transform, WalshHadamardTransform

try:
    __version__ = version("walsh")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "Action",
    "BMPImage",
    "BlockDescription",
    "ColorModel",
    "CustomizableImage",
    "PPMImage",
    "RasterImage",
    "RgbColorModel",
    "TIFFImage",
    "Task",
    "Transform",
    "UnsupportedFileFormatError",
    "WalshError",
    "WalshHadamardTransform",
    "YCbCrColorModel",
    "__version__",
    "reader_for",
]
