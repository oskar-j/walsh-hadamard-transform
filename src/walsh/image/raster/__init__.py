"""Picture file formats that carry their own header: BMP, PNG and TIFF.

Each module holds one :class:`~walsh.image.base.RasterImage` subclass and
honours its contract, RGB pixels with the top row first, whatever the file
itself stores. Import the classes from :mod:`walsh.image`, which re-exports
them; these modules are where they live.
"""

from __future__ import annotations

from walsh.image.raster.bmp import (
    BMP_HEADER_FORMAT,
    BMP_HEADER_SIZE,
    BMP_PIXEL_OFFSET,
    BMP_SIGNATURE,
    BMPImage,
)
from walsh.image.raster.png import PNG_SIGNATURE, PNGImage
from walsh.image.raster.tiff import TIFF_BIG_ENDIAN, TIFF_LITTLE_ENDIAN, TIFF_MAGIC, TIFFImage

__all__ = [
    "BMP_HEADER_FORMAT",
    "BMP_HEADER_SIZE",
    "BMP_PIXEL_OFFSET",
    "BMP_SIGNATURE",
    "PNG_SIGNATURE",
    "TIFF_BIG_ENDIAN",
    "TIFF_LITTLE_ENDIAN",
    "TIFF_MAGIC",
    "BMPImage",
    "PNGImage",
    "TIFFImage",
]
