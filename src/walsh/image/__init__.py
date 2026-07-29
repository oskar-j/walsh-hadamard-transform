"""Container formats: raster images in and out, spectral ``.cim`` in between.

Every raster format shares the contract described on
:class:`~walsh.image.base.RasterImage`: RGB pixels, top row first, whatever the
file itself stores. :func:`reader_for` picks the class to use from a filename.

This was a single ``image.py`` module up to 0.1.3; the names it exported are
re-exported here, so existing imports keep working.
"""

from __future__ import annotations

from pathlib import Path

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, align, open_binary, open_binary_read, open_binary_write
from walsh.image.base import Pixel, RasterImage
from walsh.image.bmp import (
    BMP_HEADER_FORMAT,
    BMP_HEADER_SIZE,
    BMP_PIXEL_OFFSET,
    BMP_SIGNATURE,
    BMPImage,
)
from walsh.image.cim import COEFF_DTYPE, BlockDescription, CustomizableImage
from walsh.image.ppm import PPM_ASCII_MAGIC, PPM_BINARY_MAGIC, PPM_MAX_SAMPLE, PPMImage
from walsh.image.tiff import TIFF_BIG_ENDIAN, TIFF_LITTLE_ENDIAN, TIFF_MAGIC, TIFFImage

__all__ = [
    "BMP_HEADER_FORMAT",
    "BMP_HEADER_SIZE",
    "BMP_PIXEL_OFFSET",
    "BMP_SIGNATURE",
    "COEFF_DTYPE",
    "PPM_ASCII_MAGIC",
    "PPM_BINARY_MAGIC",
    "PPM_MAX_SAMPLE",
    "SUFFIXES",
    "TIFF_BIG_ENDIAN",
    "TIFF_LITTLE_ENDIAN",
    "TIFF_MAGIC",
    "BMPImage",
    "BlockDescription",
    "CustomizableImage",
    "FileSource",
    "PPMImage",
    "Pixel",
    "RasterImage",
    "TIFFImage",
    "UnsupportedFileFormatError",
    "align",
    "open_binary",
    "open_binary_read",
    "open_binary_write",
    "reader_for",
]

#: Filename suffix to raster class. ``.pnm`` is the generic Netpbm suffix and
#: is treated as PPM, which is the only Netpbm variant supported.
SUFFIXES: dict[str, type[RasterImage]] = {
    ".bmp": BMPImage,
    ".ppm": PPMImage,
    ".pnm": PPMImage,
    ".tif": TIFFImage,
    ".tiff": TIFFImage,
}

#: Used when the format cannot be inferred, which keeps piping through stdin
#: working as it did before PPM existed.
DEFAULT_RASTER: type[RasterImage] = BMPImage


def reader_for(source: FileSource) -> RasterImage:
    """Return an empty image of the right class for ``source``.

    The format is chosen from the filename suffix, case-insensitively. Nothing
    is read, so this works for an output path that does not exist yet.

    Args:
        source: Path whose suffix selects the format, or ``None`` to get the
            default (BMP), since a stream carries no filename to inspect.

    Returns:
        A new, empty instance of the matching :class:`RasterImage` subclass.

    Raises:
        UnsupportedFileFormatError: If the suffix is present but not one this
            package handles.
    """
    if source is None:
        return DEFAULT_RASTER()

    suffix = Path(source).suffix.lower()
    if not suffix:
        return DEFAULT_RASTER()

    try:
        return SUFFIXES[suffix]()
    except KeyError:
        supported = ", ".join(sorted(SUFFIXES))
        raise UnsupportedFileFormatError(
            f"unsupported image format {suffix!r}; expected one of {supported}"
        ) from None
