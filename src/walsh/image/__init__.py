"""Container formats: raster images in and out, spectral ``.cim`` in between.

Every raster format shares the contract described on
:class:`~walsh.image.base.RasterImage`: RGB pixels, top row first, whatever the
file itself stores. :func:`reader_for` picks the class to use from a filename.

This was a single ``image.py`` module up to 0.1.3; the names it exported are
re-exported here, so existing imports keep working. The format modules are
grouped by family: :mod:`~walsh.image.raster` (BMP, PNG, TIFF),
:mod:`~walsh.image.netpbm` (PPM, PAM) and :mod:`~walsh.image.arrays` (``.npy``,
pickles). Import their classes from this package rather than from those
modules, whose paths are an implementation detail.
"""

from __future__ import annotations

from pathlib import Path

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, align, open_binary, open_binary_read, open_binary_write
from walsh.image.arrays.npy import NPY_CHANNELS, NPY_DTYPE, NPYImage
from walsh.image.arrays.pkl import PICKLE_PROTOCOL, PickleImage, pixels_from_object, safe_loads
from walsh.image.base import Pixel, PixelArray, RasterImage
from walsh.image.cim import (
    COEFF_DTYPE,
    MAX_BLOCK_SIZE,
    MAX_BLOCKS_PER_CHANNEL,
    BlockDescription,
    CustomizableImage,
    blocks_for,
)
from walsh.image.netpbm.pam import PAM_DEPTH, PAM_MAGIC, PAM_TUPLTYPE, PAMImage
from walsh.image.netpbm.ppm import PPM_ASCII_MAGIC, PPM_BINARY_MAGIC, PPM_MAX_SAMPLE, PPMImage
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
    "COEFF_DTYPE",
    "MAX_BLOCKS_PER_CHANNEL",
    "MAX_BLOCK_SIZE",
    "NPY_CHANNELS",
    "NPY_DTYPE",
    "PAM_DEPTH",
    "PAM_MAGIC",
    "PAM_TUPLTYPE",
    "PICKLE_PROTOCOL",
    "PNG_SIGNATURE",
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
    "NPYImage",
    "PAMImage",
    "PNGImage",
    "PPMImage",
    "PickleImage",
    "Pixel",
    "PixelArray",
    "RasterImage",
    "TIFFImage",
    "UnsupportedFileFormatError",
    "align",
    "blocks_for",
    "open_binary",
    "open_binary_read",
    "open_binary_write",
    "pixels_from_object",
    "reader_for",
    "safe_loads",
]

#: Filename suffix to raster class. ``.pnm`` is the generic suffix for the
#: older Netpbm formats and is treated as PPM, the only one of those
#: supported; PAM has its own ``.pam``. ``.npy`` is a bare NumPy array, and
#: ``.pkl`` / ``.pickle`` a pickled array or list of pixels, read through an
#: allowlist so that nothing in the file is executed. ``.png`` is the one
#: compressed format, inflated by the standard library's zlib.
SUFFIXES: dict[str, type[RasterImage]] = {
    ".bmp": BMPImage,
    ".npy": NPYImage,
    ".pam": PAMImage,
    ".pickle": PickleImage,
    ".pkl": PickleImage,
    ".png": PNGImage,
    ".ppm": PPMImage,
    ".pnm": PPMImage,
    ".tif": TIFFImage,
    ".tiff": TIFFImage,
}

#: Used when the format cannot be inferred: a path with no suffix, or ``None``
#: for stdin. BMP because it was the only format before 0.2.0, so this is what
#: any caller from then still expects.
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
