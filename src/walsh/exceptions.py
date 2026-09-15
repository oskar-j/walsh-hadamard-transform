"""Exception types raised by the package, and the groupings callers act on.

This module deliberately imports nothing from the rest of ``walsh``, so any
module can import it without risking a cycle.
"""

from __future__ import annotations

import struct

__all__ = [
    "EXPECTED_ERRORS",
    "UnsupportedFileFormatError",
    "WalshError",
]


class WalshError(Exception):
    """Base class for every error this package raises on its own behalf.

    Catch this to handle anything ``walsh`` reports without also catching
    unrelated failures such as a disk error.
    """


class UnsupportedFileFormatError(WalshError):
    """Raised when a file is not in a format this package supports, or is a
    variant of one that it does not.

    Every reader raises it with a message naming what it found: an unknown
    filename suffix, a bad magic number, a profile outside the supported
    subset (a BMP that is not 24-bit, 16-bit Netpbm samples, a compressed or
    planar TIFF, a NumPy array that is not ``uint8``), a truncated file, or a
    ``.cim`` header that does not describe a consistent image.
    """


#: Failures that mean "this input cannot be processed" rather than "this code is
#: broken". A front end can turn these into a message; anything else is a bug and
#: should keep its traceback.
#:
#: ``struct.error`` is listed explicitly because it derives straight from
#: ``Exception``, not from ``ValueError``: a file too short to hold a header
#: reaches ``struct.unpack`` and would otherwise escape this grouping. That is a
#: real bug this tuple existed to prevent, so keep it in the list.
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    WalshError,
    ValueError,
    struct.error,
)
