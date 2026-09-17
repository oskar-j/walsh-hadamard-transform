"""The Netpbm family: PPM (``P6`` and ``P3``) and PAM (``P7``).

Their rasters are identical behind different headers, so the sample decoder
and encoder are shared, in :mod:`walsh.image.netpbm._samples`. Import the
classes from :mod:`walsh.image`, which re-exports them.
"""

from __future__ import annotations

from walsh.image.netpbm.pam import PAM_DEPTH, PAM_MAGIC, PAM_TUPLTYPE, PAMImage
from walsh.image.netpbm.ppm import PPM_ASCII_MAGIC, PPM_BINARY_MAGIC, PPM_MAX_SAMPLE, PPMImage

__all__ = [
    "PAM_DEPTH",
    "PAM_MAGIC",
    "PAM_TUPLTYPE",
    "PPM_ASCII_MAGIC",
    "PPM_BINARY_MAGIC",
    "PPM_MAX_SAMPLE",
    "PAMImage",
    "PPMImage",
]
