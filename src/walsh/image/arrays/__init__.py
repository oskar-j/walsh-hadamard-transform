"""Pixels that are not in a picture format: NumPy arrays and pickles.

Neither carries colour metadata, so the profile is declared rather than
detected, once, in :mod:`walsh.image.arrays._rules`, and both readers apply
it. :mod:`~walsh.image.arrays.pkl` also owns the allowlisted unpickler that
both use, so that nothing in a file is ever executed. Import the classes from
:mod:`walsh.image`, which re-exports them.
"""

from __future__ import annotations

from walsh.image.arrays.npy import NPY_CHANNELS, NPY_DTYPE, NPYImage
from walsh.image.arrays.pkl import PICKLE_PROTOCOL, PickleImage, pixels_from_object, safe_loads

__all__ = [
    "NPY_CHANNELS",
    "NPY_DTYPE",
    "PICKLE_PROTOCOL",
    "NPYImage",
    "PickleImage",
    "pixels_from_object",
    "safe_loads",
]
