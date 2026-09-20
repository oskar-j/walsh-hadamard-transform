"""The format modules are grouped by family, and ``walsh.image`` exports them all.

``walsh.image`` is the import path; where a class lives below it is an
implementation detail, so these tests pin the exports rather than the files.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import walsh.image

FORMAT_CLASSES = (
    "BMPImage",
    "PNGImage",
    "TIFFImage",
    "PPMImage",
    "PAMImage",
    "NPYImage",
    "PickleImage",
)


@pytest.mark.parametrize(
    "statement",
    [
        "from walsh import BMPImage, PickleImage, PNGImage",
        "from walsh import Action, Codec",
        "from walsh.codec import Action, Codec",
        "from walsh import CompressionStats, Vectorizer",
        "from walsh.vectorizer import Vectorizer",
        "from walsh.image import BMPImage, TIFFImage, PPMImage, PAMImage, NPYImage, PickleImage",
        "from walsh.image import PNG_SIGNATURE, PNGImage",
        "from walsh.image.raster import BMPImage, PNGImage, TIFFImage",
        "import walsh.image.raster.png",
        "from walsh.image.netpbm import PAMImage, PPMImage",
        "from walsh.image.arrays import NPYImage, PickleImage",
        "import walsh.image.arrays.pkl as pkl; pkl.safe_loads",
    ],
)
def test_every_import_path_works_as_the_first_import_of_a_process(statement: str) -> None:
    """In this process the modules are long since imported, which would hide a
    path that only works once something else has loaded the package, such as
    an import cycle between the family packages."""
    result = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_family_packages_export_what_they_hold() -> None:
    from walsh.image import arrays, netpbm, raster

    for package in (raster, netpbm, arrays):
        for name in package.__all__:
            assert getattr(package, name) is getattr(walsh.image, name), name


def test_everything_public_is_still_exported_from_walsh_image() -> None:
    for name in FORMAT_CLASSES:
        assert name in walsh.image.__all__
    assert set(walsh.image.SUFFIXES.values()) == {
        getattr(walsh.image, name) for name in FORMAT_CLASSES
    }
