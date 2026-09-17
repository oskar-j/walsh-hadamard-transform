"""The format modules are grouped by family since 0.4.16; nothing else moved.

Up to 0.4.15 ``walsh/image`` was flat, so ``walsh.image.bmp`` and its siblings
were import paths people could have used. They remain importable as aliases of
the new modules, and everything public is still re-exported from
``walsh.image`` itself.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import walsh.image

MOVED = {
    "walsh.image.bmp": "walsh.image.raster.bmp",
    "walsh.image.tiff": "walsh.image.raster.tiff",
    "walsh.image.ppm": "walsh.image.netpbm.ppm",
    "walsh.image.pam": "walsh.image.netpbm.pam",
    "walsh.image.npy": "walsh.image.arrays.npy",
    "walsh.image.pkl": "walsh.image.arrays.pkl",
}


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_an_old_module_path_is_the_new_module(old: str, new: str) -> None:
    assert importlib.import_module(old) is importlib.import_module(new)
    assert getattr(walsh.image, old.rsplit(".", 1)[1]) is importlib.import_module(new)


@pytest.mark.parametrize(
    "statement",
    [
        "from walsh.image.bmp import BMPImage",
        "import walsh.image.tiff; walsh.image.tiff.TIFFImage",
        "from walsh.image import ppm; ppm.PPMImage",
        "from walsh.image.pkl import safe_loads",
        "from walsh.image.raster import BMPImage, TIFFImage",
        "from walsh.image.netpbm import PAMImage, PPMImage",
        "from walsh.image.arrays import NPYImage, PickleImage",
    ],
)
def test_old_and_new_imports_work_as_the_first_import_of_a_process(statement: str) -> None:
    """In this process the modules are long since imported, which would hide a
    path that only works once something else has loaded the package."""
    result = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_family_packages_export_what_they_hold() -> None:
    from walsh.image import arrays, netpbm, raster

    for package in (raster, netpbm, arrays):
        for name in package.__all__:
            assert getattr(package, name) is getattr(walsh.image, name), name


def test_everything_public_is_still_exported_from_walsh_image() -> None:
    for name in ("BMPImage", "TIFFImage", "PPMImage", "PAMImage", "NPYImage", "PickleImage"):
        assert name in walsh.image.__all__
        assert walsh.image.SUFFIXES  # the registry still maps suffixes to these classes
    assert set(walsh.image.SUFFIXES.values()) == {
        getattr(walsh.image, name)
        for name in ("BMPImage", "TIFFImage", "PPMImage", "PAMImage", "NPYImage", "PickleImage")
    }
