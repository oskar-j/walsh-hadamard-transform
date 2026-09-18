"""Which readers can consume a non-seekable stream, and which cannot.

The ``None`` file source means ``sys.stdin``. That is a library affordance
rather than a CLI feature, and on the read side it only works for a format that
reads forward: BMP, TIFF and ASCII PPM seek while parsing, so a *pipe* fails
for them where a ``< file`` redirect, which is seekable, succeeds. These tests
pin that table so the documentation cannot silently go stale again. The
existing stdin test in ``test_io.py`` cannot catch it: it substitutes an
``io.BytesIO``, which is seekable.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    gradient_pixels,
    write_bmp,
    write_npy,
    write_pam,
    write_png,
    write_ppm,
    write_tiff,
)
from walsh.image import (
    BlockDescription,
    BMPImage,
    CustomizableImage,
    NPYImage,
    PAMImage,
    PNGImage,
    PPMImage,
    RasterImage,
    TIFFImage,
)


class _Pipe(io.RawIOBase):
    """A forward-only stream: ``read`` works, ``seek`` and ``tell`` do not."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, target: bytearray) -> int:
        return self._buffer.readinto(target)

    def seek(self, offset: int, whence: int = 0) -> int:
        raise io.UnsupportedOperation("File or stream is not seekable.")

    def tell(self) -> int:
        raise io.UnsupportedOperation("File or stream is not seekable.")


def _as_stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    stream = io.BufferedReader(_Pipe(data))
    assert not stream.seekable()
    monkeypatch.setattr(sys, "stdin", type("Stdin", (), {"buffer": stream})())


PIXELS = gradient_pixels(2, 2)

FORWARD_ONLY = [
    ("PPM/P6", PPMImage, lambda t: write_ppm(t / "p.ppm", 2, 2, PIXELS)),
    ("PAM", PAMImage, lambda t: write_pam(t / "p.pam", 2, 2, PIXELS)),
    ("NPY", NPYImage, lambda t: write_npy(t / "p.npy", 2, 2, PIXELS)),
    ("PNG", PNGImage, lambda t: write_png(t / "p.png", 2, 2, PIXELS, filters=(4, 3))),
]
SEEKING = [
    ("BMP", BMPImage, lambda t: write_bmp(t / "p.bmp", 2, 2, PIXELS)),
    ("TIFF", TIFFImage, lambda t: write_tiff(t / "p.tif", 2, 2, PIXELS)),
    ("PPM/P3", PPMImage, lambda t: write_ppm(t / "p.ppm", 2, 2, PIXELS, ascii_form=True)),
]


@pytest.mark.parametrize(("name", "cls", "write"), FORWARD_ONLY, ids=[c[0] for c in FORWARD_ONLY])
def test_forward_only_readers_accept_a_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cls: type[RasterImage],
    write: Callable[[Path], Path],
) -> None:
    _as_stdin(monkeypatch, write(tmp_path).read_bytes())
    image = cls()
    image.load(None)
    assert image.get_dimensions() == (2, 2)
    assert image.get_raw_data() == PIXELS


@pytest.mark.parametrize(("name", "cls", "write"), SEEKING, ids=[c[0] for c in SEEKING])
def test_seeking_readers_refuse_a_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cls: type[RasterImage],
    write: Callable[[Path], Path],
) -> None:
    """Documented, not fixed: real pipe support would mean buffering stdin,
    which changes the memory profile of the readers that stream correctly."""
    _as_stdin(monkeypatch, write(tmp_path).read_bytes())
    with pytest.raises(io.UnsupportedOperation):
        cls().load(None)


def test_the_cim_reader_accepts_a_pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """extract from a piped .cim is fine: the reader never seeks."""
    image = CustomizableImage()
    image.set_dimensions(8, 8)
    description = BlockDescription(8, 4, 1)
    image.set_descriptions(description, description, description)
    block = [np.ones((8, 8))]
    image.set_data(block, block, block)
    path = tmp_path / "p.cim"
    image.save(str(path))

    _as_stdin(monkeypatch, path.read_bytes())
    assert CustomizableImage.load(None).get_dimensions() == (8, 8)


def test_the_exported_exception_is_documented_for_every_format() -> None:
    """It was described as BMP-only since the original port, and that sentence
    is what help() shows for a public symbol raised from nine modules."""
    import inspect

    import walsh

    doc = inspect.getdoc(walsh.UnsupportedFileFormatError) or ""
    assert "not in a format this package supports" in doc
    assert not doc.startswith("Raised for BMP")
