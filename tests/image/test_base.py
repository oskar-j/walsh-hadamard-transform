"""The contract every raster format shares, checked through each of them."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("cls_name", ["BMPImage", "PPMImage", "PAMImage", "TIFFImage", "NPYImage"])
@pytest.mark.parametrize(("width", "height", "count"), [(4, 4, 3), (2, 2, 16), (2, 2, 0)])
def test_save_rejects_a_pixel_count_that_contradicts_the_dimensions(
    tmp_path: Path, cls_name: str, width: int, height: int, count: int
) -> None:
    """The contract RasterImage documents but never used to enforce.

    Too few pixels wrote a file no reader in this package can load; too many
    dropped the surplus with no error at all, and left the BMP size fields
    contradicting the body. The check runs before any header is written, so a
    rejected save leaves nothing behind.
    """
    import walsh.image as image_package

    cls = getattr(image_package, cls_name)
    image = cls()
    image.set_dimensions(width, height)
    image.set_raw_data([(index % 256, 0, 0) for index in range(count)])

    path = tmp_path / "wrong.out"
    with pytest.raises(ValueError, match="require"):
        image.save(str(path))
    assert not path.exists(), "a rejected save must not leave a file behind"


@pytest.mark.parametrize("cls_name", ["BMPImage", "PPMImage", "PAMImage", "TIFFImage", "NPYImage"])
def test_save_accepts_an_exactly_matching_pixel_count(tmp_path: Path, cls_name: str) -> None:
    """The guard must not reject the ordinary case."""
    import walsh.image as image_package
    from conftest import gradient_pixels

    cls = getattr(image_package, cls_name)
    image = cls()
    image.set_dimensions(4, 3)
    image.set_raw_data(gradient_pixels(4, 3))

    path = tmp_path / "right.out"
    image.save(str(path))
    assert path.stat().st_size > 0
