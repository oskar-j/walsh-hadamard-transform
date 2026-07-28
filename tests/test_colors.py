from __future__ import annotations

import pytest

from walsh.colors import RgbColorModel, YCbCrColorModel


@pytest.mark.parametrize(
    "pixel",
    [(0, 0, 0), (255, 255, 255), (128, 64, 32), (10, 200, 90), (255, 0, 0)],
)
def test_rgb_ycbcr_roundtrip(pixel: tuple[int, int, int]) -> None:
    ycbcr = RgbColorModel().get_y_cb_cr(pixel)
    assert YCbCrColorModel().get_rgb(ycbcr) == pytest.approx(pixel, abs=1)


def test_grey_has_neutral_chroma() -> None:
    _, cb, cr = RgbColorModel().get_y_cb_cr((128, 128, 128))
    assert cb == pytest.approx(128)
    assert cr == pytest.approx(128)


def test_luma_follows_itu_weights() -> None:
    y, _, _ = RgbColorModel().get_y_cb_cr((255, 255, 255))
    assert y == pytest.approx(255)


def test_out_of_range_values_are_clamped() -> None:
    assert YCbCrColorModel().get_rgb((300.0, 128.0, 128.0)) == (255, 255, 255)
    assert YCbCrColorModel().get_rgb((-50.0, 128.0, 128.0)) == (0, 0, 0)


def test_identity_models_pass_through() -> None:
    assert RgbColorModel().get_rgb((1, 2, 3)) == (1, 2, 3)
    assert YCbCrColorModel().get_y_cb_cr((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
