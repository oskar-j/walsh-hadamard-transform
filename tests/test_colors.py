from __future__ import annotations

import numpy as np
import pytest

from walsh.colors import RgbColorModel, YCbCrColorModel, rgb_to_ycbcr, ycbcr_to_rgb


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


def _reference_ycbcr(r: int, g: int, b: int) -> tuple[float, float, float]:
    """The per-pixel arithmetic exactly as written before 0.3.3, as the oracle."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _reference_rgb(y: float, cb: float, cr: float) -> tuple[int, int, int]:
    """The per-pixel inverse exactly as written before 0.3.3: int() then clamp."""

    def clamp(value: float) -> int:
        return max(0, min(255, int(value)))

    return (
        clamp(y + 1.402 * (cr - 128)),
        clamp(y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)),
        clamp(y + 1.772 * (cb - 128)),
    )


def test_array_conversion_is_bit_identical_to_the_scalar_formula() -> None:
    """Whole-array numpy must reproduce the scalar arithmetic exactly.

    The same IEEE operations in the same order round the same way, and the
    checked-in sample outputs depend on it, so approximate is not enough.
    """
    rng = np.random.default_rng(seed=23)
    rgb = rng.integers(0, 256, size=(1000, 3))

    ycbcr = rgb_to_ycbcr(rgb)
    expected = np.array([_reference_ycbcr(*map(int, pixel)) for pixel in rgb])
    assert ycbcr.dtype == np.float64
    assert ycbcr.tobytes() == expected.tobytes()


def test_array_inverse_is_identical_to_the_scalar_formula() -> None:
    rng = np.random.default_rng(seed=29)
    ycbcr = rng.uniform(-40, 300, size=(1000, 3))  # deliberately past both ends

    rgb = ycbcr_to_rgb(ycbcr)
    expected = [_reference_rgb(*map(float, pixel)) for pixel in ycbcr]
    assert rgb.dtype == np.uint8
    assert [tuple(pixel) for pixel in rgb.tolist()] == expected


def test_array_functions_keep_the_input_shape() -> None:
    assert rgb_to_ycbcr((255, 255, 255)).shape == (3,)
    assert rgb_to_ycbcr(np.zeros((6, 3))).shape == (6, 3)
    assert ycbcr_to_rgb(np.full((2, 5, 3), 128.0)).shape == (2, 5, 3)


def test_array_inverse_truncates_towards_zero_then_clamps() -> None:
    out = ycbcr_to_rgb([[255.9, 128.0, 128.0], [-0.9, 128.0, 128.0], [300.0, 128.0, 128.0]])
    assert out[:, 0].tolist() == [255, 0, 255]


def test_per_pixel_models_are_the_array_functions_for_one_pixel() -> None:
    pixel = (10, 200, 90)
    assert RgbColorModel().get_y_cb_cr(pixel) == tuple(rgb_to_ycbcr(pixel).tolist())
    ycbcr = (120.4, 90.0, 200.0)
    assert YCbCrColorModel().get_rgb(ycbcr) == tuple(ycbcr_to_rgb(ycbcr).tolist())


def test_per_pixel_results_are_plain_python_numbers() -> None:
    """Callers compare and format these; numpy scalars would surprise them."""
    assert all(type(v) is float for v in RgbColorModel().get_y_cb_cr((1, 2, 3)))
    assert all(type(v) is int for v in YCbCrColorModel().get_rgb((1.0, 2.0, 3.0)))
    assert all(type(v) is int for v in RgbColorModel().get_rgb((1, 2, 3)))
