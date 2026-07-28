from __future__ import annotations

import numpy as np
import pytest

from walsh.transforms import WalshHadamardTransform, _sign_changes


@pytest.fixture
def transform() -> WalshHadamardTransform:
    return WalshHadamardTransform()


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_matrix_is_orthonormal(transform: WalshHadamardTransform, size: int) -> None:
    h = transform._build_matrix(size)
    np.testing.assert_allclose(h @ h.T, np.eye(size), atol=1e-12)


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_matrix_is_symmetric_so_transform_is_involutive(
    transform: WalshHadamardTransform, size: int
) -> None:
    h = transform._build_matrix(size)
    np.testing.assert_allclose(h, h.T, atol=1e-12)


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_rows_are_in_sequency_order(transform: WalshHadamardTransform, size: int) -> None:
    """Walsh ordering means row i has exactly i sign changes."""
    h = transform._build_matrix(size)
    assert list(_sign_changes(h)) == list(range(size))


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_roundtrip_is_lossless(transform: WalshHadamardTransform, size: int) -> None:
    rng = np.random.default_rng(seed=size)
    block = rng.uniform(0, 255, size=(size, size))
    np.testing.assert_allclose(
        transform.inverse_transform(transform.transform(block)), block, atol=1e-9
    )


def test_dc_coefficient_encodes_the_mean(transform: WalshHadamardTransform) -> None:
    block = np.full((8, 8), 100.0)
    spectrum = transform.transform(block)
    # A flat block puts all energy in the zero-sequency term.
    assert spectrum[0, 0] == pytest.approx(100.0 * 8)
    np.testing.assert_allclose(spectrum[1:, :], 0, atol=1e-9)
    np.testing.assert_allclose(spectrum[:, 1:], 0, atol=1e-9)


def test_matrix_is_memoised(transform: WalshHadamardTransform) -> None:
    assert transform._build_matrix(8) is transform._build_matrix(8)


def test_instances_with_different_coeff_do_not_share_cache() -> None:
    plain = WalshHadamardTransform()._build_matrix(4)
    stripped = WalshHadamardTransform(coeff=0.0)._build_matrix(4)
    assert not np.array_equal(plain, stripped)


def test_non_square_input_is_rejected(transform: WalshHadamardTransform) -> None:
    with pytest.raises(ValueError, match="square"):
        transform.transform(np.zeros((4, 8)))


def test_transform_sequence_returns_a_list(transform: WalshHadamardTransform) -> None:
    """Python 3 note: this must not be a lazy map, callers take len() of it."""
    result = transform.transform_sequence([np.zeros((4, 4))] * 3)
    assert isinstance(result, list)
    assert len(result) == 3
