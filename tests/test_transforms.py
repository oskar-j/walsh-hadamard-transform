from __future__ import annotations

import math

import numpy as np
import pytest

from walsh.transforms import WalshHadamardTransform, _sign_changes, hadamard_matrix


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


def test_matrix_does_not_depend_on_coeff() -> None:
    """Coefficient removal acts on the spectrum, never on the matrix.

    Every entry of an orthonormal Hadamard matrix has the same magnitude, so
    any threshold applied to the matrix could only ever be all-or-nothing.
    """
    plain = WalshHadamardTransform()._build_matrix(4)
    stripped = WalshHadamardTransform(coeff=100.0)._build_matrix(4)
    assert np.array_equal(plain, stripped)
    assert plain is stripped  # and they share the one memo entry


def test_coeff_removal_is_graded(transform: WalshHadamardTransform) -> None:
    """The whole point of the fix: more threshold, monotonically fewer terms."""
    rng = np.random.default_rng(seed=7)
    block = rng.uniform(0, 255, size=(8, 8))

    kept = [
        int((WalshHadamardTransform(coeff=c).transform(block) != 0).sum())
        for c in (0.0, 1.0, 5.0, 20.0, 100.0, 1e9)
    ]
    assert kept == sorted(kept, reverse=True), kept
    assert kept[0] == 64, "a zero threshold must keep everything"
    assert kept[-1] == 0, "a huge threshold must drop everything"
    assert len(set(kept)) > 2, f"expected a gradient, not a cliff: {kept}"


def test_coeff_removal_drops_exactly_the_small_coefficients() -> None:
    rng = np.random.default_rng(seed=11)
    block = rng.uniform(0, 255, size=(8, 8))
    full = WalshHadamardTransform().transform(block)

    threshold = float(np.median(np.abs(full)))
    thinned = WalshHadamardTransform(coeff=threshold).transform(block)

    below = np.abs(full) < threshold
    assert np.all(thinned[below] == 0)
    np.testing.assert_allclose(thinned[~below], full[~below])


def test_inverse_does_not_reapply_the_threshold() -> None:
    """Removal is the lossy step; doing it twice would discard detail again."""
    rng = np.random.default_rng(seed=13)
    spectrum = rng.uniform(-50, 50, size=(8, 8))

    lossy = WalshHadamardTransform(coeff=1e9)
    plain = WalshHadamardTransform()
    np.testing.assert_allclose(lossy.inverse_transform(spectrum), plain.inverse_transform(spectrum))


def test_negative_coeff_is_rejected() -> None:
    """It is compared against a magnitude, so a negative value keeps everything."""
    with pytest.raises(ValueError, match="non-negative"):
        WalshHadamardTransform(coeff=-1.0)


def test_matrix_memo_does_not_pin_transform_instances() -> None:
    """The memo is keyed on size alone, so instances stay collectable."""
    import gc
    import weakref

    from walsh.transforms import _hadamard_signs, hadamard_matrix

    hadamard_matrix.cache_clear()
    _hadamard_signs.cache_clear()
    transform = WalshHadamardTransform()
    reference = weakref.ref(transform)
    transform.transform(np.zeros((8, 8)))

    del transform
    gc.collect()
    assert reference() is None, "the memo is holding the instance alive"

    for _ in range(20):
        WalshHadamardTransform().transform(np.zeros((8, 8)))
    assert hadamard_matrix.cache_size() == 1, "cache grows with instance count"
    assert _hadamard_signs.cache_size() == 1, "cache grows with instance count"


def test_non_square_input_is_rejected(transform: WalshHadamardTransform) -> None:
    with pytest.raises(ValueError, match="square"):
        transform.transform(np.zeros((4, 8)))


def test_transform_sequence_returns_a_list(transform: WalshHadamardTransform) -> None:
    """Python 3 note: this must not be a lazy map, callers take len() of it."""
    result = transform.transform_sequence([np.zeros((4, 4))] * 3)
    assert isinstance(result, list)
    assert len(result) == 3


def _reference_matrix(size: int) -> np.ndarray:
    """The pre-0.4.0 construction, kept verbatim as the oracle.

    Negates, one bit at a time, every entry whose row and column indices share
    that bit, then sorts rows into sequency order.
    """
    n = int(math.log(size, 2))
    matrix = np.full((size, size), 1 / (np.sqrt(2) ** n), dtype=np.float64)
    for i in range(n):
        for j in range(size):
            for k in range(size):
                if (j // 2**i) % 2 == 1 and (k // 2**i) % 2 == 1:
                    matrix[j, k] = -matrix[j, k]
    return matrix[np.argsort(_sign_changes(matrix), kind="stable")]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
def test_matrix_is_bit_identical_to_the_reference_loop(size: int) -> None:
    """Sylvester's construction must reproduce the old loop exactly, not closely.

    The codec's output is defined by these entries and the checked-in sample
    files were produced with the loop, so byte equality is the bar.
    """
    assert hadamard_matrix(size).tobytes() == _reference_matrix(size).tobytes()


@pytest.mark.parametrize("size", [0, -8, 3, 6, 12, 100])
def test_size_must_be_a_power_of_two(size: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        hadamard_matrix(size)


def test_a_stack_transforms_exactly_like_its_blocks(transform: WalshHadamardTransform) -> None:
    """One 3-D call must give what one 2-D call per block gives, bit for bit."""
    rng = np.random.default_rng(seed=17)
    stack = rng.uniform(0, 255, size=(5, 8, 8))

    batched = transform.transform(stack)
    assert batched.shape == stack.shape
    for block, spectrum in zip(stack, batched, strict=True):
        np.testing.assert_array_equal(spectrum, transform.transform(block))

    restored = transform.inverse_transform(batched)
    for block, back in zip(stack, restored, strict=True):
        np.testing.assert_array_equal(back, transform.inverse_transform(transform.transform(block)))


def test_stacked_non_square_blocks_are_rejected(transform: WalshHadamardTransform) -> None:
    with pytest.raises(ValueError, match="square"):
        transform.transform(np.zeros((3, 4, 8)))
    with pytest.raises(ValueError, match="square"):
        transform.transform(np.zeros((2, 3, 4, 4)))


@pytest.mark.parametrize("coeff", [None, 10.0])
def test_sequence_methods_match_per_block_calls(coeff: float | None) -> None:
    """Batching is an optimisation, so it must be invisible in the result."""
    rng = np.random.default_rng(seed=19)
    blocks = list(rng.uniform(0, 255, size=(7, 8, 8)))
    transform = WalshHadamardTransform(coeff=coeff)

    spectra = transform.transform_sequence(blocks)
    assert len(spectra) == len(blocks)
    for block, spectrum in zip(blocks, spectra, strict=True):
        np.testing.assert_array_equal(spectrum, transform.transform(block))

    restored = transform.inverse_transform_sequence(spectra)
    assert len(restored) == len(blocks)
    for spectrum, block in zip(spectra, restored, strict=True):
        np.testing.assert_array_equal(block, transform.inverse_transform(spectrum))


def test_sequence_with_mixed_block_sizes_falls_back_to_one_call_each(
    transform: WalshHadamardTransform,
) -> None:
    blocks = [np.full((4, 4), 1.0), np.full((8, 8), 2.0), np.full((4, 4), 3.0)]
    spectra = transform.transform_sequence(blocks)
    assert [spectrum.shape for spectrum in spectra] == [(4, 4), (8, 8), (4, 4)]
    for block, spectrum in zip(blocks, spectra, strict=True):
        np.testing.assert_array_equal(spectrum, transform.transform(block))


def test_empty_sequence_gives_an_empty_list(transform: WalshHadamardTransform) -> None:
    assert transform.transform_sequence([]) == []
    assert transform.inverse_transform_sequence(iter(())) == []


def test_sequence_accepts_any_iterable(transform: WalshHadamardTransform) -> None:
    """A generator is consumed once and still comes back as a list."""
    spectra = transform.transform_sequence(np.zeros((4, 4)) for _ in range(3))
    assert isinstance(spectra, list)
    assert len(spectra) == 3


def test_base_class_sequence_methods_stay_one_call_per_block() -> None:
    """The defaults on Transform serve subclasses that know nothing of stacks."""
    from walsh.transforms import Block, Transform

    class Negate(Transform):
        def transform(self, src: Block) -> Block:
            return -np.asarray(src)

        def inverse_transform(self, src: Block) -> Block:
            return -np.asarray(src)

    blocks = [np.full((2, 2), 1.0), np.full((3, 3), 2.0)]  # mixed shapes are fine here
    spectra = Negate().transform_sequence(blocks)
    assert [s.tolist() for s in spectra] == [(-b).tolist() for b in blocks]
    restored = Negate().inverse_transform_sequence(spectra)
    assert [r.tolist() for r in restored] == [b.tolist() for b in blocks]


# --- exact arithmetic (0.4.12, #39) -----------------------------------------


def _on_a_coarse_grid(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    """Samples that are multiples of 1/1024, so the transform's snap is a no-op
    and the oracles below can be fed the very same bits."""
    return rng.integers(0, int(scale * 1024), size=shape).astype(np.float64) / 1024.0


def _natural_index_of_sequency_row(size: int) -> np.ndarray:
    """For each sequency-ordered row, its index in Sylvester's natural order."""
    natural = np.ones((1, 1))
    for _ in range(size.bit_length() - 1):
        natural = np.kron(np.array([[1.0, 1.0], [1.0, -1.0]]), natural)
    signs = np.sign(hadamard_matrix(size))
    return np.array([int(np.flatnonzero((natural == row).all(axis=1))[0]) for row in signs])


def _butterfly_spectrum(blocks: np.ndarray) -> np.ndarray:
    """The fast Walsh-Hadamard transform: the butterfly #39 proposed, as the oracle.

    It adds the same samples in a completely different order from the two
    matrix products, so bit equality between the two is possible only when
    neither of them rounds anywhere.
    """
    x = np.asarray(blocks, dtype=np.float64)
    size = x.shape[-1]
    for axis in (-2, -1):
        moved = np.moveaxis(x, axis, -1)
        y = moved.reshape(-1, size)
        half = 1
        while half < size:
            y = y.reshape(-1, size // (2 * half), 2, half)
            y = np.concatenate([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], axis=-1)
            y = y.reshape(-1, size)
            half *= 2
        x = np.moveaxis(y.reshape(moved.shape), -1, axis)
    order = _natural_index_of_sequency_row(size)
    return x[..., order, :][..., :, order] / size


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 128])
def test_transform_is_bit_identical_to_a_butterfly_in_a_different_order(size: int) -> None:
    """The exactness claim, tested the only way it can be: two unrelated
    operation orders must agree to the last bit, on pixels and on the full
    int16 coefficient range at the largest block the container accepts."""
    rng = np.random.default_rng(seed=size)
    transform = WalshHadamardTransform()
    pixels = _on_a_coarse_grid(rng, (6, size, size), 256.0)
    np.testing.assert_array_equal(transform.transform(pixels), _butterfly_spectrum(pixels))

    coefficients = rng.integers(-32768, 32768, size=(6, size, size)).astype(np.float64)
    np.testing.assert_array_equal(
        transform.inverse_transform(coefficients), _butterfly_spectrum(coefficients)
    )


def test_transform_matches_exact_rational_arithmetic() -> None:
    from fractions import Fraction

    rng = np.random.default_rng(seed=23)
    block = _on_a_coarse_grid(rng, (8, 8), 256.0)
    signs = [[int(v) for v in row] for row in np.sign(hadamard_matrix(8))]
    exact = [
        [
            float(
                sum(
                    signs[i][k] * Fraction(block[k, m]) * signs[m][j]
                    for k in range(8)
                    for m in range(8)
                )
                / 8
            )
            for j in range(8)
        ]
        for i in range(8)
    ]
    np.testing.assert_array_equal(WalshHadamardTransform().transform(block), np.array(exact))


def test_integer_results_come_out_as_exact_integers() -> None:
    """A DC of 16 at edge 8 is a flat block of exactly 2, not 1.999999999999999.

    That last bit is the whole point on the way back: the colour conversion
    truncates, so the orthonormal matrix products lost a level in about 0.25%
    of the samples of every decoded picture before 0.4.12.
    """
    transform = WalshHadamardTransform()
    spectrum = np.zeros((8, 8))
    spectrum[0, 0] = 16.0
    restored = transform.inverse_transform(spectrum)
    assert restored.tobytes() == np.full((8, 8), 2.0).tobytes()

    forward = transform.transform(np.full((8, 8), 2.0))
    assert forward.tobytes() == spectrum.tobytes()


def test_input_is_snapped_to_a_grid_that_is_part_of_the_contract() -> None:
    """Arbitrary doubles are rounded to a multiple of 2**-30 at edge 8 before
    anything else, so the result is the same as for the snapped input, and the
    snap itself is far below anything the codec's rounding can see."""
    rng = np.random.default_rng(seed=29)
    transform = WalshHadamardTransform()
    block = rng.uniform(0, 255, size=(8, 8))
    snapped = np.rint(block * 2.0**30) / 2.0**30
    assert not np.array_equal(block, snapped)
    np.testing.assert_array_equal(transform.transform(block), transform.transform(snapped))

    orthonormal = hadamard_matrix(8) @ block @ hadamard_matrix(8)
    np.testing.assert_allclose(transform.transform(block), orthonormal, atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    ("size", "fraction_bits"), [(1, 36), (2, 34), (8, 30), (16, 28), (128, 22)]
)
def test_grid_leaves_room_for_the_largest_possible_sum(size: int, fraction_bits: int) -> None:
    """53 significand bits, less 2 of margin, 15 for the sample magnitude and
    two per doubling of the edge for the n*n-term sums."""
    from walsh.transforms import _grid_scale

    assert _grid_scale(size) == 2.0**fraction_bits


def test_sign_matrix_shares_the_orthonormal_matrix_row_order() -> None:
    from walsh.transforms import _hadamard_signs

    signs = _hadamard_signs(16)
    assert set(np.unique(signs)) == {-1.0, 1.0}
    np.testing.assert_array_equal(signs, np.sign(hadamard_matrix(16)))
    assert _hadamard_signs(16) is signs, "memoised, like hadamard_matrix"
