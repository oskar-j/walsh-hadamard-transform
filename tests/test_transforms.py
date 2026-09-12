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

    from walsh.transforms import hadamard_matrix

    hadamard_matrix.cache_clear()
    transform = WalshHadamardTransform()
    reference = weakref.ref(transform)
    transform.transform(np.zeros((8, 8)))

    del transform
    gc.collect()
    assert reference() is None, "the memo is holding the instance alive"

    for _ in range(20):
        WalshHadamardTransform().transform(np.zeros((8, 8)))
    assert hadamard_matrix.cache_size() == 1, "cache grows with instance count"


def test_non_square_input_is_rejected(transform: WalshHadamardTransform) -> None:
    with pytest.raises(ValueError, match="square"):
        transform.transform(np.zeros((4, 8)))


def test_transform_sequence_returns_a_list(transform: WalshHadamardTransform) -> None:
    """Python 3 note: this must not be a lazy map, callers take len() of it."""
    result = transform.transform_sequence([np.zeros((4, 4))] * 3)
    assert isinstance(result, list)
    assert len(result) == 3


def _reference_matrix(size: int) -> np.ndarray:
    """The pre-0.3.3 construction, kept verbatim as the oracle.

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
