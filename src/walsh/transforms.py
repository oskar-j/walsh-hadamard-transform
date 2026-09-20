"""The Walsh-Hadamard transform itself.

Since 0.4.12 the transform is *exact*: every floating-point operation it
performs has an exactly representable result, so its output is a function of
its input bits alone and is identical on every platform and BLAS library. See
:meth:`WalshHadamardTransform._spectrum` for how, and #39 for why.

Two more transforms ship for comparison (0.4.14), a DCT-II and a Haar
transform, and :func:`transform_for` resolves all three by name. They are
ordinary floating-point matrix products: accurate to rounding, but only the
Walsh-Hadamard transform carries the bit-exactness guarantee, because their
matrices are irrational and every product rounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import numpy.typing as npt

from walsh.decorators import cached

__all__ = [
    "TRANSFORMS",
    "DiscreteCosineTransform",
    "HaarTransform",
    "MatrixTransform",
    "Transform",
    "WalshHadamardTransform",
    "dct_matrix",
    "haar_matrix",
    "hadamard_matrix",
    "remove_small_coefficients",
    "transform_for",
]

Block = npt.NDArray[np.float64]

#: Sylvester's construction doubles a Hadamard matrix by taking its Kronecker
#: product with this seed: ``[[H, H], [H, -H]]``.
_SYLVESTER_SEED = np.array([[1.0, 1.0], [1.0, -1.0]])


class Transform(ABC):
    """A block transform and its inverse.

    Subclass this to run another transform through the codec's pipeline with
    ``Codec(transform=...)``: a DCT or a Haar transform at the same block
    geometry, say, to compare against Walsh-Hadamard. Only :meth:`transform`
    and :meth:`inverse_transform` are required, and they see one square block
    at a time. :meth:`transform_stack` and :meth:`inverse_transform_stack` are
    what :class:`~walsh.codec.Codec` actually calls, with every block of a
    channel at once; their defaults loop over the blocks, which is correct for
    any subclass and slow on a large image, so override them when the
    transform can take a whole ``(count, edge, edge)`` stack in one operation.

    Two things the ``.cim`` container asks of a transform. A block must come
    back the same shape it went in. And coefficients are stored rounded to
    ``int16``: an orthonormal transform of 8-bit samples cannot exceed
    ``edge * 255``, which fits at every edge ``Codec`` accepts, but a transform
    with a larger gain will saturate silently.
    """

    @abstractmethod
    def transform(self, src: Block) -> Block:
        """Transform a single square block.

        Args:
            src: A square 2-D array of samples.

        Returns:
            The block's spectrum, the same shape as ``src``.
        """

    @abstractmethod
    def inverse_transform(self, src: Block) -> Block:
        """Invert :meth:`transform` for a single square block.

        Args:
            src: A square 2-D spectrum.

        Returns:
            The reconstructed samples, the same shape as ``src``.
        """

    def transform_stack(self, stack: Block) -> Block:
        """Transform every block of a ``(count, edge, edge)`` stack.

        The default calls :meth:`transform` once per block. Override it when
        the transform can take the whole stack in one array operation.

        Args:
            stack: The blocks, stacked along the first axis.

        Returns:
            The spectra, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not 3-D, or :meth:`transform` returns
                a block of a different shape from the one it was given.
        """
        return _per_block(self.transform, stack, f"{type(self).__name__}.transform")

    def inverse_transform_stack(self, stack: Block) -> Block:
        """Invert the transform for every block of a ``(count, edge, edge)`` stack.

        The default calls :meth:`inverse_transform` once per block. See
        :meth:`transform_stack`.

        Args:
            stack: The spectra, stacked along the first axis.

        Returns:
            The reconstructed blocks, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not 3-D, or :meth:`inverse_transform`
                returns a block of a different shape from the one it was given.
        """
        return _per_block(self.inverse_transform, stack, f"{type(self).__name__}.inverse_transform")

    def transform_sequence(self, src_seq: Iterable[Block]) -> list[Block]:
        """Transform every block in a sequence.

        Args:
            src_seq: The blocks to transform.

        Returns:
            One spectrum per input block, in the same order. A list rather
            than an iterator, because callers take its length.
        """
        return [self.transform(block) for block in src_seq]

    def inverse_transform_sequence(self, src_seq: Iterable[Block]) -> list[Block]:
        """Invert the transform for every block in a sequence.

        Args:
            src_seq: The spectra to invert.

        Returns:
            One reconstructed block per input, in the same order.
        """
        return [self.inverse_transform(block) for block in src_seq]


@cached
def hadamard_matrix(size: int) -> Block:
    """Build the ``size`` x ``size`` Walsh-Hadamard matrix.

    Uses Sylvester's construction: starting from ``[[1]]``, each doubling is
    the Kronecker product ``[[H, H], [H, -H]]``. That yields the natural
    (Hadamard) ordering, in which entry ``(j, k)`` is negative exactly when
    ``j & k`` has an odd number of set bits -- the same matrix as negating,
    one bit at a time, every entry whose row and column indices share that
    bit, which is how the pre-0.4.0 triple loop built it. Rows are then sorted
    by their number of sign changes to reach sequency (Walsh) ordering.

    Every step is a whole-array numpy operation, so the build is ``log2(size)``
    Kronecker products rather than ``size ** 2 * log2(size)`` Python
    iterations. The scale is applied by multiplication, so every entry is
    bit-identical to the loop's ``+scale`` or ``-scale``.

    Memoised on ``size`` alone. It is a module-level function rather than a
    method precisely so the memo cannot hold a transform instance alive.

    Args:
        size: Edge length of the matrix. Must be a power of two.

    Returns:
        The ``size`` by ``size`` orthonormal matrix, rows in sequency order.

    Raises:
        ValueError: If ``size`` is not a positive power of two. The
            construction can only produce power-of-two edges, so any other
            size would come back the wrong shape rather than merely wrong.
    """
    if size < 1 or size & (size - 1):
        raise ValueError(f"size must be a power of two, got {size}")
    n = size.bit_length() - 1

    matrix = np.ones((1, 1), dtype=np.float64)
    for _ in range(n):
        matrix = np.kron(_SYLVESTER_SEED, matrix)
    matrix *= 1 / (np.sqrt(2) ** n)

    return matrix[np.argsort(_sign_changes(matrix), kind="stable")]


class WalshHadamardTransform(Transform):
    """Orthonormal Walsh-Hadamard transform in sequency (Walsh) order.

    Without ``coeff`` the transform is symmetric and involutive, so
    :meth:`inverse_transform` is :meth:`transform` applied again.

    The arithmetic is exact (0.4.12, #39): the input is snapped to a binary
    grid, multiplied by the ``+-1`` sign matrix on both sides, and scaled by
    ``1 / n`` at the end. Each of those steps is exact in IEEE double for
    samples of magnitude below ``2 ** 15`` at any block edge the codec accepts,
    so the result does not depend on the order in which the platform's BLAS
    sums the products. The output is therefore bit-identical everywhere, and
    a coefficient whose true value is an integer comes out as exactly that
    integer -- which matters on the way back, where the colour conversion
    truncates and ``1.999999999999999`` is a level lower than ``2.0``.

    Both accept a single square block or a 3-D stack of them, and the
    sequence methods use that to push every block of an image through one
    broadcast matrix product instead of one Python call each.

    :param coeff: optional coefficient-removal threshold. Spectral coefficients
        whose magnitude falls below it are zeroed by :meth:`transform`. This is
        a lossy step, so :meth:`inverse_transform` does not apply it, and the
        two stop being an exact round trip once it is set.
    """

    def __init__(self, coeff: float | None = None) -> None:
        """Configure the transform.

        Args:
            coeff: Optional coefficient-removal threshold, as an absolute
                magnitude: a spectral coefficient strictly below it is zeroed.
                ``None``, the default, keeps every coefficient. Because it is
                absolute and the surviving low-frequency coefficients grow with
                the block edge, one value prunes less at a larger block size;
                tune it for the block size in use.

        Raises:
            ValueError: If ``coeff`` is negative or NaN. It is compared against
                a magnitude, so either value could only be a mistake -- it
                would silently keep everything.
        """
        if coeff is not None and not coeff >= 0:
            raise ValueError(f"coeff must be non-negative, got {coeff}")
        self._coeff = coeff

    def _build_matrix(self, size: int) -> Block:
        """Return the sequency-ordered matrix for ``size``.

        Retained so existing callers keep working; the construction and the
        memo both live in :func:`hadamard_matrix`, which does not depend on
        this instance.

        Args:
            size: Edge length of the matrix. Must be a power of two.

        Returns:
            The ``size`` by ``size`` orthonormal matrix, rows in sequency order.
        """
        return hadamard_matrix(size)

    def transform(self, src: Block) -> Block:
        """Transform one square block, then drop its smallest coefficients.

        Args:
            src: A square 2-D array of samples, or a 3-D stack of them along
                the first axis.

        Returns:
            ``h @ src @ h``, with coefficients below ``coeff`` in magnitude
            zeroed when a threshold was configured. The same shape as ``src``.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        spectrum = self._spectrum(src)
        if self._coeff is None:
            return spectrum
        return remove_small_coefficients(spectrum, self._coeff)

    def inverse_transform(self, src: Block) -> Block:
        """Invert the transform by applying the matrix again.

        The matrix is orthonormal and symmetric, so it is its own inverse.
        Coefficient removal is deliberately *not* repeated here: it is the
        lossy step, and applying it on the way back would discard reconstructed
        detail a second time.

        Args:
            src: A square 2-D spectrum, or a 3-D stack of them along the first
                axis.

        Returns:
            The reconstructed block, or stack of blocks.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        return self._spectrum(src)

    def transform_stack(self, stack: Block) -> Block:
        """Transform a whole stack in one broadcast matrix product.

        Args:
            stack: The blocks, stacked along the first axis.

        Returns:
            The spectra, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not a stack of square blocks.
        """
        return self.transform(stack)

    def inverse_transform_stack(self, stack: Block) -> Block:
        """Invert a whole stack in one broadcast matrix product.

        Args:
            stack: The spectra, stacked along the first axis.

        Returns:
            The reconstructed blocks, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not a stack of square blocks.
        """
        return self.inverse_transform(stack)

    def transform_sequence(self, src_seq: Iterable[Block]) -> list[Block]:
        """Transform every block in a sequence with one batched call.

        An image is thousands of small blocks, and for those the per-call
        Python overhead of :meth:`transform` outweighs its arithmetic several
        times over. Uniformly shaped blocks are therefore stacked and
        transformed together; mixed shapes, which the codec never produces,
        fall back to one call per block.

        Args:
            src_seq: The blocks to transform.

        Returns:
            One spectrum per input block, in the same order. A list rather
            than an iterator, because callers take its length.
        """
        return _apply_batched(self.transform, src_seq)

    def inverse_transform_sequence(self, src_seq: Iterable[Block]) -> list[Block]:
        """Invert the transform for every block in a sequence with one batched call.

        See :meth:`transform_sequence` for why.

        Args:
            src_seq: The spectra to invert.

        Returns:
            One reconstructed block per input, in the same order.
        """
        return _apply_batched(self.inverse_transform, src_seq)

    @staticmethod
    def _spectrum(src: Block) -> Block:
        """Apply the matrix on both sides of ``src``, exactly.

        Mathematically this is ``h @ src @ h`` for the orthonormal matrix of
        the block edge. It is computed as ``(s @ snap(src) @ s) / n`` with
        ``s`` the ``+-1`` sign matrix, and that form is exact:

        * ``snap`` rounds every sample to a multiple of ``2 ** -f`` with
          ``f = 36 - 2 * log2(n)``, chosen so that a sum of ``n * n`` samples
          of magnitude below ``2 ** 15`` still fits in the 53 significand
          bits of a double, with two bits to spare. Rounding to and from a
          power of two is itself exact.
        * A product with ``+-1`` is exact, and every partial sum of the two
          matrix products is a sum of grid values within that bound, so it is
          exact whatever order the BLAS adds them in, with or without fused
          multiply-add.
        * ``1 / n`` is a power of two, so the final scale is exact.

        So the result is the exact rational transform of the snapped input,
        identical on every platform. The orthonormal entries ``+-1/sqrt(n)``
        are irrational and must not be used here: multiplying by them rounds
        at every step, and the rounding depends on the summation order.

        Args:
            src: A square 2-D array, or a 3-D stack of them along the first
                axis. Samples of magnitude ``2 ** 15`` or more are still
                transformed, but the exactness guarantee stops there.

        Returns:
            The spectrum, the same shape as ``src``. For a stack, ``@``
            broadcasts over the leading axis.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        src = _square(src)
        size = src.shape[-1]
        signs = _hadamard_signs(size)
        grid = _grid_scale(size)
        snapped: Block = np.rint(src * grid) / grid
        spectrum: Block = (signs @ snapped @ signs) * (1.0 / size)
        return spectrum


#: Magnitude bound, as a power of two, on the samples the exactness guarantee
#: covers: pixels are below ``2 ** 8`` and ``.cim`` coefficients below
#: ``2 ** 15``.
_SAMPLE_BITS = 15

#: Bits of the 53-bit double significand kept in reserve beyond the bound.
_MARGIN_BITS = 2


def _grid_scale(size: int) -> float:
    """Return ``2 ** f``, the reciprocal of the snapping grid for ``size``.

    A sum of ``size * size`` samples below ``2 ** _SAMPLE_BITS``, each a
    multiple of ``2 ** -f``, is below ``2 ** (_SAMPLE_BITS + 2 * log2(size))``
    and must be representable in 53 bits: ``f`` is what is left.

    Args:
        size: Edge length of the block. Must be a power of two.

    Returns:
        The scale as a float, exactly a power of two.
    """
    log2_size = size.bit_length() - 1
    fraction_bits = 53 - _MARGIN_BITS - _SAMPLE_BITS - 2 * log2_size
    return float(2**fraction_bits)


@cached
def _hadamard_signs(size: int) -> Block:
    """Return the ``+-1`` matrix with the rows of :func:`hadamard_matrix`.

    Memoised for the same reason and in the same way as
    :func:`hadamard_matrix`, from which it takes its row order.

    Args:
        size: Edge length of the matrix. Must be a power of two.

    Returns:
        The ``size`` by ``size`` matrix of ``+1.0`` and ``-1.0``, rows in
        sequency order.
    """
    return np.sign(hadamard_matrix(size)).astype(np.float64)


def remove_small_coefficients(spectrum: Block, coeff: float) -> Block:
    """Zero every spectral coefficient whose magnitude is strictly below ``coeff``.

    The codec's second lossy knob, as one function so that
    :class:`WalshHadamardTransform` and :class:`~walsh.codec.Codec`, which
    applies it after whatever transform it was given, cannot drift apart.

    Args:
        spectrum: Spectral coefficients, any shape.
        coeff: The threshold, as an absolute magnitude. A coefficient exactly
            equal to it is kept.

    Returns:
        A new array the same shape as ``spectrum``.
    """
    thinned: Block = np.where(np.abs(spectrum) < coeff, 0.0, spectrum)
    return thinned


def _per_block(apply: Callable[[Block], Block], stack: Block, name: str) -> Block:
    """Run a single-block operation over every block of a stack.

    Args:
        apply: Takes one square block and returns one of the same shape.
        stack: The blocks, stacked along the first axis.
        name: What to call ``apply`` in an error message.

    Returns:
        The results, the same shape as ``stack``. An empty stack comes back
        empty without ``apply`` being called.

    Raises:
        ValueError: If ``stack`` is not 3-D, or ``apply`` returns a block of a
            different shape. The shape is compared rather than left to
            broadcasting, which would quietly spread a scalar over the block.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a (count, edge, edge) stack, got shape {stack.shape}")
    result = np.empty_like(stack)
    for index, block in enumerate(stack):
        transformed = np.asarray(apply(block), dtype=np.float64)
        if transformed.shape != block.shape:
            raise ValueError(
                f"{name} returned shape {transformed.shape} for a block of shape {block.shape}"
            )
        result[index] = transformed
    return result


def _apply_batched(apply: Callable[[Block], Block], src_seq: Iterable[Block]) -> list[Block]:
    """Run a stack-aware block operation over a sequence of blocks.

    Args:
        apply: Accepts one square block or a 3-D stack of them and returns the
            same shape.
        src_seq: The blocks.

    Returns:
        One result per input block, in order. Uniformly shaped blocks go
        through ``apply`` as a single stack, and the list then holds views
        into that one result array. Mixed shapes are applied one at a time.
    """
    blocks = [np.asarray(block, dtype=np.float64) for block in src_seq]
    if not blocks:
        return []
    if len({block.shape for block in blocks}) == 1:
        return list(apply(np.stack(blocks)))
    return [apply(block) for block in blocks]


def _sign_changes(matrix: Block) -> npt.NDArray[np.int64]:
    """Count sign changes along each row, which is that row's sequency.

    Args:
        matrix: The matrix whose rows to measure.

    Returns:
        One count per row, in row order.
    """
    counts: npt.NDArray[np.int64] = np.count_nonzero(matrix[:, 1:] * matrix[:, :-1] < 0, axis=1)
    return counts


def _square(src: Block) -> Block:
    """Return ``src`` as ``float64``, insisting it is square blocks.

    Args:
        src: A square 2-D array, or a 3-D stack of them along the first axis.

    Returns:
        ``src`` as a ``float64`` array.

    Raises:
        ValueError: If ``src`` is not a square block or a stack of them.
    """
    array = np.asarray(src, dtype=np.float64)
    if array.ndim not in (2, 3) or array.shape[-1] != array.shape[-2]:
        raise ValueError(f"expected a square block or a stack of them, got shape {array.shape}")
    return array


# -- transforms shipped for comparison (0.4.14) -------------------------------


class MatrixTransform(Transform):
    """A separable orthonormal transform defined by one ``size x size`` matrix.

    ``transform`` is ``m @ block @ m.T`` and, the matrix being orthonormal, the
    inverse is ``m.T @ spectrum @ m``. A subclass supplies :meth:`matrix` and
    nothing else. ``@`` broadcasts, so a stack goes through in one product.

    These are plain floating-point products. Unlike
    :class:`WalshHadamardTransform` they are not bit-exact across platforms:
    an irrational matrix makes every product round, and the rounding follows
    the BLAS library's summation order.
    """

    @abstractmethod
    def matrix(self, size: int) -> Block:
        """Return the orthonormal matrix for blocks of edge ``size``.

        Args:
            size: Edge length of the block.

        Returns:
            A ``size`` by ``size`` matrix whose rows are orthonormal. It is
            called once per transform call, so memoise anything expensive.
        """

    def transform(self, src: Block) -> Block:
        """Transform one square block, or a stack of them.

        Args:
            src: A square 2-D array of samples, or a 3-D stack of them along
                the first axis.

        Returns:
            ``m @ src @ m.T``, the same shape as ``src``.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        src = _square(src)
        m = self.matrix(src.shape[-1])
        spectrum: Block = m @ src @ m.T
        return spectrum

    def inverse_transform(self, src: Block) -> Block:
        """Invert :meth:`transform` with the transposed matrix.

        Args:
            src: A square 2-D spectrum, or a 3-D stack of them along the first
                axis.

        Returns:
            ``m.T @ src @ m``, the same shape as ``src``.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        src = _square(src)
        m = self.matrix(src.shape[-1])
        restored: Block = m.T @ src @ m
        return restored

    def transform_stack(self, stack: Block) -> Block:
        """Transform a whole stack in one broadcast matrix product.

        Args:
            stack: The blocks, stacked along the first axis.

        Returns:
            The spectra, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not a stack of square blocks.
        """
        return self.transform(stack)

    def inverse_transform_stack(self, stack: Block) -> Block:
        """Invert a whole stack in one broadcast matrix product.

        Args:
            stack: The spectra, stacked along the first axis.

        Returns:
            The reconstructed blocks, the same shape as ``stack``.

        Raises:
            ValueError: If ``stack`` is not a stack of square blocks.
        """
        return self.inverse_transform(stack)


@cached
def dct_matrix(size: int) -> Block:
    """Build the orthonormal DCT-II matrix, the transform inside JPEG.

    Row ``k`` is ``cos(pi * (2 * i + 1) * k / (2 * size))`` over the columns
    ``i``, scaled to unit length, so row 0 is constant and the rows rise in
    frequency. Memoised on ``size``; the array is read-only because every
    caller shares it.

    Args:
        size: Edge length of the matrix. Any positive integer.

    Returns:
        The ``size`` by ``size`` orthonormal matrix.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    k = np.arange(size, dtype=np.float64)[:, None]
    i = np.arange(size, dtype=np.float64)[None, :]
    matrix: Block = np.cos(np.pi * (2 * i + 1) * k / (2 * size)) * np.sqrt(2 / size)
    matrix[0] /= np.sqrt(2)
    matrix.flags.writeable = False
    return matrix


@cached
def haar_matrix(size: int) -> Block:
    """Build the orthonormal Haar wavelet matrix, rows from coarse to fine.

    Each doubling keeps the rows so far, stretched to twice the width, and
    appends one ``[1, -1]`` difference per pair of samples; rows are then
    scaled to unit length. Row 0 is constant, like the other transforms here.
    Memoised on ``size``; the array is read-only because every caller shares
    it.

    Args:
        size: Edge length of the matrix. Must be a power of two.

    Returns:
        The ``size`` by ``size`` orthonormal matrix.

    Raises:
        ValueError: If ``size`` is not a positive power of two.
    """
    if size < 1 or size & (size - 1):
        raise ValueError(f"size must be a power of two, got {size}")
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < size:
        coarse = np.kron(matrix, np.array([1.0, 1.0]))
        fine = np.kron(np.eye(matrix.shape[0]), np.array([1.0, -1.0]))
        matrix = np.vstack([coarse, fine])
    normalised: Block = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised.flags.writeable = False
    return normalised


class DiscreteCosineTransform(MatrixTransform):
    """The orthonormal DCT-II. Packs more of a smooth picture into its first
    coefficients than the Walsh-Hadamard transform, at the cost of
    multiplications and of bit-exactness."""

    def matrix(self, size: int) -> Block:
        """Return :func:`dct_matrix` for ``size``.

        Args:
            size: Edge length of the block.

        Returns:
            The ``size`` by ``size`` orthonormal DCT-II matrix.
        """
        return dct_matrix(size)


class HaarTransform(MatrixTransform):
    """The orthonormal Haar wavelet transform.

    Keeping the first ``p`` coefficients per axis gives the same picture as
    the Walsh-Hadamard transform whenever ``p`` is a power of two: the first
    ``p`` Walsh functions and the first ``p`` Haar functions span the same
    piecewise-constant subspace. They differ at any other ``p``.
    """

    def matrix(self, size: int) -> Block:
        """Return :func:`haar_matrix` for ``size``.

        Args:
            size: Edge length of the block. Must be a power of two.

        Returns:
            The ``size`` by ``size`` orthonormal Haar matrix.

        Raises:
            ValueError: If ``size`` is not a positive power of two.
        """
        return haar_matrix(size)


#: The transforms :func:`transform_for` knows, by name. ``"walsh"`` is the
#: codec's own and the only one the ``.cim`` format and the ``walsh`` command
#: mean; the others are for comparison.
TRANSFORMS: Mapping[str, type[Transform]] = {
    "dct": DiscreteCosineTransform,
    "haar": HaarTransform,
    "walsh": WalshHadamardTransform,
}


def transform_for(name: str) -> Transform:
    """Return a new instance of the transform called ``name``.

    Args:
        name: One of the keys of :data:`TRANSFORMS`, in any case.

    Returns:
        A fresh instance of that transform.

    Raises:
        ValueError: If ``name`` is not a known transform. The message lists
            the names that are.
    """
    try:
        transform_class = TRANSFORMS[name.lower()]
    except KeyError:
        known = ", ".join(sorted(TRANSFORMS))
        raise ValueError(f"unknown transform {name!r}; known transforms: {known}") from None
    return transform_class()
