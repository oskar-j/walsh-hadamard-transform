"""The Walsh-Hadamard transform itself."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from walsh.decorators import cached

__all__ = ["Transform", "WalshHadamardTransform", "hadamard_matrix"]

Block = npt.NDArray[np.float64]

#: Sylvester's construction doubles a Hadamard matrix by taking its Kronecker
#: product with this seed: ``[[H, H], [H, -H]]``.
_SYLVESTER_SEED = np.array([[1.0, 1.0], [1.0, -1.0]])


class Transform(ABC):
    """A block transform and its inverse."""

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
    bit, which is how the pre-0.3.3 triple loop built it. Rows are then sorted
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
                magnitude. ``None``, the default, keeps every coefficient.

        Raises:
            ValueError: If ``coeff`` is negative. It is compared against a
                magnitude, so a negative value could only be a mistake -- it
                would silently keep everything.
        """
        if coeff is not None and coeff < 0:
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
        return np.where(np.abs(spectrum) < self._coeff, 0.0, spectrum)

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
        """Apply the matrix on both sides of ``src``.

        Args:
            src: A square 2-D array, or a 3-D stack of them along the first
                axis.

        Returns:
            ``h @ src @ h`` for the matrix of the block edge, the same shape
            as ``src``. For a stack, ``@`` broadcasts over the leading axis.

        Raises:
            ValueError: If ``src`` is not a square block or a stack of them.
        """
        src = np.asarray(src, dtype=np.float64)
        if src.ndim not in (2, 3) or src.shape[-1] != src.shape[-2]:
            raise ValueError(f"expected a square block or a stack of them, got shape {src.shape}")

        h = hadamard_matrix(src.shape[-1])
        return h @ src @ h


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
