"""The Walsh-Hadamard transform itself."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from walsh.decorators import cached

__all__ = ["Transform", "WalshHadamardTransform"]

Block = npt.NDArray[np.float64]

#: Tolerance used when comparing a matrix entry against the removal coefficient.
COEFF_TOLERANCE = 0.000001


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


class WalshHadamardTransform(Transform):
    """Orthonormal Walsh-Hadamard transform in sequency (Walsh) order.

    The transform is symmetric and involutive, so :meth:`inverse_transform`
    is simply :meth:`transform` applied again.

    :param coeff: optional coefficient-removal threshold. When set, matrix
        entries that fall at or below it during construction are zeroed,
        which discards part of the spectrum. See :meth:`_build_matrix`.
    """

    def __init__(self, coeff: float | None = None) -> None:
        """Configure the transform.

        Args:
            coeff: Optional coefficient-removal threshold. ``None``, the
                default, leaves the matrix intact.
        """
        self._coeff = coeff

    @cached
    def _build_matrix(self, size: int) -> Block:
        """Build the ``size`` x ``size`` Walsh-Hadamard matrix.

        Starts from a uniform matrix of ``1 / sqrt(2) ** n`` and negates the
        entries whose row and column indices share a set bit, which yields the
        natural (Hadamard) ordering. Rows are then sorted by the number of
        sign changes to reach sequency (Walsh) ordering.

        Memoised, so the cost is paid once per size per instance.

        Args:
            size: Edge length of the matrix. Must be a power of two.

        Returns:
            The ``size`` by ``size`` orthonormal matrix, rows in sequency
            order.
        """
        n = int(math.log(size, 2))
        matrix = np.full((size, size), 1 / (np.sqrt(2) ** n), dtype=np.float64)

        for i in range(n):
            for j in range(size):
                for k in range(size):
                    if (j // 2**i) % 2 == 1 and (k // 2**i) % 2 == 1:
                        matrix[j, k] = -matrix[j, k]
                        # NOTE: preserved verbatim from the original Python 2
                        # implementation. The comparison is one-sided rather
                        # than on the magnitude, so in practice any non-None
                        # coeff above -1/sqrt(2)**n zeroes every entry that
                        # ever gets negated. Changing it changes output.
                        if self._coeff is not None and matrix[j, k] - self._coeff < COEFF_TOLERANCE:
                            matrix[j, k] = 0

        return matrix[np.argsort(_sign_changes(matrix), kind="stable")]

    def transform(self, src: Block) -> Block:
        """Apply the transform to one square block.

        Args:
            src: A square 2-D array of samples.

        Returns:
            ``h @ src @ h``, where ``h`` is the sequency-ordered matrix of the
            block's size.

        Raises:
            ValueError: If ``src`` is not a square 2-D array.
        """
        src = np.asarray(src, dtype=np.float64)
        if src.ndim != 2 or src.shape[0] != src.shape[1]:
            raise ValueError(f"expected a square block, got shape {src.shape}")

        h = self._build_matrix(src.shape[0])
        return h @ src @ h

    def inverse_transform(self, src: Block) -> Block:
        """Invert the transform, which for this transform means reapplying it.

        The matrix is orthonormal and symmetric, so it is its own inverse.

        Args:
            src: A square 2-D spectrum.

        Returns:
            The reconstructed block.

        Raises:
            ValueError: If ``src`` is not a square 2-D array.
        """
        return self.transform(src)


def _sign_changes(matrix: Block) -> npt.NDArray[np.int64]:
    """Count sign changes along each row, which is that row's sequency.

    Args:
        matrix: The matrix whose rows to measure.

    Returns:
        One count per row, in row order.
    """
    counts: npt.NDArray[np.int64] = np.count_nonzero(matrix[:, 1:] * matrix[:, :-1] < 0, axis=1)
    return counts
