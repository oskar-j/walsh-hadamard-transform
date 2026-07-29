"""The Walsh-Hadamard transform itself."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from walsh.decorators import cached

__all__ = ["Transform", "WalshHadamardTransform", "hadamard_matrix"]

Block = npt.NDArray[np.float64]


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

    Starts from a uniform matrix of ``1 / sqrt(2) ** n`` and negates the
    entries whose row and column indices share a set bit, which yields the
    natural (Hadamard) ordering. Rows are then sorted by the number of sign
    changes to reach sequency (Walsh) ordering.

    Memoised on ``size`` alone. It is a module-level function rather than a
    method precisely so the memo cannot hold a transform instance alive.

    Args:
        size: Edge length of the matrix. Must be a power of two.

    Returns:
        The ``size`` by ``size`` orthonormal matrix, rows in sequency order.
    """
    n = int(math.log(size, 2))
    matrix = np.full((size, size), 1 / (np.sqrt(2) ** n), dtype=np.float64)

    for i in range(n):
        for j in range(size):
            for k in range(size):
                if (j // 2**i) % 2 == 1 and (k // 2**i) % 2 == 1:
                    matrix[j, k] = -matrix[j, k]

    return matrix[np.argsort(_sign_changes(matrix), kind="stable")]


class WalshHadamardTransform(Transform):
    """Orthonormal Walsh-Hadamard transform in sequency (Walsh) order.

    Without ``coeff`` the transform is symmetric and involutive, so
    :meth:`inverse_transform` is :meth:`transform` applied again.

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
            src: A square 2-D array of samples.

        Returns:
            ``h @ src @ h``, with coefficients below ``coeff`` in magnitude
            zeroed when a threshold was configured.

        Raises:
            ValueError: If ``src`` is not a square 2-D array.
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
            src: A square 2-D spectrum.

        Returns:
            The reconstructed block.

        Raises:
            ValueError: If ``src`` is not a square 2-D array.
        """
        return self._spectrum(src)

    @staticmethod
    def _spectrum(src: Block) -> Block:
        """Apply the matrix on both sides of ``src``.

        Args:
            src: A square 2-D array.

        Returns:
            ``h @ src @ h`` for the matrix of ``src``'s size.

        Raises:
            ValueError: If ``src`` is not a square 2-D array.
        """
        src = np.asarray(src, dtype=np.float64)
        if src.ndim != 2 or src.shape[0] != src.shape[1]:
            raise ValueError(f"expected a square block, got shape {src.shape}")

        h = hadamard_matrix(src.shape[0])
        return h @ src @ h


def _sign_changes(matrix: Block) -> npt.NDArray[np.int64]:
    """Count sign changes along each row, which is that row's sequency.

    Args:
        matrix: The matrix whose rows to measure.

    Returns:
        One count per row, in row order.
    """
    counts: npt.NDArray[np.int64] = np.count_nonzero(matrix[:, 1:] * matrix[:, :-1] < 0, axis=1)
    return counts
