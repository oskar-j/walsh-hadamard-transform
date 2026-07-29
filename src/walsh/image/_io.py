"""Shared byte-level helpers for the raster format readers and writers."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from os import PathLike
from typing import BinaryIO

__all__ = ["FileSource", "align", "open_binary", "open_binary_read", "open_binary_write"]

#: Where an image is read from or written to. ``None`` means the standard
#: streams, which lets the codec be used in a shell pipeline.
FileSource = str | PathLike[str] | None


def align(x: int, a: int) -> int:
    """Round ``x`` up to the next multiple of ``a``.

    Args:
        x: The value to round. Assumed non-negative.
        a: The alignment. Must be positive.

    Returns:
        The smallest multiple of ``a`` that is greater than or equal to ``x``.
    """
    return (((x - 1) // a) + 1) * a


@contextlib.contextmanager
def open_binary_read(source: FileSource) -> Generator[BinaryIO, None, None]:
    """Open ``source`` for binary reading, or fall back to stdin.

    Args:
        source: Path to open, or ``None`` to read from stdin.

    Yields:
        The open binary stream. Standard input is deliberately not closed.

    Raises:
        OSError: If ``source`` names a file that cannot be opened.
    """
    if source is None:
        yield sys.stdin.buffer
        return
    with open(source, "rb") as handle:
        yield handle


@contextlib.contextmanager
def open_binary_write(source: FileSource) -> Generator[BinaryIO, None, None]:
    """Open ``source`` for binary writing, or fall back to stdout.

    Args:
        source: Path to open, or ``None`` to write to stdout.

    Yields:
        The open binary stream. Standard output is deliberately not closed.

    Raises:
        OSError: If ``source`` names a file that cannot be opened.
    """
    if source is None:
        yield sys.stdout.buffer
        return
    with open(source, "wb") as handle:
        yield handle


@contextlib.contextmanager
def open_binary(source: FileSource, mode: str) -> Generator[BinaryIO, None, None]:
    """Open ``source`` in binary mode, falling back to the standard streams.

    Kept for callers that predate the split into :func:`open_binary_read` and
    :func:`open_binary_write`. Prefer those: passing the mode as a string means
    neither the type checker nor the reader can tell which stream comes back,
    which is what forced a suppression here before.

    Args:
        source: Path to open, or ``None`` to use stdin/stdout.
        mode: Either a read mode or a write mode; the presence of ``"r"``
            selects stdin over stdout when ``source`` is ``None``.

    Yields:
        The open binary stream.

    Raises:
        OSError: If ``source`` names a file that cannot be opened.
    """
    opener = open_binary_read if "r" in mode else open_binary_write
    with opener(source) as handle:
        yield handle
